#!/usr/bin/env python3
"""
equitrack_model_v6_cluster.py
EquiTrack v6 -- within-country school clustering to improve slope reliability.

Rationale:
  v5 computed ESCS slope from ~30 students/school (PISA sample size by design).
  This creates large measurement noise in the target variable -- XGBoost
  cannot distinguish true variation from sampling noise (error-in-variables).
  This is why v5's R2 is capped at ~0.21 despite 18,687 rows.

  Fix: cluster schools within each country using SC feature profiles (k-means,
  k=5 per country). Pool all students from each cluster and compute ESCS slope
  from ~150-300 pooled students instead of ~30. Noise scales as 1/sqrt(n) --
  10x more students => ~3x more reliable slope estimate.

  Tradeoff: fewer rows (~435 cluster rows vs 18,687 schools in v5), but each
  observation has a far more reliable target. Model is tuned for small-n:
  shallow trees, strong regularisation, LOCO CV as primary evaluation metric.

Metric: within-cluster OLS slope of MATH on ESCS (pooled students).
Features: mean SC values per cluster + country covariates + country FE.

Outputs:
  cache/equitrack_panel_v6.parquet
  reports/xgb_equity_slope_v6.json
  reports/shap_importance_v6.csv
  reports/intervention_ranking_v6.csv
  figures/shap_bar_v6.png
  figures/pred_vs_actual_v6.png
  figures/cluster_diagnostics_v6.png

Run:
  python equitrack_model_v6_cluster.py
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import linregress
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_score, train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb
import shap
from matplotlib.patches import Patch

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ── config ────────────────────────────────────────────────────────────────────
ROOT         = Path('d:/analytics_for_society')
CACHE        = ROOT / 'cache'
REPORTS      = ROOT / 'reports'
FIGURES      = ROOT / 'figures'
FIGURES.mkdir(exist_ok=True)

K_CLUSTERS          = 5    # target clusters per country
MIN_SCHOOLS_CLUSTER = 3    # each cluster must have at least this many schools
MIN_STUDENTS_SLOPE  = 30   # minimum pooled students to compute a reliable slope
RANDOM_STATE        = 42
TEST_SIZE           = 0.20

PV_COLS_2022 = [f'PV{i}MATH' for i in range(1, 11)]
PISA_MISSING = [9996, 9997, 9998, 9999, 996, 997, 998, 999, 96, 97, 98, 99]

POLICY_FEATURES = {
    'SC012Q01TA', 'SC012Q02TA', 'SC012Q03TA', 'SC012Q04TA',
    'SC012Q05TA', 'SC012Q06TA', 'SC004Q01TA',
    'SC002Q01TA', 'SC002Q02TA', 'SCHSIZE', 'SCHLTYPE',
}

COST_TIERS = {
    'TEAFDBK':    ('LOW', 1, 'Increase teacher feedback to students'),
    'MTTRAIN':    ('LOW', 1, 'Increase maths-specific teacher PD'),
    'NEGSCLIM':   ('LOW', 1, 'Reduce negative school climate'),
    'EDULEAD':    ('LOW', 1, 'Strengthen instructional leadership'),
    'INSTLEAD':   ('LOW', 1, 'Increase instructional leadership practices'),
    'STAFFSHORT': ('MED', 3, 'Reduce staff shortage'),
    'EDUSHORT':   ('MED', 3, 'Reduce educational resource shortage'),
    'STRATIO':    ('MED', 3, 'Reduce student-teacher ratio'),
    'CLSIZE':     ('MED', 3, 'Reduce average class size'),
    'RATCMP1':    ('MED', 3, 'Increase computers per student'),
    'SCHSEL':     ('HIGH', 9, 'Reduce academic selectivity'),
}


# ── helpers ───────────────────────────────────────────────────────────────────

def compute_escs_slope(df):
    """OLS slope of MATH_MEAN ~ ESCS from a pooled student DataFrame."""
    g = df.dropna(subset=['ESCS', 'MATH_MEAN'])
    if len(g) < MIN_STUDENTS_SLOPE:
        return np.nan
    slope, *_ = linregress(g['ESCS'].values, g['MATH_MEAN'].values)
    return float(slope)


# ── Phase 1: load 2022 student data ──────────────────────────────────────────

print('\n' + '='*62)
print('PHASE 1 -- Load 2022 student data')
print('='*62)

stu = pd.read_parquet(CACHE / 'pisa_2022.parquet')
stu['CNT']      = stu['CNT'].astype(str)
stu['CNTSCHID'] = stu['CNTSCHID'].astype(float)

pvs = [c for c in PV_COLS_2022 if c in stu.columns]
stu['MATH_MEAN'] = stu[pvs].mean(axis=1)

print(f'Students       : {len(stu):,}')
print(f'Schools        : {stu["CNTSCHID"].nunique():,}')
print(f'Countries      : {stu["CNT"].nunique()}')
print(f'PV cols used   : {len(pvs)}')

# ── Phase 2: load school questionnaire ───────────────────────────────────────

print('\n' + '='*62)
print('PHASE 2 -- Load school questionnaire features')
print('='*62)

sch = pd.read_parquet(CACHE / 'school_2022.parquet')
sch['CNT']      = sch['CNT'].astype(str)
sch['CNTSCHID'] = sch['CNTSCHID'].astype(float)

num_cols = sch.select_dtypes(include=[np.number]).columns
sch[num_cols] = sch[num_cols].replace(PISA_MISSING, np.nan)

sc_feature_cols = [c for c in sch.columns if c not in ('CNT', 'CNTSCHID')]

# filter to SC cols with <50% missing AND numeric dtype
# (categorical cols can't be averaged for cluster-level features or used in k-means)
miss_rate   = sch[sc_feature_cols].isna().mean()
sc_cols_all = miss_rate[miss_rate < 0.50].index.tolist()
sc_cols     = [c for c in sc_cols_all if sch[c].dtype.kind in ('f', 'i', 'u')]
n_dropped_cat = len(sc_cols_all) - len(sc_cols)
print(f'SC cols (raw)              : {len(sc_feature_cols)}')
print(f'SC cols retained (<50% NA) : {len(sc_cols_all)}')
print(f'SC cols dropped (category) : {n_dropped_cat}')
print(f'SC cols for model          : {len(sc_cols)}  (numeric only)')
print(f'School questionnaire       : {sch.shape[0]:,} schools x {len(sc_feature_cols)} SC vars')

# ── Phase 3: cluster schools within each country ──────────────────────────────

print('\n' + '='*62)
print('PHASE 3 -- K-means clustering within each country')
print(f'           k=min({K_CLUSTERS}, n_schools//{MIN_SCHOOLS_CLUSTER}) per country')
print('='*62)

# Median-impute SC matrix for clustering only (model still uses raw/NaN)
# sc_cols is already numeric-only (filtered in Phase 2)
sc_matrix = sch[sc_cols].copy().astype(np.float32)
col_medians = sc_matrix.median()
sc_matrix_imputed = sc_matrix.fillna(col_medians).values  # numpy array
print(f'SC cols for k-means        : {len(sc_cols)}')

sch = sch.copy()
sch['CLUSTER_ID'] = ''

cluster_stats = []
countries = sorted(sch['CNT'].unique())

for cnt in countries:
    mask      = (sch['CNT'] == cnt).values
    idx       = np.where(mask)[0]
    n_schools = mask.sum()
    k_use     = min(K_CLUSTERS, n_schools // MIN_SCHOOLS_CLUSTER)

    if k_use < 2:
        # country too small to split -- single cluster
        k_use = 1
        labels = np.zeros(n_schools, dtype=int)
    else:
        # sc_sub  = sc_matrix_imputed[idx]
        # scaler  = StandardScaler()
        # X_sc    = scaler.fit_transform(sc_sub)
        # within-country standardisation (replace global scaling)
        sc_sub = sc_matrix_imputed[idx]  # shape: (n_schools_country, n_features)

        # compute mean and std per feature within country
        mean_c = sc_sub.mean(axis=0)
        std_c  = sc_sub.std(axis=0)

        # avoid division by zero
        std_c[std_c == 0] = 1.0

        # standardise
        X_sc = (sc_sub - mean_c) / std_c
        km      = KMeans(n_clusters=k_use, random_state=RANDOM_STATE, n_init=10)
        labels  = km.fit_predict(X_sc)

    for i, ix in enumerate(idx):
        sch.iloc[ix, sch.columns.get_loc('CLUSTER_ID')] = f'{cnt}_{labels[i]}'

    cluster_stats.append({'CNT': cnt, 'n_schools': n_schools, 'k_used': k_use})

stats_df          = pd.DataFrame(cluster_stats)
n_clusters_total  = sch['CLUSTER_ID'].nunique()
avg_k             = stats_df['k_used'].mean()
avg_schools_per_c = stats_df['n_schools'].sum() / n_clusters_total
capped            = (stats_df['k_used'] < K_CLUSTERS).sum()

print(f'Countries processed        : {len(countries)}')
print(f'Total clusters created     : {n_clusters_total}')
print(f'Avg clusters per country   : {avg_k:.1f}')
print(f'Avg schools per cluster    : {avg_schools_per_c:.1f}')
print(f'Countries with k < {K_CLUSTERS}         : {capped}')

# ── Phase 4: aggregate students to cluster level ──────────────────────────────

print('\n' + '='*62)
print('PHASE 4 -- Pool students per cluster, compute slope')
print('='*62)

stu_c = stu.merge(
    sch[['CNT', 'CNTSCHID', 'CLUSTER_ID']],
    on=['CNT', 'CNTSCHID'],
    how='inner',
)
print(f'Students matched to cluster: {len(stu_c):,}')
print(f'Students unmatched         : {len(stu) - len(stu_c):,}')

cluster_rows = []
for cid, grp in stu_c.groupby('CLUSTER_ID'):
    cnt_val  = grp['CNT'].iloc[0]
    valid    = grp.dropna(subset=['ESCS', 'MATH_MEAN'])
    n_stu    = len(valid)
    slope    = compute_escs_slope(grp)

    mean_escs = valid['ESCS'].mean()    if n_stu > 0 else np.nan
    sd_escs   = valid['ESCS'].std()     if n_stu > 1 else np.nan
    mean_math = valid['MATH_MEAN'].mean() if n_stu > 0 else np.nan

    member_schools = sch[sch['CLUSTER_ID'] == cid]
    n_member       = len(member_schools)
    sc_means       = member_schools[sc_cols].mean()  # NaN where all members NaN

    row = {
        'CLUSTER_ID' : cid,
        'CNT'        : cnt_val,
        'N_SCHOOLS'  : n_member,
        'N_STUDENTS' : n_stu,
        'ESCS_SLOPE' : slope,
        'MEAN_ESCS'  : mean_escs,
        'SD_ESCS'    : sd_escs,
        'MEAN_MATH'  : mean_math,
    }
    for sc_col in sc_cols:
        row[sc_col] = sc_means[sc_col]

    cluster_rows.append(row)

panel   = pd.DataFrame(cluster_rows)
n_total = len(panel)
n_valid = panel['ESCS_SLOPE'].notna().sum()
pct     = 100 * n_valid / n_total

print(f'\nCluster rows               : {n_total}')
print(f'With valid slope           : {n_valid} ({pct:.1f}%)')
print(f'Dropped (<{MIN_STUDENTS_SLOPE} pooled students)  : {n_total - n_valid}')
print(f'Mean students per cluster  : {panel["N_STUDENTS"].mean():.0f}')
print(f'Median students per cluster: {panel["N_STUDENTS"].median():.0f}')
print(f'Std slope                  : {panel["ESCS_SLOPE"].std():.2f}')
print(f'Mean slope                 : {panel["ESCS_SLOPE"].mean():.2f} pts/unit ESCS')
print(f'Median slope               : {panel["ESCS_SLOPE"].median():.2f} pts/unit ESCS')

del stu, stu_c

# ── Phase 5: join country covariates ─────────────────────────────────────────

print('\n' + '='*62)
print('PHASE 5 -- Join country covariates')
print('='*62)

covars = pd.read_csv(REPORTS / 'country_covariates.csv')
if covars.columns[0] != 'CNT':
    covars = covars.rename(columns={covars.columns[0]: 'CNT'})

panel = panel.merge(covars, on='CNT', how='left')
panel['LOG_GDP_PC'] = np.log1p(panel['GDP_PC'])

print(f'GINI coverage  : {panel["GINI"].notna().mean():.0%}')
print(f'EDU_PCT cov.   : {panel["EDU_PCT"].notna().mean():.0%}')
print(f'GDP_PC cov.    : {panel["GDP_PC"].notna().mean():.0%}')

panel.to_parquet(CACHE / 'equitrack_panel_v6.parquet', index=False)
print(f'\nSaved: cache/equitrack_panel_v6.parquet  ({panel.shape[0]} x {panel.shape[1]} cols)')

# ── Phase 6: feature matrix ───────────────────────────────────────────────────

print('\n' + '='*62)
print('PHASE 6 -- Feature matrix')
print('='*62)

model_df = panel[panel['ESCS_SLOPE'].notna()].copy()

# covar_cols = ['GINI', 'EDU_PCT', 'LOG_GDP_PC']
# base_cols  = ['N_SCHOOLS', 'N_STUDENTS', 'MEAN_ESCS', 'SD_ESCS']
# groups     = model_df['CNT'].values  # for LOCO CV

# # country_dummies = pd.get_dummies(model_df['CNT'], prefix='CNT', drop_first=True, dtype=np.float32)

# feature_df = pd.concat([
#     model_df[base_cols + covar_cols + sc_cols].reset_index(drop=True),
#     # country_dummies.reset_index(drop=True),
# ], axis=1)

# X = feature_df.astype(np.float32)
# y = model_df['ESCS_SLOPE'].astype(np.float32).values

# print(f'Cluster rows   : {len(X)} (vs 18,687 schools in v5)')
# print(f'Features       : {X.shape[1]} total')
# print(f'  SC features  : {len(sc_cols)}')
# print(f'  Base         : {len(base_cols)} (N_SCHOOLS, N_STUDENTS, MEAN_ESCS, SD_ESCS)')
# print(f'  Covariates   : {len(covar_cols)} (GINI, EDU_PCT, LOG_GDP_PC)')
# # print(f'  Country FE   : {len(country_dummies.columns)}')
# print(f'Target         : mean={y.mean():.2f}  median={np.median(y):.2f}  std={y.std():.2f}')
# print(f'\nPrimary evaluation: Leave-One-Country-Out (LOCO) CV')
# print(f'Hold-out R2 shown for reference only (n_test ~{int(len(X)*TEST_SIZE)}).')
# ── Phase 6A: full feature matrix (for feature selection) ─────────────

print('\n[Stage 1] Building FULL feature matrix for SHAP selection')

base_cols  = ['N_SCHOOLS', 'N_STUDENTS', 'MEAN_ESCS', 'SD_ESCS']
covar_cols = ['GINI', 'EDU_PCT', 'LOG_GDP_PC']

full_features = base_cols + covar_cols + sc_cols

X_full = model_df[full_features].astype(np.float32)
y = model_df['ESCS_SLOPE'].astype(np.float32).values
groups = model_df['CNT'].values

print(f'Full feature count: {X_full.shape[1]}')


# ── Phase 7: XGBoost + LOCO CV ───────────────────────────────────────────────

print('\n' + '='*62)
print('PHASE 7 -- XGBoost model (small-n config: max_depth=3, strong reg)')
print('='*62)

xgb_params = dict(
    n_estimators     = 300,
    max_depth        = 3,
    learning_rate    = 0.02,
    subsample        = 0.8,
    colsample_bytree = 0.5,
    min_child_weight = 5,
    reg_alpha        = 1.0,
    reg_lambda       = 5.0,
    random_state     = RANDOM_STATE,
    n_jobs           = -1,
    tree_method      = 'hist',
)

# ── Phase 7A: train model for SHAP ───────────────────────────────────

model_stage1 = xgb.XGBRegressor(**xgb_params)
model_stage1.fit(X_full, y, verbose=False)

# ── Phase 8A: SHAP feature importance ───────────────────────────────

print('\n[Stage 1] Computing SHAP and selecting top features')

explainer = shap.TreeExplainer(model_stage1)
shap_values = explainer.shap_values(X_full)

feat_names = list(X_full.columns)
mean_abs_shap = np.abs(shap_values).mean(axis=0)

shap_df = pd.DataFrame({
    'feature': feat_names,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False)

# Select top 30 SC features
TOP_K = 30
non_sc = set(base_cols + covar_cols)

top_sc_features = (
    shap_df[~shap_df['feature'].isin(non_sc)]
    .head(TOP_K)['feature']
    .tolist()
)

print(f'\nSelected top {TOP_K} SC features:')
for f in top_sc_features:
    print(f'  {f}')


# ── Phase 6B: reduced feature matrix ────────────────────────────────

print('\n[Stage 2] Building REDUCED feature matrix')

selected_features = base_cols + covar_cols + top_sc_features

X = model_df[selected_features].astype(np.float32)

print(f'Reduced feature count: {X.shape[1]}')

# ── Phase 7B: final model evaluation ────────────────────────────────

print('\n[Stage 2] Final model with selected features')

model_final = xgb.XGBRegressor(**xgb_params)

# LOCO CV
loco_cv = GroupKFold(n_splits=len(np.unique(groups)))
loco_scores = cross_val_score(
    model_final, X, y,
    cv=loco_cv,
    scoring='r2',
    groups=groups,
    n_jobs=-1
)

print(f'LOCO CV R2     : {loco_scores.mean():.3f} +/- {loco_scores.std():.3f}')
print(f'Median LOCO R2 : {np.median(loco_scores):.3f}')
print(f'Best  LOCO R2  : {loco_scores.max():.3f}')
print(f'Worst LOCO R2  : {loco_scores.min():.3f}')

# 5-fold for reference
cv5_scores = cross_val_score(model_final, X, y, cv=5, scoring='r2', n_jobs=-1)
print(f'\n5-fold CV R2   : {cv5_scores.mean():.3f} +/- {cv5_scores.std():.3f}')

# model_cv = xgb.XGBRegressor(**xgb_params)

# # Leave-One-Country-Out (LOCO) -- primary metric
# n_countries_model = len(np.unique(groups))
# loco_cv    = GroupKFold(n_splits=n_countries_model)
# loco_scores = cross_val_score(model_cv, X, y, cv=loco_cv,
#                                scoring='r2', groups=groups, n_jobs=-1)

# print(f'LOCO CV R2     : {loco_scores.mean():.3f} +/- {loco_scores.std():.3f}  [PRIMARY]')
# print(f'LOCO folds     : {n_countries_model} (one per country)')
# print(f'Median LOCO R2 : {np.median(loco_scores):.3f}')
# print(f'Best  LOCO R2  : {loco_scores.max():.3f}')
# print(f'Worst LOCO R2  : {loco_scores.min():.3f}')

# # Standard 5-fold for direct comparison with v3-v5
# cv5_scores = cross_val_score(model_cv, X, y, cv=5, scoring='r2', n_jobs=-1)
# print(f'\n5-fold CV R2   : {cv5_scores.mean():.3f} +/- {cv5_scores.std():.3f}')

# Fit on train/test split for hold-out reference
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
model_holdout = xgb.XGBRegressor(**xgb_params)
model_holdout.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)

y_pred     = model_holdout.predict(X_test)
r2_holdout = r2_score(y_test, y_pred)
print(f'Hold-out R2    : {r2_holdout:.3f}  (n_test={len(y_test)}, reference only)')

model_holdout.save_model(str(REPORTS / 'xgb_equity_slope_v6.json'))
print('Saved: reports/xgb_equity_slope_v6.json')

# ── Phase 8: SHAP (refit on full data) ───────────────────────────────────────

print('\n' + '='*62)
print('PHASE 8 -- SHAP feature importance (full-data model)')
print('='*62)

model_full = xgb.XGBRegressor(**xgb_params)
model_full.fit(X, y, verbose=False)

explainer   = shap.TreeExplainer(model_full)
shap_values = explainer.shap_values(X)

feat_names    = list(X.columns)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
mean_shap     = shap_values.mean(axis=0)

shap_df = pd.DataFrame({
    'feature'       : feat_names,
    'mean_abs_shap' : mean_abs_shap,
    'mean_shap'     : mean_shap,
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

shap_df['is_policy'] = shap_df['feature'].isin(POLICY_FEATURES)
shap_df['direction'] = shap_df['mean_shap'].apply(
    lambda v: 'increases_slope' if v > 0 else 'reduces_slope'
)

shap_df.to_csv(REPORTS / 'shap_importance_v6.csv', index=False)
print('Saved: reports/shap_importance_v6.csv')

print('\nTop 20 features by |SHAP|:')
for _, row in shap_df.head(20).iterrows():
    tag   = '[POLICY]' if row['is_policy'] else ''
    arrow = ('increases slope (worse equity)'
             if row['direction'] == 'increases_slope'
             else 'reduces slope (better equity)')
    print(f"  {row['feature']:<22} |SHAP|={row['mean_abs_shap']:5.3f}  {arrow}  {tag}")

# SHAP bar chart
fig, ax = plt.subplots(figsize=(10, 8))
top25  = shap_df.head(25)
colors = ['#d45f5f' if d == 'increases_slope' else '#4c8fbd'
          for d in top25['direction']]
ax.barh(range(len(top25)), top25['mean_abs_shap'], color=colors)
ax.set_yticks(range(len(top25)))
ax.set_yticklabels(top25['feature'], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Mean |SHAP| value (cluster-level model)')
ax.set_title('v6 (clustered schools): Top 25 features\n'
             'red=increases slope (worse equity)   blue=reduces slope (better equity)')
ax.legend(handles=[
    Patch(facecolor='#d45f5f', label='Increases slope (worse equity)'),
    Patch(facecolor='#4c8fbd', label='Reduces slope (better equity)'),
], loc='lower right')
plt.tight_layout()
plt.savefig(FIGURES / 'shap_bar_v6.png', dpi=130)
plt.close()
print('Saved: figures/shap_bar_v6.png')

# ── Phase 9: figures ─────────────────────────────────────────────────────────

# pred vs actual + cluster size
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax1 = axes[0]
ax1.scatter(y_test, y_pred, alpha=0.4, s=50, color='#4c8fbd')
lim = [min(y_test.min(), y_pred.min()) - 5, max(y_test.max(), y_pred.max()) + 5]
ax1.plot(lim, lim, 'r--', linewidth=1, label='Perfect prediction')
ax1.set_xlim(lim); ax1.set_ylim(lim)
ax1.set_xlabel('Actual cluster ESCS slope')
ax1.set_ylabel('Predicted cluster ESCS slope')
ax1.set_title(f'v6 Predicted vs Actual\n(hold-out R2={r2_holdout:.3f}, n={len(y_test)})')
ax1.legend()

ax2 = axes[1]
ax2.hist(panel['N_STUDENTS'], bins=40, color='#4c8fbd',
         edgecolor='white', linewidth=0.3)
ax2.axvline(panel['N_STUDENTS'].median(), color='orange', linestyle='--',
            label=f'Median={panel["N_STUDENTS"].median():.0f} students')
ax2.axvline(MIN_STUDENTS_SLOPE, color='red', linestyle=':', alpha=0.7,
            label=f'Min threshold={MIN_STUDENTS_SLOPE}')
ax2.set_xlabel('Students pooled per cluster')
ax2.set_ylabel('Number of clusters')
ax2.set_title('v6: Pooled students per cluster\n(noise reduction vs v5 ~30/school)')
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig(FIGURES / 'pred_vs_actual_v6.png', dpi=130)
plt.close()
print('Saved: figures/pred_vs_actual_v6.png')

# cluster diagnostics
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax1 = axes[0]
ax1.hist(panel['ESCS_SLOPE'].dropna(), bins=40, color='#4c8fbd',
         edgecolor='white', linewidth=0.3, alpha=0.85)
ax1.axvline(panel['ESCS_SLOPE'].median(), color='orange', linestyle='--',
            label=f'Median={panel["ESCS_SLOPE"].median():.1f}')
ax1.axvline(0, color='red', linestyle=':', alpha=0.6, label='Zero slope')
ax1.set_xlabel('ESCS slope (PISA pts per unit ESCS)')
ax1.set_ylabel('Number of clusters')
ax1.set_title('v6: Distribution of cluster-level ESCS slope (2022)')
ax1.legend(fontsize=8)

ax2 = axes[1]
cnt_cluster_counts = panel.groupby('CNT').size().sort_values(ascending=False).head(20)
ax2.bar(range(len(cnt_cluster_counts)), cnt_cluster_counts.values, color='#4c8fbd')
ax2.set_xticks(range(len(cnt_cluster_counts)))
ax2.set_xticklabels(cnt_cluster_counts.index, rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('Number of clusters')
ax2.set_title('Clusters per country (top 20)')

plt.tight_layout()
plt.savefig(FIGURES / 'cluster_diagnostics_v6.png', dpi=130)
plt.close()
print('Saved: figures/cluster_diagnostics_v6.png')

# ── Phase 10: intervention ranking ───────────────────────────────────────────

print('\n' + '='*62)
print('PHASE 10 -- Intervention ranking')
print('='*62)

rank = shap_df[~shap_df['feature'].str.startswith('CNT_')].copy()
rank['cost_tier']   = rank['feature'].apply(
    lambda f: COST_TIERS[f][0] if f in COST_TIERS
              else ('POLICY' if f in POLICY_FEATURES else 'MED'))
rank['cost_weight'] = rank['feature'].apply(
    lambda f: COST_TIERS[f][1] if f in COST_TIERS
              else (0 if f in POLICY_FEATURES else 3))
rank['action'] = rank['feature'].apply(
    lambda f: COST_TIERS[f][2] if f in COST_TIERS else f)

school_rank = rank[rank['cost_tier'] != 'POLICY'].copy()
school_rank['efficiency'] = school_rank['mean_abs_shap'] / school_rank['cost_weight'].replace(0, np.nan)
policy_rank = rank[rank['cost_tier'] == 'POLICY'].sort_values('mean_abs_shap', ascending=False)

rank.to_csv(REPORTS / 'intervention_ranking_v6.csv', index=False)
print('Saved: reports/intervention_ranking_v6.csv')

print('\nTop 15 SCHOOL-LEVEL interventions:')
print(f'{"Feature":<22} {"Tier":<6} {"|SHAP|":>7} {"Direction":<28} {"Efficiency":>10}')
print('-' * 76)
for _, row in school_rank.head(15).iterrows():
    eff   = f'{row["efficiency"]:.3f}' if pd.notna(row['efficiency']) else '  n/a'
    arrow = ('increases slope(worse)'
             if row['direction'] == 'increases_slope'
             else 'reduces slope(better)')
    print(f'{row["feature"]:<22} {row["cost_tier"]:<6} {row["mean_abs_shap"]:>7.3f} '
          f'{arrow:<28} {eff:>10}')

print(f'\nTop POLICY findings ({len(policy_rank)} structural features):')
for _, row in policy_rank.head(8).iterrows():
    arrow = 'increases slope' if row['direction'] == 'increases_slope' else 'reduces slope'
    print(f'  {row["feature"]:<22} |SHAP|={row["mean_abs_shap"]:.3f}  {arrow}')

# ── Phase 11: version comparison ─────────────────────────────────────────────

print('\n' + '='*72)
print('VERSION COMPARISON')
print('='*72)
hdr = f'{"Ver":<5} {"Metric":<20} {"Obs":>8} {"Hold-out R2":>12} {"CV R2":>18} {"Pseudorep":>10}'
print(hdr)
print('-' * 73)
print(f'{"v3":<5} {"Quartile gap":<20} {"3,590":>8} {"0.250":>12} {"0.243+/-0.026":>18} {"YES":>10}')
print(f'{"v4":<5} {"ESCS slope":<20} {"53,249":>8} {"0.168":>12} {"0.132+/-0.024":>18} {"YES":>10}')
print(f'{"v5":<5} {"ESCS slope":<20} {"18,687":>8} {"0.211":>12} {"0.123+/-0.038":>18} {"NO":>10}')
print(f'{"v6":<5} {"ESCS slope (clust)":<20} {str(len(X)):>8} '
      f'{f"{r2_holdout:.3f}":>12} '
      f'{f"{loco_scores.mean():.3f}+/-{loco_scores.std():.3f}":>18} '
      f'{"NO":>10}')
print('='*72)
print('Note: v6 primary CV = LOCO (leave-one-country-out); v3-v5 used 5-fold.')
print(f'      5-fold for v6: {cv5_scores.mean():.3f} +/- {cv5_scores.std():.3f}')

print('\n-- FINAL SUMMARY -- EquiTrack v6 (clustered schools, 2022 only)')
print(f'Clusters       : {n_total} total | {n_valid} with valid slope ({pct:.1f}%)')
print(f'Avg students/cluster : {panel["N_STUDENTS"].mean():.0f}'
      f'  (vs ~30 in v5 -- {panel["N_STUDENTS"].mean()/30:.0f}x more students per obs)')
print(f'Features       : {X.shape[1]}')
print(f'Hold-out R2    : {r2_holdout:.3f}  (n_test={len(y_test)}, reference only)')
print(f'5-fold CV R2   : {cv5_scores.mean():.3f} +/- {cv5_scores.std():.3f}')
print(f'LOCO CV R2     : {loco_scores.mean():.3f} +/- {loco_scores.std():.3f}  [PRIMARY]')
print('\nOutputs:')
for f in [
    'cache/equitrack_panel_v6.parquet',
    'reports/xgb_equity_slope_v6.json',
    'reports/shap_importance_v6.csv',
    'reports/intervention_ranking_v6.csv',
    'figures/shap_bar_v6.png',
    'figures/pred_vs_actual_v6.png',
    'figures/cluster_diagnostics_v6.png',
]:
    print(f'  {f}')
