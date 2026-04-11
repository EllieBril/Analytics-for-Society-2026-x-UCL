"""
Benchmark #2 - Random Forest vs LightGBM Robustness Comparison
===============================================================
EquiTrack - Analytics for Society 2026 x UCL

Purpose:  Second non-linear benchmark to check whether predictive signal
          is robust across model families, or model-specific.

Targets:
  1. Trajectory classification  (Closing / Stable / Widening)
  2. Gap-band classification    (Low / Medium / High)
  3. Risk-tier classification   (Low / Moderate / High)

Outputs -> results/  directory
"""

import os, sys, time, warnings, logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, RandomizedSearchCV
)
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, confusion_matrix
)
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from scipy.stats import randint, uniform

import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# -- PATHS --
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "school_risk_scores.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "benchmark_log.txt")

# -- LOGGING --
logger = logging.getLogger("benchmark")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
fh.setFormatter(fmt)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
logger.addHandler(fh)
logger.addHandler(sh)
log = logger.info

# -- CONSTANTS --
SEED = 42
N_FOLDS = 5
N_SEARCH = 50
HELD_OUT_K = 5
HELD_OUT_REPEATS = 5
np.random.seed(SEED)

# =============================================================================
# 1. LOAD & PREPARE DATA
# =============================================================================
log("=" * 70)
log("BENCHMARK #2 - Random Forest vs LightGBM")
log("=" * 70)
log(f"Start time: {datetime.now().isoformat()}")
log(f"Data path: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
log(f"Loaded school_risk_scores.csv - {df.shape[0]:,} rows, {df.shape[1]} cols")
log(f"Columns: {list(df.columns)}")
log(f"Countries: {df['CNT'].nunique()}")

# -- Derive targets --
q25 = df["COUNTRY_AVG_GAP"].quantile(0.25)
q75 = df["COUNTRY_AVG_GAP"].quantile(0.75)
log(f"COUNTRY_AVG_GAP Q25={q25:.2f}, Q75={q75:.2f}")

df["GAP_BAND"] = pd.cut(
    df["COUNTRY_AVG_GAP"],
    bins=[-np.inf, q25, q75, np.inf],
    labels=["Low", "Medium", "High"]
)

t33 = df["EQUITY_RISK_SCORE"].quantile(0.333)
t66 = df["EQUITY_RISK_SCORE"].quantile(0.666)
log(f"EQUITY_RISK_SCORE T33={t33:.2f}, T66={t66:.2f}")

df["RISK_TIER"] = pd.cut(
    df["EQUITY_RISK_SCORE"],
    bins=[-np.inf, t33, t66, np.inf],
    labels=["Low", "Moderate", "High"]
)

log(f"GAP_BAND distribution:\n{df['GAP_BAND'].value_counts().to_string()}")
log(f"RISK_TIER distribution:\n{df['RISK_TIER'].value_counts().to_string()}")
log(f"TRAJECTORY distribution:\n{df['TRAJECTORY'].value_counts(dropna=False).to_string()}")

# -- Encode categoricals --
le_traj = LabelEncoder()
df["TRAJECTORY_ENC"] = le_traj.fit_transform(df["TRAJECTORY"].fillna("Unknown"))

# -- Feature sets per target --
TARGET_CONFIG = {
    "trajectory": {
        "target": "TRAJECTORY",
        "features": ["SCHTYPE", "SCHSIZE", "STRATIO", "COUNTRY_AVG_GAP", "EQUITY_RISK_SCORE"],
        "drop_na_target": True,
    },
    "gap_band": {
        "target": "GAP_BAND",
        "features": ["SCHTYPE", "SCHSIZE", "STRATIO", "TRAJECTORY_ENC", "EQUITY_RISK_SCORE"],
        "drop_na_target": True,
    },
    "risk_tier": {
        "target": "RISK_TIER",
        "features": ["SCHTYPE", "SCHSIZE", "STRATIO", "COUNTRY_AVG_GAP", "TRAJECTORY_ENC"],
        "drop_na_target": True,
    },
}

# -- Hyperparameter search spaces --
RF_PARAMS = {
    "n_estimators": randint(100, 600),
    "max_depth": [None, 6, 10, 15, 20],
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
    "class_weight": ["balanced", "balanced_subsample", None],
}

LGBM_PARAMS = {
    "n_estimators": randint(100, 600),
    "max_depth": [-1, 6, 10, 15, 20],
    "learning_rate": uniform(0.01, 0.29),
    "num_leaves": randint(15, 127),
    "min_child_samples": randint(5, 50),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.5, 0.5),
    "reg_alpha": uniform(0, 1),
    "reg_lambda": uniform(0, 1),
    "class_weight": ["balanced", None],
}


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================
def prepare_data(target_name):
    cfg = TARGET_CONFIG[target_name]
    sub = df.copy()
    if cfg["drop_na_target"]:
        sub = sub.dropna(subset=[cfg["target"]])
    X = sub[cfg["features"]].copy()
    y = sub[cfg["target"]].values
    countries = sub["CNT"].values
    imp = SimpleImputer(strategy="median")
    X = pd.DataFrame(imp.fit_transform(X), columns=cfg["features"])
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X.values, y_enc, le, countries


def tune_and_cv(model_cls, param_dist, X, y, model_name, target_name):
    log(f"  [{model_name}] RandomizedSearchCV ({N_SEARCH} iters, {N_FOLDS}-fold)...")
    t0 = time.time()
    base = model_cls(random_state=SEED)
    if model_cls == lgb.LGBMClassifier:
        base.set_params(verbose=-1)
    search = RandomizedSearchCV(
        base, param_dist,
        n_iter=N_SEARCH,
        cv=StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED),
        scoring="f1_macro",
        n_jobs=-1,
        random_state=SEED,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y)
    elapsed = time.time() - t0
    log(f"  [{model_name}] Best CV macro-F1 = {search.best_score_:.4f}  ({elapsed:.1f}s)")
    log(f"  [{model_name}] Best params: {search.best_params_}")
    best = search.best_estimator_
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    oof_pred = cross_val_predict(best, X, y, cv=skf, method="predict")
    oof_proba = cross_val_predict(best, X, y, cv=skf, method="predict_proba")
    return best, oof_pred, oof_proba


def compute_metrics(y_true, y_pred, label_names):
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "per_class_f1": dict(zip(
            label_names,
            f1_score(y_true, y_pred, average=None).tolist()
        )),
    }


def plot_confusion(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"    Saved confusion matrix -> {os.path.basename(path)}")


def plot_calibration(y_true, y_proba, labels, model_name, target_name, path):
    n_classes = len(labels)
    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 3.5))
    if n_classes == 1:
        axes = [axes]
    for i, (ax, lbl) in enumerate(zip(axes, labels)):
        y_bin = (y_true == i).astype(int)
        if y_proba.shape[1] > i:
            prob_pos = y_proba[:, i]
        else:
            continue
        try:
            frac_pos, mean_pred = calibration_curve(y_bin, prob_pos, n_bins=10, strategy="uniform")
            ax.plot(mean_pred, frac_pos, "s-", color="#1D9E75", label="Model")
            ax.plot([0, 1], [0, 1], "--", color="#9CA3AF", linewidth=1)
        except ValueError:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        ax.set_title(f"{lbl}", fontsize=9)
        ax.set_xlabel("Mean predicted prob", fontsize=8)
        ax.set_ylabel("Fraction of positives", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.suptitle(f"Calibration - {model_name} / {target_name}", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"    Saved calibration plot -> {os.path.basename(path)}")


def cross_country_f1(y_true, y_pred, countries, labels):
    unique_c = np.unique(countries)
    results = {}
    for c in unique_c:
        mask = countries == c
        if mask.sum() < 5:
            continue
        n_classes_present = len(np.unique(y_true[mask]))
        if n_classes_present < 2:
            continue
        f1 = f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
        results[c] = f1
    return results


def held_out_country_eval(model, X, y, countries, target_name, model_name):
    log(f"  [{model_name}] Held-out-country eval ({HELD_OUT_K} countries x {HELD_OUT_REPEATS} repeats)...")
    unique_countries = np.unique(countries)
    results = []
    for rep in range(HELD_OUT_REPEATS):
        rng = np.random.RandomState(SEED + rep)
        held_out = rng.choice(unique_countries, size=min(HELD_OUT_K, len(unique_countries)), replace=False)
        mask_test = np.isin(countries, held_out)
        mask_train = ~mask_test
        if mask_test.sum() < 10 or mask_train.sum() < 10:
            continue
        if len(np.unique(y[mask_train])) < 2 or len(np.unique(y[mask_test])) < 2:
            continue
        model.fit(X[mask_train], y[mask_train])
        y_pred = model.predict(X[mask_test])
        macro_f1 = f1_score(y[mask_test], y_pred, average="macro", zero_division=0)
        bal_acc = balanced_accuracy_score(y[mask_test], y_pred)
        results.append({
            "repeat": rep + 1,
            "held_out_countries": ", ".join(held_out),
            "n_test": int(mask_test.sum()),
            "macro_f1": round(macro_f1, 4),
            "balanced_accuracy": round(bal_acc, 4),
        })
        log(f"    Rep {rep+1}: held-out={list(held_out)} -> F1={macro_f1:.4f}, BA={bal_acc:.4f}")
    return pd.DataFrame(results)


# =============================================================================
# 3. MAIN BENCHMARK LOOP
# =============================================================================
all_results = []

for target_name, cfg in TARGET_CONFIG.items():
    log("")
    log("=" * 70)
    log(f"TARGET: {target_name.upper()} ({cfg['target']})")
    log("=" * 70)

    X, y, le, countries = prepare_data(target_name)
    labels = list(le.classes_)
    log(f"  Data: X={X.shape}, classes={labels}, counts={np.bincount(y).tolist()}")

    # -- Random Forest --
    rf_model, rf_pred, rf_proba = tune_and_cv(
        RandomForestClassifier, RF_PARAMS, X, y, "RandomForest", target_name
    )
    rf_metrics = compute_metrics(y, rf_pred, labels)
    log(f"  [RF] OOF F1={rf_metrics['macro_f1']:.4f}, BA={rf_metrics['balanced_accuracy']:.4f}")
    log(f"  [RF] Per-class F1: {rf_metrics['per_class_f1']}")

    # -- LightGBM --
    lgbm_model, lgbm_pred, lgbm_proba = tune_and_cv(
        lgb.LGBMClassifier, LGBM_PARAMS, X, y, "LightGBM", target_name
    )
    lgbm_metrics = compute_metrics(y, lgbm_pred, labels)
    log(f"  [LGBM] OOF F1={lgbm_metrics['macro_f1']:.4f}, BA={lgbm_metrics['balanced_accuracy']:.4f}")
    log(f"  [LGBM] Per-class F1: {lgbm_metrics['per_class_f1']}")

    # -- Confusion matrices --
    plot_confusion(y, rf_pred, labels, f"Random Forest - {target_name}",
                   os.path.join(RESULTS_DIR, f"confusion_rf_{target_name}.png"))
    plot_confusion(y, lgbm_pred, labels, f"LightGBM - {target_name}",
                   os.path.join(RESULTS_DIR, f"confusion_lgbm_{target_name}.png"))

    # -- Calibration curves --
    plot_calibration(y, rf_proba, labels, "RandomForest", target_name,
                     os.path.join(RESULTS_DIR, f"calibration_rf_{target_name}.png"))
    plot_calibration(y, lgbm_proba, labels, "LightGBM", target_name,
                     os.path.join(RESULTS_DIR, f"calibration_lgbm_{target_name}.png"))

    # -- Cross-country stability --
    log(f"  Computing cross-country stability...")
    rf_country_f1 = cross_country_f1(y, rf_pred, countries, labels)
    lgbm_country_f1 = cross_country_f1(y, lgbm_pred, countries, labels)
    rf_f1_vals = list(rf_country_f1.values())
    lgbm_f1_vals = list(lgbm_country_f1.values())
    log(f"  [RF]   XCountry F1: mean={np.mean(rf_f1_vals):.4f} +/- {np.std(rf_f1_vals):.4f} ({len(rf_f1_vals)} countries)")
    log(f"  [LGBM] XCountry F1: mean={np.mean(lgbm_f1_vals):.4f} +/- {np.std(lgbm_f1_vals):.4f} ({len(lgbm_f1_vals)} countries)")

    # Plot cross-country stability
    fig, ax = plt.subplots(figsize=(10, 4))
    common_countries = sorted(set(rf_country_f1.keys()) & set(lgbm_country_f1.keys()))
    x_idx = np.arange(len(common_countries))
    width = 0.35
    ax.bar(x_idx - width/2, [rf_country_f1[c] for c in common_countries],
           width, label="Random Forest", color="#1D9E75", alpha=0.8)
    ax.bar(x_idx + width/2, [lgbm_country_f1[c] for c in common_countries],
           width, label="LightGBM", color="#6366F1", alpha=0.8)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(common_countries, rotation=90, fontsize=6)
    ax.set_ylabel("Macro F1")
    ax.set_title(f"Cross-Country Stability - {target_name}", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"cross_country_{target_name}.png"), dpi=150)
    plt.close(fig)
    log(f"    Saved cross-country plot -> cross_country_{target_name}.png")

    # -- Held-out country evaluation --
    rf_held = held_out_country_eval(clone(rf_model), X, y, countries, target_name, "RandomForest")
    lgbm_held = held_out_country_eval(clone(lgbm_model), X, y, countries, target_name, "LightGBM")

    if len(rf_held) > 0 and len(lgbm_held) > 0:
        rf_held["model"] = "RandomForest"
        lgbm_held["model"] = "LightGBM"
        held_all = pd.concat([rf_held, lgbm_held], ignore_index=True)
        held_all["target"] = target_name
        held_all.to_csv(os.path.join(RESULTS_DIR, f"held_out_countries_{target_name}.csv"), index=False)
        log(f"    Saved held-out results -> held_out_countries_{target_name}.csv")
        log(f"  [RF]   HeldOut F1: mean={rf_held['macro_f1'].mean():.4f} +/- {rf_held['macro_f1'].std():.4f}")
        log(f"  [LGBM] HeldOut F1: mean={lgbm_held['macro_f1'].mean():.4f} +/- {lgbm_held['macro_f1'].std():.4f}")

    # -- Collect summary --
    for model_name, metrics, country_f1_dict, held_df in [
        ("RandomForest", rf_metrics, rf_country_f1, rf_held),
        ("LightGBM", lgbm_metrics, lgbm_country_f1, lgbm_held),
    ]:
        row = {
            "target": target_name,
            "model": model_name,
            "macro_f1": round(metrics["macro_f1"], 4),
            "balanced_accuracy": round(metrics["balanced_accuracy"], 4),
            "cross_country_f1_mean": round(np.mean(list(country_f1_dict.values())), 4) if country_f1_dict else np.nan,
            "cross_country_f1_std": round(np.std(list(country_f1_dict.values())), 4) if country_f1_dict else np.nan,
            "held_out_f1_mean": round(held_df["macro_f1"].mean(), 4) if len(held_df) > 0 else np.nan,
            "held_out_f1_std": round(held_df["macro_f1"].std(), 4) if len(held_df) > 0 else np.nan,
        }
        for cls_name, cls_f1 in metrics["per_class_f1"].items():
            row[f"f1_{cls_name}"] = round(cls_f1, 4)
        all_results.append(row)


# =============================================================================
# 4. SUMMARY TABLE
# =============================================================================
log("")
log("=" * 70)
log("SUMMARY - HEAD-TO-HEAD COMPARISON")
log("=" * 70)

df_results = pd.DataFrame(all_results)
df_results.to_csv(os.path.join(RESULTS_DIR, "benchmark_comparison.csv"), index=False)
log(f"Saved benchmark_comparison.csv")

for target_name in TARGET_CONFIG:
    subset = df_results[df_results["target"] == target_name]
    log(f"\n  -- {target_name.upper()} --")
    for _, row in subset.iterrows():
        log(f"    {row['model']:15s}  F1={row['macro_f1']:.4f}  BA={row['balanced_accuracy']:.4f}  "
            f"XCountry={row['cross_country_f1_mean']:.4f}+/-{row['cross_country_f1_std']:.4f}  "
            f"HeldOut={row['held_out_f1_mean']:.4f}+/-{row['held_out_f1_std']:.4f}")

# -- Robustness verdict --
log("")
log("=" * 70)
log("ROBUSTNESS VERDICT")
log("=" * 70)

for target_name in TARGET_CONFIG:
    subset = df_results[df_results["target"] == target_name]
    rf_row = subset[subset["model"] == "RandomForest"].iloc[0]
    lgbm_row = subset[subset["model"] == "LightGBM"].iloc[0]
    delta_f1 = abs(rf_row["macro_f1"] - lgbm_row["macro_f1"])
    delta_ba = abs(rf_row["balanced_accuracy"] - lgbm_row["balanced_accuracy"])
    if delta_f1 < 0.03 and delta_ba < 0.03:
        verdict = "ROBUST - Signal is stable across model families"
    elif delta_f1 < 0.06:
        verdict = "MIXED - Moderate difference; signal partly model-dependent"
    else:
        verdict = "FRAGILE - Large gap between models; signal may be model-specific"
    better = "LightGBM" if lgbm_row["macro_f1"] > rf_row["macro_f1"] else "Random Forest"
    log(f"  {target_name}: {verdict}")
    log(f"    Delta_F1 = {delta_f1:.4f}, Delta_BA = {delta_ba:.4f}  (better: {better})")

log("")
log(f"Completed at {datetime.now().isoformat()}")
log(f"All results saved to {RESULTS_DIR}")
log("=" * 70)
