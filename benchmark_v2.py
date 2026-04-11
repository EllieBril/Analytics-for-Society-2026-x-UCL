import os, sys, time, logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict, RandomizedSearchCV
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from scipy.stats import randint, uniform
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")
RESULTS_DIR = "results_v2"
os.makedirs(RESULTS_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", 
                    handlers=[logging.FileHandler(f"{RESULTS_DIR}/bench_log.txt", "w"), logging.StreamHandler(sys.stdout)])
log = logging.getLogger().info

log("STARTING PIPELINE...")
t0 = time.time()
df_pisa = pd.read_csv("pisa_2022.csv", low_memory=False)
df_gap = pd.read_csv("equity_gap_by_country_year.csv").rename(columns={"CNT":"country", "YEAR":"year", "GAP":"country_gap"})
df_traj = pd.read_csv("country_trajectories.csv").rename(columns={"CNT":"country", "TRAJECTORY":"target_traj"})

df_pisa = df_pisa[df_pisa["year"].isin([2009, 2012, 2015, 2018, 2022])].copy()
df_merged = df_pisa.merge(df_gap[["country", "year", "country_gap"]], on=["country", "year"], how="left") \
                   .merge(df_traj[["country", "target_traj"]], on="country", how="left")
df_merged = df_merged.dropna(subset=["country_gap", "target_traj"]).copy()
log(f"[1/4] Data merged: {df_merged.shape[0]} rows")

def assign_gap(g):
    return pd.cut(g["country_gap"], bins=[-np.inf, g["country_gap"].quantile(0.25), g["country_gap"].quantile(0.75), np.inf], labels=["Low", "Medium", "High"])
df_merged["target_gap"] = df_merged.groupby("year", group_keys=False).apply(assign_gap)

y_stat = df_merged.groupby("year")["country_gap"].agg(["mean", "std"])
def calc_risk(r):
    s = y_stat.loc[r["year"]]
    g_score = np.clip(30 + (r["country_gap"] - s["mean"]) / s["std"] * 15, 0, 60)
    t_score = {"Closing": 0, "Stable": 10, "Widening": 20}.get(r["target_traj"], 10)
    sch_score = 10.0 + (2 if "Public" in str(r["public_private"]) else -2 if "Independent" in str(r["public_private"]) else 0)
    if pd.notnull(r["stratio"]): sch_score += -1 if r["stratio"] < 14.4 else 1
    return np.clip(g_score + t_score + np.clip(sch_score, 0, 20), 0, 100)

df_merged["DYN_RISK"] = df_merged.apply(calc_risk, axis=1)
df_merged["target_risk"] = pd.cut(df_merged["DYN_RISK"], bins=[-np.inf, df_merged["DYN_RISK"].quantile(0.33), df_merged["DYN_RISK"].quantile(0.66), np.inf], labels=["Low", "Mod", "High"])
df_merged["pp_enc"] = LabelEncoder().fit_transform(df_merged["public_private"].astype(str))
log("[2/4] Engineering done. Starting models...")

sf = ["fund_gov", "fund_fees", "fund_donation", "enrol_boys", "enrol_girls", "stratio", "pp_enc", "staff_shortage", "school_size", "year"]

T = {
    "trajectory": ("target_traj", GroupKFold(5)),
    "gap_band": ("target_gap", StratifiedKFold(5, shuffle=True, random_state=42)),
    "risk_tier": ("target_risk", StratifiedKFold(5, shuffle=True, random_state=42))
}

res = []
for i, tgt in enumerate(T.keys()):
    log(f"--- Training target {i+1}/3 ({tgt}) ---")
    sub = df_merged.dropna(subset=[T[tgt][0]]).copy()
    X = SimpleImputer(strategy="median").fit_transform(sub[sf])
    y = LabelEncoder().fit_transform(sub[T[tgt][0]])
    grps = sub["country"].values
    
    cv_obj = T[tgt][1]
    cv_g = grps if isinstance(cv_obj, GroupKFold) else None
    
    for mod_name, mod, params in [
        ("RF", RandomForestClassifier(random_state=42, n_jobs=-1), {"max_depth": [None, 10, 20], "n_estimators": [100]}),
        ("LGBM", lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1), {"max_depth": [-1, 10, 20], "n_estimators": [100]})
    ]:
        s = RandomizedSearchCV(mod, params, n_iter=5, cv=cv_obj, scoring="f1_macro", n_jobs=-1, random_state=42)
        s.fit(X, y, groups=cv_g) if cv_g is not None else s.fit(X, y)
        
        oof = np.zeros_like(y)
        for tr, te in cv_obj.split(X, y, grps):
            c = clone(s.best_estimator_).fit(X[tr], y[tr])
            oof[te] = c.predict(X[te])
            
        f1 = f1_score(y, oof, average="macro")
        log(f"      {mod_name} OOF F1: {f1:.4f}")
        res.append({"target": tgt, "model": mod_name, "f1": round(f1, 4)})

pd.DataFrame(res).to_csv(f"{RESULTS_DIR}/v2_comparison.csv", index=False)
log(f"[4/4] Complete in {time.time()-t0:.1f}s!")
