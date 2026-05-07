"""
Regenerate app/data/school_risk_scores.csv using the new OLS-evidence-based
equity risk score formula.

Run from the repo root:
    python scripts/regenerate_risk_scores.py

Sources:
  - d:/analytics_for_society/cache/school_2022.parquet  (21,629 PISA 2022 schools)
  - d:/afs_main/app/data/equity_gap_by_country_year.csv (country gap data)
  - d:/afs_main/app/data/country_trajectories.csv       (gap trajectories)

Formula (same as app/model.py get_equity_risk_score):
  gap_score    (0-50) : country gap normalised vs global avg
  traj_score   (0-20) : Closing=0, Stable=10, Widening=20
  school_score (0-30) : OLS regression weights
      ABGMATH    β=+1.225  derived from SC042Q01TA  (1/2 = grouped, 3 = not)
      NEGSCLIM   β=+0.885  mean of SC061Q01-Q04TA
      SC061Q05TA β=−1.468  student bullying/intimidation item
"""

import os
import sys
import numpy as np
import pandas as pd

APP_DATA = os.path.join(os.path.dirname(__file__), '..', 'app', 'data')
SCHOOL_PARQUET = 'd:/analytics_for_society/cache/school_2022.parquet'
OUT_CSV = os.path.join(APP_DATA, 'school_risk_scores.csv')

BETAS = dict(abgmath=1.225, negsclim=0.885, bullying=1.468)  # all |β|, all positive risk
CONTRIB_MIN = 0.0
CONTRIB_MAX = 1.225 + 0.885 + 1.468  # 3.578
CONTRIB_RANGE = CONTRIB_MAX


def load_country_data():
    gap_df = pd.read_csv(os.path.join(APP_DATA, 'equity_gap_by_country_year.csv'))
    gap_2022 = gap_df[gap_df['YEAR'].astype(str) == '2022'][['CNT', 'GAP']].copy()

    traj_df = pd.read_csv(os.path.join(APP_DATA, 'country_trajectories.csv'))
    traj_df = traj_df[['CNT', 'TRAJECTORY']].copy()

    global_avg = float(gap_2022['GAP'].mean())
    global_std = float(gap_2022['GAP'].std())

    return gap_2022, traj_df, global_avg, global_std


def gap_score(country_gap, global_avg, global_std):
    return float(np.clip(25 + (country_gap - global_avg) / global_std * 12.5, 0, 50))


def traj_score(trajectory):
    return {'Closing': 0, 'Stable': 10, 'Widening': 20}.get(str(trajectory), 10)


def school_profile_score(ability_01, climate_01, bullying_01):
    contribution = (
        BETAS['abgmath'] * ability_01
        + BETAS['negsclim'] * climate_01
        + BETAS['bullying'] * bullying_01
    )
    return float(np.clip(contribution / CONTRIB_RANGE * 30, 0, 30))


def main():
    print('Loading school data...')
    df = pd.read_parquet(SCHOOL_PARQUET, columns=[
        'CNT', 'CNTSCHID', 'SCHLTYPE', 'SCHSIZE',
        'SC061Q05TA',
        'SC042Q01TA',
        'SC061Q01TA', 'SC061Q02TA', 'SC061Q03TA', 'SC061Q04TA',
    ])
    print(f'  {len(df):,} schools loaded')

    # --- Derive ABGMATH proxy ---
    # SC042Q01TA: 1=grouped all classes, 2=grouped some, 3=not grouped, 95=invalid
    df['SC042Q01TA'] = df['SC042Q01TA'].where(df['SC042Q01TA'] <= 4)
    df['ABGMATH_proxy'] = (df['SC042Q01TA'].fillna(3) < 3).astype(float)

    # --- Derive NEGSCLIM proxy ---
    # Mean of SC061Q01-Q04 on 1-4 scale; clip 95s (invalid codes)
    negsclim_items = ['SC061Q01TA', 'SC061Q02TA', 'SC061Q03TA', 'SC061Q04TA']
    for col in negsclim_items:
        df[col] = df[col].where(df[col] <= 4)
    df['NEGSCLIM_proxy'] = df[negsclim_items].mean(axis=1)
    # Clip SC061Q05TA similarly
    df['SC061Q05TA'] = df['SC061Q05TA'].where(df['SC061Q05TA'] <= 4)

    # --- Load country gap / trajectory ---
    gap_2022, traj_df, global_avg, global_std = load_country_data()
    df = df.merge(gap_2022, on='CNT', how='left')
    df = df.merge(traj_df, on='CNT', how='left')
    df['GAP'] = df['GAP'].fillna(gap_2022['GAP'].mean())
    df['TRAJECTORY'] = df['TRAJECTORY'].fillna('Stable')

    # --- Country-level imputation for missing school inputs ---
    for col in ['ABGMATH_proxy', 'NEGSCLIM_proxy', 'SC061Q05TA']:
        country_means = df.groupby('CNT')[col].transform('mean')
        df[col] = df[col].fillna(country_means)
        # Fallback: global mean
        df[col] = df[col].fillna(df[col].mean())

    # --- Normalise to 0-1 ---
    df['ability_01']  = df['ABGMATH_proxy']                          # already 0/1
    df['climate_01']  = (df['NEGSCLIM_proxy'].clip(1, 4) - 1) / 3
    df['bullying_01'] = (df['SC061Q05TA'].clip(1, 4) - 1) / 3

    # --- Compute components ---
    df['gap_score']    = df['GAP'].apply(lambda g: gap_score(g, global_avg, global_std))
    df['traj_score']   = df['TRAJECTORY'].apply(traj_score)
    df['school_score'] = df.apply(
        lambda r: school_profile_score(r['ability_01'], r['climate_01'], r['bullying_01']),
        axis=1
    )
    df['EQUITY_RISK_SCORE'] = (
        df['gap_score'] + df['traj_score'] + df['school_score']
    ).clip(0, 100).round(1)

    # --- Build output frame ---
    schtype_map = {1.0: 1.0, 2.0: 2.0, 3.0: 3.0}
    out = pd.DataFrame({
        'CNT':              df['CNT'],
        'SCHOOLID':         df['CNTSCHID'],
        'SCHTYPE':          df['SCHLTYPE'].map(schtype_map),
        'SCHSIZE':          df['SCHSIZE'],
        'STRATIO':          np.nan,          # not in source; kept for schema compat
        'COUNTRY_AVG_GAP':  df['GAP'].round(1),
        'TRAJECTORY':       df['TRAJECTORY'],
        'EQUITY_RISK_SCORE': df['EQUITY_RISK_SCORE'],
    })

    out.to_csv(OUT_CSV, index=False)
    print(f'Saved {len(out):,} rows to {OUT_CSV}')
    print(f"Score distribution:\n{out['EQUITY_RISK_SCORE'].describe().round(1).to_string()}")


if __name__ == '__main__':
    main()
