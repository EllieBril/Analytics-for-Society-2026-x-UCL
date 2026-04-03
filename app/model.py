import pandas as pd
import numpy as np


def get_equity_risk_score(
    country_code,
    school_type,
    stratio,
    df_scores,
    df_traj,
    df_gap
):
    # Get country gap from gap dataset (2022)
    gap_2022 = df_gap[df_gap['YEAR'].astype(str) == '2022']
    country_gap_row = gap_2022[gap_2022['CNT'] == country_code]

    if len(country_gap_row) == 0:
        country_gap = df_gap['GAP'].mean()
    else:
        country_gap = float(country_gap_row['GAP'].values[0])

    global_avg = float(gap_2022['GAP'].mean())
    global_std = float(gap_2022['GAP'].std())

    # Component A: gap size score (0-60)
    gap_score = float(np.clip(
        30 + (country_gap - global_avg) / global_std * 15,
        0, 60
    ))

    # Component B: trajectory score (0-20)
    traj_row = df_traj[df_traj['CNT'] == country_code]
    if len(traj_row) == 0:
        trajectory = 'Unknown'
        traj_score = 10
    else:
        trajectory = str(traj_row['TRAJECTORY'].values[0])
        traj_map = {'Closing': 0, 'Stable': 10, 'Widening': 20}
        traj_score = traj_map.get(trajectory, 10)

    # Component C: school profile score (0-20)
    school_score = 10.0
    schtype_adj = {
        'Public': 2,
        'Government-dependent private': 0,
        'Independent private': -2
    }
    school_score += schtype_adj.get(school_type, 0)

    global_median_stratio = 14.4
    if stratio < global_median_stratio:
        school_score -= 1
    elif stratio > global_median_stratio:
        school_score += 1

    school_score = float(np.clip(school_score, 0, 20))

    equity_risk = float(np.clip(
        gap_score + traj_score + school_score,
        0, 100
    ))

    return {
        'equity_risk':  round(equity_risk, 1),
        'gap_score':    round(gap_score, 1),
        'traj_score':   traj_score,
        'school_score': round(school_score, 1),
        'country_gap':  round(country_gap, 1),
        'trajectory':   trajectory
    }


def get_interventions(
    country_code,
    annual_budget_gbp,
    school_size,
    disadvantaged_pct,
    existing_practices,
    df_interventions
):
    """
    Return ranked interventions filtered by budget and existing practices.

    Parameters:
    -----------
    country_code      : str   — ISO 3-letter country code
    annual_budget_gbp : float — annual discretionary budget in GBP
    school_size       : int   — total number of students
    disadvantaged_pct : float — proportion of disadvantaged students (0-1)
    existing_practices: list  — interventions already in place
    df_interventions  : DataFrame — EEF intervention library

    Returns:
    --------
    dict with target_students count and ranked interventions DataFrame
    """

    target_students = max(1, round(school_size * disadvantaged_pct))

    df_int = df_interventions.copy()

    # Calculate total cost for target group
    df_int['total_cost'] = df_int['cost_per_pupil'] * target_students

    # Filter: exclude existing practices
    df_int = df_int[~df_int['intervention'].isin(existing_practices)]

    # Filter: within budget
    df_int = df_int[df_int['total_cost'] <= annual_budget_gbp]

    # Filter: positive impact only
    df_int = df_int[df_int['impact_months'] > 0]

    # Filter: minimum evidence strength
    df_int = df_int[df_int['evidence'] >= 2]

    # Sort by cost-effectiveness descending
    df_int = df_int.sort_values('cost_effectiveness', ascending=False)
    df_int['rank'] = range(1, len(df_int) + 1)

    return {
        'target_students': target_students,
        'interventions':   df_int[[
            'rank', 'intervention', 'gap_reduction_pts',
            'total_cost', 'cost_effectiveness',
            'evidence', 'impact_months', 'cost_rating'
        ]]
    }


def calculate_realistic_reduction(
    interventions_df,
    current_gap,
    target_pct=0.25
):
    """
    Calculate realistic gap reduction applying:
    1. Target group scaling (disadvantaged students only)
    2. Diminishing returns when stacking multiple interventions
    3. Cap at 60% of current gap

    Source: EEF impact estimates converted to PISA points
    (1 month EEF impact ≈ 3.5 PISA points, OECD benchmark)
    """

    sorted_ints = interventions_df.sort_values(
        'gap_reduction_pts', ascending=False
    )

    total = 0.0
    for i, (_, row) in enumerate(sorted_ints.iterrows()):
        # Scale to target group
        scaled = row['gap_reduction_pts'] * target_pct
        # Diminishing returns — each additional intervention adds less
        decay = 0.7 ** i
        total += scaled * decay

    # Cap at 60% of current gap
    max_reduction = current_gap * 0.60
    realistic = min(total, max_reduction)

    return round(realistic, 1)
