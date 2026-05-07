import pandas as pd
import numpy as np
import joblib
import os

SEGMENT_INFO = {
    'Resilient equitable performers': {
        'icon': chr(0x1F7E2), 'color': '#2C6E49',
        'desc': 'Higher maths performance with relatively lower within-school SES inequality.',
        'priorities': ['learning_support', 'family_engagement'],
    },
    'High-achievement unequal schools': {
        'icon': chr(0x1F7E0), 'color': '#D96C06',
        'desc': 'Higher maths performance but sizeable within-school SES inequality.',
        'priorities': ['learning_support', 'family_engagement', 'climate_support'],
    },
    'Low-achievement broad support need': {
        'icon': chr(0x1F535), 'color': '#2F6690',
        'desc': 'Lower maths performance with broad achievement challenges.',
        'priorities': ['learning_support', 'family_engagement', 'climate_support'],
    },
    'Strained high-inequality schools': {
        'icon': chr(0x1F534), 'color': '#B33F40',
        'desc': 'Lower maths performance combined with wider SES inequality and climate strain.',
        'priorities': ['climate_support', 'learning_support', 'resource_intensive'],
    },
    'Digitally constrained schools': {
        'icon': chr(0x1F7E3), 'color': '#6B5CA5',
        'desc': 'Schools where digital access constraints stand out.',
        'priorities': ['resource_intensive', 'learning_support', 'family_engagement'],
    },
}

INTERVENTION_CATEGORIES = {
    'Metacognition & self-regulation': 'learning_support',
    'Feedback': 'learning_support',
    'Mastery learning': 'learning_support',
    'Individualised instruction': 'learning_support',
    'Peer tutoring': 'learning_support',
    'Small group tuition': 'learning_support',
    'One to one tuition': 'learning_support',
    'Reading comprehension strategies': 'learning_support',
    'Homework': 'learning_support',
    'Phonics': 'learning_support',
    'Behaviour interventions': 'climate_support',
    'Social & emotional learning': 'climate_support',
    'Mentoring': 'climate_support',
    'Collaborative learning': 'climate_support',
    'Parental engagement': 'family_engagement',
    'Reducing class size': 'resource_intensive',
    'Teaching assistant interventions': 'resource_intensive',
    'Extending school time': 'resource_intensive',
}


_segment_bundle = None

def _load_segment_bundle():
    global _segment_bundle
    if _segment_bundle is None:
        model_path = os.path.join(os.path.dirname(__file__), 'data', 'segment_classifier.pkl')
        _segment_bundle = joblib.load(model_path)
    return _segment_bundle


def predict_segment(risk_score, school_mean_math, staffshort, edushort, negsclim, disadvantaged_pct):
    # Rule-based segmentation using only app-available inputs.
    # Thresholds: OECD 2022 average maths = 472; risk score median ≈ 44 (PISA 2022 schools).
    MATH_CUT  = 472   # OECD 2022 average maths score
    RISK_CUT  = 55    # elevated school-level risk (above global median + ~0.5 SD)
    STRAIN_CUT = 3    # slider value: "To some extent" or "A lot"

    high_math    = school_mean_math >= MATH_CUT
    high_risk    = risk_score >= RISK_CUT
    high_strain  = (int(staffshort) >= STRAIN_CUT) or (int(edushort) >= STRAIN_CUT)
    high_climate = int(negsclim) >= STRAIN_CUT

    if high_math and not high_risk:
        segment = 'Resilient equitable performers'
    elif high_math and high_risk:
        segment = 'High-achievement unequal schools'
    elif not high_math and int(edushort) == 4:
        segment = 'Digitally constrained schools'
    elif not high_math and (high_risk or high_strain or high_climate):
        segment = 'Strained high-inequality schools'
    else:
        segment = 'Low-achievement broad support need'

    return {
        'segment': segment,
        'confidence': None,
        'probabilities': {},
        'info': SEGMENT_INFO.get(segment, {}),
    }


def get_equity_risk_score(
    country_code,
    df_traj,
    df_gap,
    bullying_severity=2,
    ability_grouping='No',
    negsclim=2,
):
    """
    Three-component equity risk score (0–100):
      A. Country gap score   (0–50): normalized SES gap vs global mean
      B. Trajectory score    (0–20): Closing=0, Stable=10, Widening=20
      C. School profile score (0–30): OLS regression weights (PISA 2022, n=15238)
           ABGMATH    β=+1.225 (ability grouping widens gap)
           NEGSCLIM   β=+0.885 (negative climate widens gap)
           SC061Q05TA β=−1.468 (bullying compresses gap — protective effect)
         # TODO: add RATCMP1 (β=−0.574) once computers-per-student input is collected
    """
    # Get country gap from gap dataset (2022)
    gap_2022 = df_gap[df_gap['YEAR'].astype(str) == '2022']
    country_gap_row = gap_2022[gap_2022['CNT'] == country_code]

    if len(country_gap_row) == 0:
        country_gap = df_gap['GAP'].mean()
    else:
        country_gap = float(country_gap_row['GAP'].values[0])

    global_avg = float(gap_2022['GAP'].mean())
    global_std = float(gap_2022['GAP'].std())

    # Component A: country gap score (0–50)
    gap_score = float(np.clip(
        25 + (country_gap - global_avg) / global_std * 12.5,
        0, 50
    ))

    # Component B: trajectory score (0–20)
    traj_row = df_traj[df_traj['CNT'] == country_code]
    if len(traj_row) == 0:
        trajectory = 'Unknown'
        traj_score = 10
    else:
        trajectory = str(traj_row['TRAJECTORY'].values[0])
        traj_map = {'Closing': 0, 'Stable': 10, 'Widening': 20}
        traj_score = traj_map.get(trajectory, 10)

    # Component C: school profile score (0–30) via OLS |β| weights
    # All three inputs are treated as positive risk factors regardless of regression sign.
    # β magnitudes determine relative weight; direction is set by domain logic (more = more risk).
    # (SC061Q05TA β=−1.468 in the gap regression reflects bullying compressing the SES distribution,
    #  not that bullying is protective — high bullying is still a school risk factor.)
    ability_01   = 1.0 if str(ability_grouping) == 'Yes' else 0.0
    climate_01   = (float(negsclim) - 1) / 3.0
    bullying_01  = (float(bullying_severity) - 1) / 3.0

    # Weights = |β|; range: 0 → (1.225 + 0.885 + 1.468) = 3.578
    contribution = (1.225 * ability_01) + (0.885 * climate_01) + (1.468 * bullying_01)
    school_score = float(np.clip(contribution / 3.578 * 30, 0, 30))

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
    df_interventions,
    segment_priorities=None
):
    """
    Return ranked interventions filtered by budget and existing practices.

    Parameters:
    -----------
    country_code        : str   — ISO 3-letter country code
    annual_budget_gbp   : float — annual discretionary budget in GBP
    school_size         : int   — total number of students
    disadvantaged_pct   : float — proportion of disadvantaged students (0-1)
    existing_practices  : list  — interventions already in place
    df_interventions    : DataFrame — EEF intervention library
    segment_priorities  : list | None — ordered category priorities from predict_segment()

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

    # Map each intervention to its category
    inv_to_cat = {k: v for k, v in INTERVENTION_CATEGORIES.items()}
    df_int['category'] = df_int['intervention'].map(inv_to_cat).fillna('other')

    # Sort: by segment priority tier first, then cost-effectiveness within each tier
    if segment_priorities:
        priority_index = {cat: i for i, cat in enumerate(segment_priorities)}
        df_int['priority_tier'] = df_int['category'].map(
            lambda c: priority_index.get(c, len(segment_priorities))
        )
        df_int = df_int.sort_values(
            ['priority_tier', 'cost_effectiveness'],
            ascending=[True, False]
        )
    else:
        df_int['priority_tier'] = 0
        df_int = df_int.sort_values('cost_effectiveness', ascending=False)

    df_int['rank'] = range(1, len(df_int) + 1)

    return {
        'target_students': target_students,
        'interventions':   df_int[[
            'rank', 'intervention', 'gap_reduction_pts',
            'total_cost', 'cost_effectiveness',
            'evidence', 'impact_months', 'cost_rating',
            'category', 'priority_tier'
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
