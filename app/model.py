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
    bundle = _load_segment_bundle()
    mdl = bundle['model']
    le = bundle['label_encoder']
    school_mean_escs = 0.5 - (disadvantaged_pct * 2.0)
    features = pd.DataFrame([{
        'EQUITY_RISK_SCORE': risk_score,
        'school_mean_math': school_mean_math,
        'STAFFSHORT_b': float(staffshort),
        'EDUSHORT_b': float(edushort),
        'NEGSCLIM_b': float(negsclim),
        'school_mean_escs': school_mean_escs,
    }])
    segment = le.inverse_transform(mdl.predict(features))[0]
    probas = dict(zip(le.classes_, mdl.predict_proba(features)[0]))
    return {
        'segment': segment,
        'confidence': round(max(probas.values()) * 100, 1),
        'probabilities': {k: round(v * 100, 1) for k, v in probas.items()},
        'info': SEGMENT_INFO.get(segment, {}),
    }


def get_equity_risk_score(
    country_code,
    school_type,
    stratio,
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
    global_std = float(gap_2022['GAP'].std())  # ddof=1 (sample std, pandas default)

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
