from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import normalize_id_series, write_text


PROTOTYPE_SCRIPT = """from __future__ import annotations

import json


def classify_risk(score: float, thresholds: tuple[float, float, float] | None = None) -> str:
    moderate_cut, high_cut, very_high_cut = thresholds or (40, 55, 70)
    if score >= very_high_cut:
        return "Very high"
    if score >= high_cut:
        return "High"
    if score >= moderate_cut:
        return "Moderate"
    return "Lower"


def infer_segment(profile: dict) -> str:
    mean_math = profile.get("school_mean_math")
    gap = profile.get("within_school_gap")
    climate = profile.get("NEGSCLIM")
    if mean_math is None or gap is None:
        return "Mixed profile"
    if mean_math >= profile.get("performance_cut", 0) and gap < profile.get("gap_cut", 0):
        return "Resilient equitable performers"
    if mean_math >= profile.get("performance_cut", 0) and gap >= profile.get("gap_cut", 0):
        return "High-achievement unequal schools"
    if mean_math < profile.get("performance_cut", 0) and gap >= profile.get("gap_cut", 0) and climate is not None:
        return "Strained high-inequality schools"
    return "Low-achievement broad support need"


def recommend(profile: dict) -> dict:
    segment = infer_segment(profile)
    thresholds = (
        profile.get("risk_cut_moderate", 40),
        profile.get("risk_cut_high", 55),
        profile.get("risk_cut_very_high", 70),
    )
    risk_class = classify_risk(profile.get("EQUITY_RISK_SCORE", 0), thresholds=thresholds)
    if segment == "Strained high-inequality schools":
        interventions = ["Behaviour interventions", "Social & emotional learning", "Small group tuition"]
    elif segment == "High-achievement unequal schools":
        interventions = ["Metacognition & self-regulation", "Parental engagement", "Feedback"]
    elif segment == "Resilient equitable performers":
        interventions = ["Metacognition & self-regulation", "Feedback", "Parental engagement"]
    else:
        interventions = ["Peer tutoring", "Small group tuition", "Parental engagement"]
    return {
        "risk_classification": risk_class,
        "segment": segment,
        "top_interventions": interventions,
        "rationale": f"Segment={segment}; risk={risk_class}; recommendation is evidence-informed, not causal."
    }


if __name__ == "__main__":
    example = {
        "EQUITY_RISK_SCORE": 72,
        "school_mean_math": 430,
        "within_school_gap": 88,
        "NEGSCLIM": 0.9,
        "performance_cut": 470,
        "gap_cut": 62,
        "risk_cut_moderate": 45,
        "risk_cut_high": 56,
        "risk_cut_very_high": 68,
    }
    print(json.dumps(recommend(example), indent=2))
"""


def build_prototype(
    segment_assignments: pd.DataFrame,
    intervention_map: pd.DataFrame,
    school_profiles: pd.DataFrame,
    output_dir: Path,
    logger,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text(output_dir / "prototype_recommender.py", PROTOTYPE_SCRIPT)

    cut_perf = segment_assignments["school_mean_math"].median()
    cut_gap = segment_assignments["within_school_gap"].median()
    risk_cut_moderate = segment_assignments["EQUITY_RISK_SCORE"].quantile(0.4)
    risk_cut_high = segment_assignments["EQUITY_RISK_SCORE"].quantile(0.6)
    risk_cut_very_high = segment_assignments["EQUITY_RISK_SCORE"].quantile(0.8)
    examples = segment_assignments.sort_values("EQUITY_RISK_SCORE", ascending=False).groupby("official_segment").head(2).copy()
    bins = [-1e9, risk_cut_moderate, risk_cut_high, risk_cut_very_high, 1e9]
    examples["risk_classification"] = pd.cut(
        examples["EQUITY_RISK_SCORE"], bins=bins, labels=["Lower", "Moderate", "High", "Very high"], include_lowest=True
    ).astype("string")
    top_map = (
        intervention_map.groupby("segment")["intervention"]
        .apply(lambda s: " | ".join(s.head(3)))
        .rename("recommended_interventions")
        .reset_index()
    )
    examples = examples.merge(top_map, left_on="official_segment", right_on="segment", how="left")
    examples["rationale"] = (
        "Evidence-informed prioritisation based on observed school profile; not causal proof."
    )
    examples["performance_cut"] = cut_perf
    examples["gap_cut"] = cut_gap
    examples["risk_cut_moderate"] = risk_cut_moderate
    examples["risk_cut_high"] = risk_cut_high
    examples["risk_cut_very_high"] = risk_cut_very_high
    keep_cols = [
        "CNT",
        "school_key",
        "official_segment",
        "risk_classification",
        "EQUITY_RISK_SCORE",
        "school_mean_math",
        "within_school_gap",
        "recommended_interventions",
        "rationale",
        "performance_cut",
        "gap_cut",
        "risk_cut_moderate",
        "risk_cut_high",
        "risk_cut_very_high",
    ]
    examples[keep_cols].to_csv(output_dir / "prototype_examples.csv", index=False)
    school_view_cols = [
        "CNT",
        "school_key",
        "student_count",
        "low_perf_share_400",
        "STAFFSHORT",
        "EDUSHORT",
        "NEGSCLIM",
        "student_belong_mean",
        "student_math_anxiety_mean",
        "student_device_problem_mean",
        "student_internet_problem_mean",
        "TRAJECTORY",
        "teacher_response_count",
        "risk_quintile",
        "country_school_count",
    ]
    school_view = segment_assignments.copy()
    school_view["CNT"] = school_view["CNT"].astype("string")
    school_view["school_key"] = normalize_id_series(school_view["school_key"])
    profiles_for_merge = school_profiles.copy()
    profiles_for_merge["CNT"] = profiles_for_merge["CNT"].astype("string")
    profiles_for_merge["school_key"] = normalize_id_series(profiles_for_merge["school_key"])
    merge_cols = ["CNT", "school_key"] + [col for col in school_view_cols if col in profiles_for_merge.columns and col not in {"CNT", "school_key"}]
    school_view = school_view.merge(
        profiles_for_merge[merge_cols],
        on=["CNT", "school_key"],
        how="left",
    )
    school_view.to_csv(output_dir / "dashboard_school_view.csv", index=False)
    write_text(
        output_dir / "prototype_logic_note.md",
        "\n".join(
            [
                "# Prototype Logic Note",
                "",
                "- Input: school profile metrics already produced by the pipeline.",
                "- Output: risk classification, segment, recommended intervention categories, and a short rationale.",
                "- Design choice: simple inspectable rules, no opaque black-box recommender.",
                "- Risk tiers use percentile-based score cut points from the current school-profile distribution.",
                "- The dashboard uses `dashboard_school_view.csv` as a lightweight public-facing school-profile artifact.",
                "- Interpretation: recommendations are evidence-informed prioritisation suggestions only.",
            ]
        ),
    )
    logger.info("Step 10 complete: prototype examples=%s", len(examples))
    return {"prototype_examples": examples[keep_cols], "dashboard_school_view": school_view}
