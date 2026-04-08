from __future__ import annotations

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
