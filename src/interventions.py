from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import write_text


INTERVENTION_CATEGORY_RULES = {
    "learning_support": ["Metacognition", "Feedback", "Mastery learning", "Individualised instruction", "Peer tutoring", "Small group tuition", "One to one tuition"],
    "climate_support": ["Behaviour interventions", "Social & emotional learning", "Mentoring", "Collaborative learning"],
    "family_engagement": ["Parental engagement"],
    "resource_intensive": ["Reducing class size", "Teaching assistant interventions", "Extending school time"],
    "reading_and_study": ["Reading comprehension strategies", "Homework", "Phonics"],
}


def categorize_intervention(name: str) -> str:
    for category, patterns in INTERVENTION_CATEGORY_RULES.items():
        if any(pattern.lower() in name.lower() for pattern in patterns):
            return category
    return "other"


def select_recommendation_categories(segment_name: str) -> list[str]:
    if segment_name == "Strained high-inequality schools":
        return ["climate_support", "learning_support", "resource_intensive"]
    if segment_name == "High-achievement unequal schools":
        return ["learning_support", "family_engagement", "climate_support"]
    if segment_name == "Low-achievement broad support need":
        return ["learning_support", "family_engagement", "climate_support"]
    if segment_name == "Resilient equitable performers":
        return ["learning_support", "family_engagement"]
    if segment_name == "Digitally constrained schools":
        return ["resource_intensive", "learning_support", "family_engagement"]
    return ["learning_support", "climate_support"]


def map_interventions(segment_assignments: pd.DataFrame, intervention_library: pd.DataFrame, output_dir: Path, logger) -> dict[str, pd.DataFrame]:
    interventions = intervention_library.copy()
    interventions["category"] = interventions["intervention"].map(categorize_intervention)

    segment_names = sorted(segment_assignments["official_segment"].dropna().astype(str).unique().tolist())
    rows = []
    for segment_name in segment_names:
        categories = select_recommendation_categories(segment_name)
        for category_rank, category in enumerate(categories, start=1):
            selected = interventions.loc[interventions["category"] == category].sort_values(
                ["evidence", "cost_effectiveness"], ascending=[False, False]
            )
            if selected.empty:
                continue
            row = selected.iloc[0]
            rows.append(
                {
                    "segment": segment_name,
                    "category_rank": category_rank,
                    "recommended_category": row["category"],
                    "intervention": row["intervention"],
                    "evidence": row["evidence"],
                    "cost_rating": row["cost_rating"],
                    "gap_reduction_pts": row["gap_reduction_pts"],
                    "cost_effectiveness": row["cost_effectiveness"],
                    "rule_rationale": f"Segment `{segment_name}` maps to `{row['category']}` under transparent profile rules.",
                }
            )
    mapping = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(output_dir / "segment_to_intervention_map.csv", index=False)

    lines = [
        "# Recommendation Rules",
        "",
        "## Core logic",
        "- High inequality plus climate or behaviour strain maps toward climate/support interventions.",
        "- Low achievement maps toward structured learning-support interventions.",
        "- Parent expectation or support deficits map toward family-engagement interventions.",
        "- Severe resource strain maps to resource-intensive interventions, but these are flagged as higher-burden and not necessarily cost-effective.",
        "",
        "## Caution",
        "- The intervention library is generic and evidence-informed; it does not prove PISA-causal effectiveness for a given school.",
        "- Recommendations are prioritisation suggestions, not deterministic prescriptions.",
    ]
    write_text(output_dir / "recommendation_rules.md", "\n".join(lines))
    logger.info("Step 9 complete: intervention mapping rows=%s", len(mapping))
    return {"mapping": mapping}
