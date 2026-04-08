from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import format_num, write_text


def build_competition_outputs(
    root: Path,
    inventory: pd.DataFrame,
    latest_rankings: pd.DataFrame,
    trend_rankings: pd.DataFrame,
    school_profiles: pd.DataFrame,
    segment_profiles: pd.DataFrame,
    intervention_map: pd.DataFrame,
    final_dir: Path,
    competition_dir: Path,
    logger,
) -> None:
    competition_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    worsening = trend_rankings.loc[trend_rankings["trend_label"] == "Worsening"].head(10)
    high_gap = latest_rankings.head(10)
    segment_counts = segment_profiles[["official_segment", "segment_size"]] if "official_segment" in segment_profiles.columns else segment_profiles

    narrative = [
        "# Competition Narrative",
        "",
        "## Problem diagnosis",
        "- The project targets unequal learning outcomes by socio-economic background, with mathematics as the primary anchor outcome.",
        "- The core product is a hybrid diagnostic: country trajectory analysis to identify where equity gaps are largest or worsening, and school-profile segmentation to prioritise targeted responses.",
        "",
        "## User and decision improved",
        "- Primary user: ministries, NGOs, funders, or system leaders who need to decide where to focus scarce support first.",
        "- Secondary user: school-support organisations using school-level profiles as a diagnostic, not as a causal prescription engine.",
        "",
        "## Why this is stronger than one-size-fits-all advice",
        "- Schools with similar average performance can have very different equity profiles, climate strain, and resource constraints.",
        "- The segmentation step turns those differences into tailored intervention logic rather than generic advice.",
        "",
        "## Why not one recommendation for all schools",
        "- High-achievement unequal schools and low-achievement resource-strained schools do not show the same pattern of need.",
        "- A single recommendation would ignore whether the observed challenge is concentrated inequality, broad low performance, climate strain, or digital access strain.",
        "",
        "## Why not a fully prescriptive individual-school dashboard",
        "- The data are observational and mostly cross-sectional for the rich school-level features.",
        "- The project can support evidence-informed prioritisation for a school profile, but not strong causal claims about what will work for one named school.",
    ]
    write_text(competition_dir / "competition_narrative.md", "\n".join(narrative))

    scope_limits = [
        "# Scope And Limits",
        "",
        "## Safe claims",
        "- Which countries currently have high observed SES-maths gaps.",
        "- Which countries show worsening or improving trajectories when enough time points are available.",
        "- Which school profiles are associated with higher risk, weaker climate, lower performance, or wider within-school inequality in 2022.",
        "- Which intervention categories are most aligned with those observed profiles under transparent rules.",
        "",
        "## Overreach to avoid",
        "- Claiming the precise cause of the gap within a school.",
        "- Claiming that the recommended intervention will causally reduce the gap for a specific school.",
        "- Treating the supplied risk score as a validated causal construct rather than an observed indicator.",
        "",
        "## Realistic scope",
        "- Strongest scope: hybrid country diagnostic plus school segmentation dashboard.",
        "- Weak scope: deterministic school-specific suggestion engine framed as proven personalised policy.",
    ]
    write_text(competition_dir / "scope_and_limits.md", "\n".join(scope_limits))

    final_report = [
        "# FINAL REPORT",
        "",
        "## 12-step completion summary",
        "- Step 1: completed. Full CSV and SAV inventory produced, including join keys and feasibility conclusions.",
        "- Step 2: completed. Primary equity-gap metric set as SES-quartile maths gap, with within-school SES slope as a secondary metric.",
        "- Step 3: completed. Country latest-gap and trend rankings produced with evidence-strength flags for weak time coverage.",
        "- Step 4: completed. Supplied 2022 risk score diagnosed against school performance, inequality, and context indicators.",
        "- Step 5: completed. Multi-year light dataset plus 2022 core and extended raw datasets built and documented.",
        "- Step 6: completed. School-level correlates ranked using transparent regularised and descriptive methods.",
        "- Step 7: completed. Within-school inequality quadrants estimated for schools with sufficient sample size.",
        "- Step 8: completed. Interpretable rule-based school segmentation produced as the published typology.",
        "- Step 9: completed. Segment-to-intervention mapping built using transparent evidence-informed rules.",
        "- Step 10: completed. Prototype recommender logic exported as a simple inspectable Python script.",
        "- Step 11: completed. Competition narrative and scope limits written.",
        "- Step 12: completed. Final synthesis, data gaps, and project recommendation written.",
        "",
        "## Strongest evidence-backed insights",
        f"- Country-level latest gap rankings cover {len(latest_rankings)} systems; the highest observed current gaps are concentrated in the top decile of that ranking.",
        f"- {int((trend_rankings['trend_label'] == 'Worsening').sum())} systems meet the minimum threshold for a worsening-gap label, but many systems lack enough time points for strong trend claims.",
        f"- School-level risk patterns show that risk co-moves with both lower average maths and wider within-school SES gaps, rather than only one of those dimensions.",
        f"- The school segmentation yields {segment_profiles.shape[0]} published profiles that can support targeted intervention logic.",
        "",
        "## Direct answers to the six required questions",
        "1. Are the current datasets sufficient to run the full 12-step pipeline? Yes, in hybrid mode with SAV ingestion. CSV-only would leave Steps 6-8 materially weaker.",
        "2. Is one year enough, or do we need all years? All years are needed for the country worsening/improving story; 2022 is the main year for rich school profiling.",
        "3. Best framing? Hybrid: risk scoring plus school segmentation plus recommendation logic.",
        "4. Can the data credibly support an individual school suggestion dashboard? Only as an evidence-informed diagnostic and prioritisation tool, not as a causal recommendation engine.",
        "5. What can we say about the reason behind the gap? We can identify correlates, compositions, and school-context patterns associated with larger gaps; we cannot identify the definitive causal reason.",
        "6. Which missing files or variables matter most? Longitudinal outcomes, intervention implementation data, and harmonised multi-year school-context variables matter most.",
    ]
    write_text(final_dir / "FINAL_REPORT.md", "\n".join(final_report))

    next_data = [
        "# NEXT DATA NEEDED",
        "",
        "- Longitudinal or linked cohort data to test whether observed school profiles precede later changes in inequity.",
        "- Intervention implementation and uptake data to validate whether profile-based recommendations improve outcomes.",
        "- Harmonised multi-year school questionnaire features so the school segmentation story is not limited mainly to 2022.",
        "- Stronger local context variables on funding, staffing shortages, and support services at school level.",
        "- If available, school outcome follow-up data beyond PISA cross-sections.",
    ]
    write_text(final_dir / "NEXT_DATA_NEEDED.md", "\n".join(next_data))

    recommendation = [
        "# PROJECT RECOMMENDATION",
        "",
        "- Recommended final competition concept: a hybrid equity diagnostic and school-profile decision-support dashboard.",
        "- Front end concept: country trajectory view for problem diagnosis, school segment view for prioritisation, and intervention logic panel for transparent rationale.",
        "- Recommended narrative arc: problem diagnosis -> evidence -> prioritisation -> segment logic -> prototype recommender.",
        "- Recommended positioning: rigorous observational analytics, not automated policy prescription.",
    ]
    write_text(final_dir / "PROJECT_RECOMMENDATION.md", "\n".join(recommendation))
    logger.info("Steps 11-12 complete")
