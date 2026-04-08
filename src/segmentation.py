from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import format_num, write_text


SEGMENT_FEATURES = [
    "EQUITY_RISK_SCORE",
    "school_mean_math",
    "within_school_gap",
    "school_mean_escs",
    "STAFFSHORT",
    "EDUSHORT",
    "NEGSCLIM",
    "student_belong_mean",
    "student_math_anxiety_mean",
]


SEGMENT_DESCRIPTIONS = {
    "Resilient equitable performers": "Higher maths performance with relatively lower within-school SES inequality.",
    "High-achievement unequal schools": "Higher maths performance but sizeable within-school SES inequality.",
    "Low-achievement broad support need": "Lower maths performance with broad achievement challenges rather than concentrated inequality.",
    "Strained high-inequality schools": "Lower maths performance combined with wider SES inequality and visible climate or resource strain.",
    "Digitally constrained schools": "Schools where digital access or device constraints stand out relative to the rest of the profile.",
    "Mixed profile": "Schools with a mixed signal that do not clearly fit the main archetypes.",
}


def _safe_row_mean(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    available = [col for col in cols if col in df.columns]
    if not available:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return df[available].mean(axis=1, skipna=True)


def assign_rule_based_segments(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["context_strain"] = _safe_row_mean(out, ["STAFFSHORT", "EDUSHORT", "NEGSCLIM"])
    out["digital_strain"] = _safe_row_mean(out, ["student_device_problem_mean", "student_internet_problem_mean"])

    performance_cut = out["school_mean_math"].median()
    gap_cut = out["within_school_gap"].median()
    risk_cut = out["EQUITY_RISK_SCORE"].median()
    context_cut = out["context_strain"].median()
    digital_cut = out["digital_strain"].quantile(0.75)

    segment = pd.Series("Mixed profile", index=out.index, dtype="string")

    segment.loc[(out["school_mean_math"] >= performance_cut) & (out["within_school_gap"] < gap_cut)] = "Resilient equitable performers"
    segment.loc[(out["school_mean_math"] >= performance_cut) & (out["within_school_gap"] >= gap_cut)] = "High-achievement unequal schools"
    segment.loc[(out["school_mean_math"] < performance_cut) & (out["within_school_gap"] < gap_cut)] = "Low-achievement broad support need"
    segment.loc[
        (out["school_mean_math"] < performance_cut)
        & (out["within_school_gap"] >= gap_cut)
        & ((out["context_strain"] >= context_cut) | (out["EQUITY_RISK_SCORE"] >= risk_cut))
    ] = "Strained high-inequality schools"
    segment.loc[
        segment.eq("Mixed profile") & out["digital_strain"].notna() & (out["digital_strain"] >= digital_cut)
    ] = "Digitally constrained schools"

    out["rule_segment"] = segment
    out["performance_cut"] = performance_cut
    out["gap_cut"] = gap_cut
    out["risk_cut"] = risk_cut
    out["context_cut"] = context_cut
    out["digital_cut"] = digital_cut
    return out


def analyze_segmentation(school_profiles: pd.DataFrame, output_dir: Path, logger) -> dict[str, pd.DataFrame]:
    base = school_profiles.loc[
        school_profiles["EQUITY_RISK_SCORE"].notna()
        & school_profiles["school_mean_math"].notna()
        & school_profiles["within_school_gap"].notna()
    ].copy()
    for col in ["student_device_problem_mean", "student_internet_problem_mean"]:
        if col not in base.columns:
            base[col] = np.nan

    base = assign_rule_based_segments(base)
    base["official_segment"] = base["rule_segment"]
    base["published_method"] = "rule_based"
    base["clustering_status"] = "not_published"

    segment_profile_cols = [col for col in SEGMENT_FEATURES if col in base.columns]
    segment_profiles = (
        base.groupby("official_segment", dropna=False)[segment_profile_cols]
        .mean()
        .reset_index()
        .sort_values(["EQUITY_RISK_SCORE", "within_school_gap"], ascending=[False, False], na_position="last")
    )
    segment_sizes = base.groupby("official_segment", dropna=False).size().rename("segment_size").reset_index()
    segment_profiles = segment_profiles.merge(segment_sizes, on="official_segment", how="left")
    segment_profiles["segment_description"] = segment_profiles["official_segment"].map(SEGMENT_DESCRIPTIONS)

    output_dir.mkdir(parents=True, exist_ok=True)
    assignment_cols = [
        "CNT",
        "school_key",
        "official_segment",
        "rule_segment",
        "published_method",
        "clustering_status",
        "EQUITY_RISK_SCORE",
        "school_mean_math",
        "within_school_gap",
        "school_mean_escs",
        "context_strain",
        "digital_strain",
        "performance_cut",
        "gap_cut",
    ]
    keep_assignment_cols = [col for col in assignment_cols if col in base.columns]
    base[keep_assignment_cols].to_csv(output_dir / "segment_assignments.csv", index=False)
    segment_profiles.to_csv(output_dir / "segment_profiles.csv", index=False)

    segment_counts = base["official_segment"].value_counts(dropna=False)
    method_lines = [
        "# Segmentation Method Note",
        "",
        "## Completion",
        "- Status: fully completed.",
        "",
        "## Published method",
        "- Official published segmentation: `rule_based`.",
        "- Unit of segmentation: schools.",
        "- Core inputs: performance, within-school SES gap, overall risk, SES composition, and climate/resource strain proxies.",
        "",
        "## Why rule-based was preferred",
        "- The competition use case needs interpretable profiles that can be mapped to transparent intervention logic.",
        "- The school profiles already support a stable rule-based typology without relying on opaque optimisation.",
        "- Earlier unsupervised clustering attempts were not retained as the published method because they were less stable and less interpretable in this execution environment.",
        "",
        "## Published segment counts",
    ]
    for segment_name, count in segment_counts.items():
        method_lines.append(f"- `{segment_name}`: {int(count)} schools.")
    method_lines += [
        "",
        "## Interpretation",
        "- These segments are descriptive school archetypes, not causal types.",
        "- They support prioritisation and tailored recommendation logic, not deterministic prescriptions for a named school.",
        "",
        "## Limitation",
        "- This step deliberately favours interpretability over technical clustering novelty. That is appropriate for the current observational data and competition narrative.",
    ]
    write_text(output_dir / "segmentation_method_note.md", "\n".join(method_lines))
    logger.info("Step 8 complete: published rule-based segmentation for %s schools", len(base))
    return {
        "segment_assignments": base,
        "segment_profiles": segment_profiles,
        "official_method": pd.DataFrame(
            [
                {
                    "official_method": "rule_based",
                    "comparison_status": "unsupervised_clustering_not_published",
                    "school_count": len(base),
                    "segment_count": base["official_segment"].nunique(dropna=True),
                    "performance_cut": base["performance_cut"].iloc[0] if len(base) else np.nan,
                    "gap_cut": base["gap_cut"].iloc[0] if len(base) else np.nan,
                }
            ]
        ),
    }
