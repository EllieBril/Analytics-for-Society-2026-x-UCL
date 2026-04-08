from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .utils import format_num, write_text


def analyze_risk(school_profiles: pd.DataFrame, output_dir: Path, logger) -> dict[str, pd.DataFrame]:
    profiles = school_profiles.copy()
    profiles["risk_band"] = pd.qcut(
        profiles["EQUITY_RISK_SCORE"].rank(method="first"), q=5, labels=["Very low", "Low", "Middle", "High", "Very high"], duplicates="drop"
    )
    high_risk_cut = profiles["EQUITY_RISK_SCORE"].quantile(0.9)
    low_risk_cut = profiles["EQUITY_RISK_SCORE"].quantile(0.1)
    high_risk = profiles.loc[profiles["EQUITY_RISK_SCORE"] >= high_risk_cut].copy()
    low_risk = profiles.loc[profiles["EQUITY_RISK_SCORE"] <= low_risk_cut].copy()

    compare_vars = [
        "school_mean_math",
        "within_school_gap",
        "school_mean_escs",
        "low_perf_share_400",
        "STAFFSHORT",
        "EDUSHORT",
        "NEGSCLIM",
        "SCHAUTO",
        "student_belong_mean",
        "student_math_anxiety_mean",
        "teacher_response_count",
    ]
    diagnostics = []
    for col in compare_vars:
        if col in profiles.columns:
            diagnostics.append(
                {
                    "metric": col,
                    "high_risk_mean": high_risk[col].mean(),
                    "low_risk_mean": low_risk[col].mean(),
                    "difference_high_minus_low": high_risk[col].mean() - low_risk[col].mean(),
                    "corr_with_risk": profiles[["EQUITY_RISK_SCORE", col]].corr(numeric_only=True).iloc[0, 1],
                }
            )
    diagnostics_df = pd.DataFrame(diagnostics).sort_values("corr_with_risk", key=lambda s: s.abs(), ascending=False)

    high_risk_school_profiles = high_risk[
        [
            "CNT",
            "school_key",
            "EQUITY_RISK_SCORE",
            "school_mean_math",
            "within_school_gap",
            "school_mean_escs",
            "low_perf_share_400",
            "STAFFSHORT",
            "EDUSHORT",
            "NEGSCLIM",
            "student_belong_mean",
            "student_math_anxiety_mean",
            "TRAJECTORY",
        ]
    ].sort_values("EQUITY_RISK_SCORE", ascending=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    high_risk_school_profiles.to_csv(output_dir / "high_risk_school_profiles.csv", index=False)
    diagnostics_df.to_csv(output_dir / "risk_feature_diagnostics.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(profiles["EQUITY_RISK_SCORE"].dropna(), bins=40, color="#4c72b0", alpha=0.85)
    ax.axvline(high_risk_cut, color="#c44e52", linestyle="--", label="Top decile cut")
    ax.axvline(low_risk_cut, color="#55a868", linestyle="--", label="Bottom decile cut")
    ax.set_title("Distribution of Supplied Equity Risk Score")
    ax.set_xlabel("Risk score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "risk_score_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(profiles["within_school_gap"], profiles["school_mean_math"], c=profiles["EQUITY_RISK_SCORE"], cmap="viridis", alpha=0.65)
    ax.set_xlabel("Within-school SES-maths gap")
    ax.set_ylabel("School mean maths")
    ax.set_title("Risk Score vs School Performance and Inequality")
    fig.tight_layout()
    fig.savefig(output_dir / "risk_vs_performance_gap.png", dpi=150)
    plt.close(fig)

    strongest = diagnostics_df.head(5)
    lines = [
        "# Risk Score Diagnostics",
        "",
        "## Completion",
        "- Status: fully completed.",
        "",
        "## Direct findings",
        f"- School risk file covers {profiles['CNT'].nunique()} countries and {profiles['school_key'].nunique()} schools.",
        f"- High-risk top decile threshold: {format_num(high_risk_cut)}.",
        f"- Low-risk bottom decile threshold: {format_num(low_risk_cut)}.",
        "",
        "## Evidence-based interpretation",
    ]
    for _, row in strongest.iterrows():
        lines.append(
            f"- `{row['metric']}` correlates with risk at {format_num(row['corr_with_risk'], 3)}; high-risk minus low-risk mean difference is {format_num(row['difference_high_minus_low'], 2)}."
        )
    lines += [
        "",
        "## Caution",
        "- The supplied risk score is treated as an observed input, not a validated causal construct.",
        "- Diagnostic patterns suggest whether the score aligns more with low performance, inequality, or context strain, but they do not reveal the original score formula.",
    ]
    write_text(output_dir / "risk_score_diagnostics.md", "\n".join(lines))
    logger.info("Step 4 complete: high-risk schools=%s", len(high_risk_school_profiles))
    return {
        "school_profiles": profiles,
        "high_risk_school_profiles": high_risk_school_profiles,
        "risk_diagnostics": diagnostics_df,
    }
