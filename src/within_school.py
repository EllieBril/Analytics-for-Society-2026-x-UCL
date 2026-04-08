from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .utils import format_num, write_text


def analyze_within_school(school_profiles: pd.DataFrame, output_dir: Path, logger) -> dict[str, pd.DataFrame]:
    valid = school_profiles.loc[
        (school_profiles["student_count"] >= 20)
        & school_profiles["within_school_gap"].notna()
        & school_profiles["school_mean_math"].notna()
    ].copy()
    perf_cut = valid["school_mean_math"].median()
    gap_cut = valid["within_school_gap"].median()
    valid["performance_band"] = valid["school_mean_math"].ge(perf_cut).map({True: "High performance", False: "Low performance"})
    valid["inequality_band"] = valid["within_school_gap"].ge(gap_cut).map({True: "High inequality", False: "Low inequality"})
    valid["school_quadrant"] = valid["performance_band"] + " / " + valid["inequality_band"]

    school_quadrants = valid[
        [
            "CNT",
            "school_key",
            "school_mean_math",
            "within_school_gap",
            "within_school_ses_slope",
            "school_mean_escs",
            "low_perf_share_400",
            "EQUITY_RISK_SCORE",
            "school_quadrant",
            "NEGSCLIM",
            "STAFFSHORT",
            "EDUSHORT",
            "student_belong_mean",
            "student_math_anxiety_mean",
        ]
    ].copy()

    quadrant_profiles = valid.groupby("school_quadrant")[
        [
            "school_mean_math",
            "within_school_gap",
            "school_mean_escs",
            "low_perf_share_400",
            "EQUITY_RISK_SCORE",
            "NEGSCLIM",
            "STAFFSHORT",
            "EDUSHORT",
            "student_belong_mean",
            "student_math_anxiety_mean",
        ]
    ].mean().reset_index()

    output_dir.mkdir(parents=True, exist_ok=True)
    school_quadrants.to_csv(output_dir / "school_quadrants.csv", index=False)
    quadrant_profiles.to_csv(output_dir / "quadrant_profiles.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {
        "High performance / Low inequality": "#55a868",
        "High performance / High inequality": "#dd8452",
        "Low performance / Low inequality": "#8172b3",
        "Low performance / High inequality": "#c44e52",
    }
    ax.scatter(
        valid["within_school_gap"],
        valid["school_mean_math"],
        c=valid["school_quadrant"].map(colors),
        alpha=0.7,
    )
    ax.axhline(perf_cut, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(gap_cut, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Within-school SES-maths gap")
    ax.set_ylabel("School mean maths")
    ax.set_title("Within-school inequality quadrants")
    fig.tight_layout()
    fig.savefig(output_dir / "within_school_quadrants.png", dpi=150)
    plt.close(fig)

    lines = [
        "# Within-School Gap Report",
        "",
        "## Completion",
        "- Status: fully completed.",
        "",
        "## Design",
        "- Schools are included only when at least 20 students are available and the within-school SES gap can be estimated.",
        f"- Performance split uses the sample median school mean maths of {format_num(perf_cut)}.",
        f"- Inequality split uses the sample median within-school gap of {format_num(gap_cut)}.",
        "",
        "## Findings",
    ]
    for _, row in quadrant_profiles.iterrows():
        lines.append(
            f"- `{row['school_quadrant']}`: mean maths {format_num(row['school_mean_math'])}, within-school gap {format_num(row['within_school_gap'])}, risk {format_num(row['EQUITY_RISK_SCORE'])}, negative climate {format_num(row['NEGSCLIM'], 2)}."
        )
    lines += [
        "",
        "## Interpretation rule",
        "- Same-school inequality patterns show which school contexts are associated with more unequal outcomes among students in the same institution.",
        "- They do not identify the causal reason for the gap.",
    ]
    write_text(output_dir / "within_school_gap_report.md", "\n".join(lines))
    logger.info("Step 7 complete: quadrant schools=%s", len(school_quadrants))
    return {"school_quadrants": school_quadrants, "quadrant_profiles": quadrant_profiles}
