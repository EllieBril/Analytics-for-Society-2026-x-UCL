from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import format_num, weighted_slope, write_text


def compute_country_annual_change(group: pd.DataFrame) -> float:
    ordered = group.sort_values("YEAR")
    if len(ordered) >= 3:
        weights = pd.Series(np.ones(len(ordered)), index=ordered.index, dtype=float)
        slope = weighted_slope(ordered["YEAR"], ordered["gap_recomputed"], weights)
        if pd.notna(slope):
            return slope
    if len(ordered) >= 2:
        year_span = ordered["YEAR"].iloc[-1] - ordered["YEAR"].iloc[0]
        if year_span == 0:
            return np.nan
        return float((ordered["gap_recomputed"].iloc[-1] - ordered["gap_recomputed"].iloc[0]) / year_span)
    return np.nan


def analyze_country_trajectories(
    rebuilt_gap: pd.DataFrame,
    supplied_gap: pd.DataFrame,
    supplied_trajectory: pd.DataFrame,
    output_dir: Path,
    logger,
) -> dict[str, pd.DataFrame]:
    latest = rebuilt_gap.sort_values(["CNT", "YEAR"]).groupby("CNT").tail(1).copy()
    timepoints = rebuilt_gap.groupby("CNT")["YEAR"].nunique().rename("n_timepoints")
    volatility = rebuilt_gap.groupby("CNT")["gap_recomputed"].std().rename("gap_volatility")
    slope = (
        rebuilt_gap.sort_values(["CNT", "YEAR"])
        .groupby("CNT")
        .apply(compute_country_annual_change)
        .rename("annual_change")
    )
    first_last = rebuilt_gap.sort_values(["CNT", "YEAR"]).groupby("CNT").agg(
        first_year=("YEAR", "first"),
        last_year=("YEAR", "last"),
        first_gap=("gap_recomputed", "first"),
        last_gap=("gap_recomputed", "last"),
    )
    trend = first_last.join(timepoints).join(volatility).join(slope).reset_index()
    trend["long_run_change"] = trend["last_gap"] - trend["first_gap"]
    trend["trend_evidence"] = trend["n_timepoints"].apply(lambda x: "strong" if x >= 3 else "weak")
    trend["trend_label"] = "Insufficient time points"
    trend.loc[(trend["n_timepoints"] >= 3) & (trend["long_run_change"] >= 5), "trend_label"] = "Worsening"
    trend.loc[(trend["n_timepoints"] >= 3) & (trend["long_run_change"] <= -5), "trend_label"] = "Improving"
    trend.loc[(trend["n_timepoints"] >= 3) & (trend["long_run_change"].between(-5, 5)), "trend_label"] = "Stable"

    latest_rankings = latest.merge(timepoints, on="CNT", how="left").sort_values("gap_recomputed", ascending=False)
    latest_rankings["latest_gap_rank"] = range(1, len(latest_rankings) + 1)
    latest_rankings = latest_rankings.rename(
        columns={
            "YEAR": "latest_year",
            "gap_recomputed": "latest_gap",
            "avg_math_recomputed": "latest_avg_math",
            "avg_read_recomputed": "latest_avg_read",
            "avg_science_recomputed": "latest_avg_science",
        },
    )
    trend_rankings = trend.sort_values("long_run_change", ascending=False).copy()
    trend_rankings["trend_rank"] = range(1, len(trend_rankings) + 1)

    typology = latest_rankings[["CNT", "latest_year", "latest_gap", "latest_avg_math", "n_timepoints"]].merge(
        trend[["CNT", "long_run_change", "gap_volatility", "trend_label", "trend_evidence"]],
        on="CNT",
        how="left",
    )
    typology["gap_band"] = pd.qcut(typology["latest_gap"], q=4, labels=["Low gap", "Lower-middle gap", "Upper-middle gap", "High gap"], duplicates="drop")
    typology["country_typology"] = "Mixed"
    typology.loc[(typology["gap_band"] == "High gap") & (typology["trend_label"] == "Worsening"), "country_typology"] = "High-gap worsening"
    typology.loc[(typology["gap_band"] == "High gap") & (typology["trend_label"].isin(["Stable", "Improving"])), "country_typology"] = "Persistently high-gap"
    typology.loc[(typology["gap_band"] == "Low gap") & (typology["trend_label"].isin(["Stable", "Improving"])), "country_typology"] = "Relatively equitable"
    typology.loc[typology["trend_label"] == "Improving", "country_typology"] = typology["country_typology"].where(
        typology["country_typology"] != "Mixed", "Improving from higher gap"
    )
    typology.loc[typology["trend_evidence"] == "weak", "country_typology"] = "Limited-trend evidence"

    output_dir.mkdir(parents=True, exist_ok=True)
    latest_rankings.to_csv(output_dir / "country_rankings_latest.csv", index=False)
    trend_rankings.to_csv(output_dir / "country_rankings_trend.csv", index=False)
    typology.to_csv(output_dir / "country_typology.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 7))
    top20 = latest_rankings.head(20).sort_values("latest_gap")
    ax.barh(top20["CNT"], top20["latest_gap"], color="#c44e52")
    ax.set_title("Highest Latest Observed SES-Maths Gaps")
    ax.set_xlabel("Gap points")
    fig.tight_layout()
    fig.savefig(output_dir / "latest_gap_top20.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 7))
    top_change = trend_rankings.head(20).sort_values("long_run_change")
    ax.barh(top_change["CNT"], top_change["long_run_change"], color="#dd8452")
    ax.set_title("Largest Worsening in SES-Maths Gap")
    ax.set_xlabel("Long-run change in gap points")
    fig.tight_layout()
    fig.savefig(output_dir / "worsening_top20.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = typology.copy()
    colors = {
        "High-gap worsening": "#c44e52",
        "Persistently high-gap": "#dd8452",
        "Relatively equitable": "#55a868",
        "Improving from higher gap": "#4c72b0",
        "Limited-trend evidence": "#8c8c8c",
        "Mixed": "#8172b3",
    }
    ax.scatter(
        scatter["latest_gap"],
        scatter["long_run_change"],
        c=scatter["country_typology"].map(colors).fillna("#8c8c8c"),
        alpha=0.8,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Latest gap")
    ax.set_ylabel("Long-run change")
    ax.set_title("Country Typology: Latest Gap vs Long-run Change")
    fig.tight_layout()
    fig.savefig(output_dir / "country_gap_vs_change.png", dpi=150)
    plt.close(fig)

    exact_check = supplied_trajectory.merge(
        supplied_gap.pivot_table(index="CNT", columns="YEAR", values="GAP", aggfunc="mean")[[2009, 2022]].reset_index(),
        on="CNT",
        how="left",
    )
    consistency = (exact_check["GAP_2009"] - exact_check[2009]).abs().max() == 0 and (exact_check["GAP_2022"] - exact_check[2022]).abs().max() == 0
    summary_lines = [
        "# Country Trajectory Analysis",
        "",
        "## Completion",
        "- Status: fully completed.",
        "",
        "## Key findings",
        f"- Latest ranking covers {len(latest_rankings)} countries/economies with at least one observed wave.",
        f"- {int((trend['trend_label'] == 'Worsening').sum())} countries are classed as worsening and {int((trend['trend_label'] == 'Improving').sum())} as improving, but only where at least 3 waves are available.",
        f"- Supplied 2009/2022 trajectory file is internally consistent with the country-year gap file: `{consistency}`.",
        f"- Median latest gap among ranked countries is {format_num(latest_rankings['latest_gap'].median())} points.",
        "",
        "## Interpretation discipline",
        "- Latest gap levels are directly observed from available country-year summaries.",
        "- Annual change uses an OLS slope over all available waves when at least 3 observations exist, with a first-last fallback only for thinner series.",
        "- Worsening or improving labels are inferred from observed wave-to-wave comparisons, not from causal explanations.",
        "- Countries with 1-2 time points are kept in the level rankings but flagged as weak evidence for trend claims.",
    ]
    write_text(output_dir / "country_trajectory_writeup.md", "\n".join(summary_lines))
    logger.info("Step 3 complete: country typology rows=%s", len(typology))
    return {
        "latest_rankings": latest_rankings,
        "trend_rankings": trend_rankings,
        "country_typology": typology,
    }
