from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
V7_DIR = ROOT / "v7"
REQUIRED_OUTPUT_FILES = [
    OUTPUTS / "03_country_trajectories" / "country_rankings_latest.csv",
    OUTPUTS / "03_country_trajectories" / "country_rankings_trend.csv",
    OUTPUTS / "03_country_trajectories" / "country_typology.csv",
    OUTPUTS / "02_equity_definition" / "computed_equity_gap_by_country_year.csv",
    OUTPUTS / "08_segmentation" / "segment_profiles.csv",
    OUTPUTS / "09_intervention_mapping" / "segment_to_intervention_map.csv",
    OUTPUTS / "10_prototype" / "prototype_examples.csv",
    OUTPUTS / "10_prototype" / "dashboard_school_view.csv",
    OUTPUTS / "06_correlates" / "candidate_driver_rankings.csv",
    OUTPUTS / "11_competition_fit" / "competition_narrative.md",
    OUTPUTS / "11_competition_fit" / "scope_and_limits.md",
    OUTPUTS / "12_final" / "FINAL_REPORT.md",
]

SEGMENT_COLORS = {
    "Resilient equitable performers": "#2C6E49",
    "High-achievement unequal schools": "#D96C06",
    "Strained high-inequality schools": "#B33F40",
    "Low-achievement broad support need": "#2F6690",
    "Digitally constrained schools": "#6B5CA5",
    "Mixed profile": "#8C7A6B",
}

TYPOLOGY_COLORS = {
    "High-gap worsening": "#B33F40",
    "Persistently high-gap": "#D96C06",
    "Improving but still high-gap": "#C2841A",
    "Already more equitable": "#2C6E49",
    "Low-gap but worsening": "#7F5539",
    "Mixed / weak trend evidence": "#2F6690",
}

OUTCOME_LABELS = {
    "school_mean_math": "Average school maths",
    "school_risk_score": "Supplied equity risk score",
    "within_school_gap": "Within-school SES gap",
}

PROFILE_METRICS = {
    "EQUITY_RISK_SCORE": "Risk score",
    "school_mean_math": "Mean maths",
    "within_school_gap": "Within-school gap",
    "school_mean_escs": "Mean ESCS",
    "low_perf_share_400": "Low-perf share <400",
}

COST_LABELS = {
    1: "Low implementation burden",
    2: "Moderate implementation burden",
    3: "Higher implementation burden",
}


def classify_risk(score: float, thresholds: tuple[float, float, float] | None = None) -> str:
    moderate_cut, high_cut, very_high_cut = thresholds or (40, 55, 70)
    if pd.isna(score):
        return "Unknown"
    if score >= very_high_cut:
        return "Very high"
    if score >= high_cut:
        return "High"
    if score >= moderate_cut:
        return "Moderate"
    return "Lower"


def compute_risk_thresholds(scores: pd.Series) -> tuple[float, float, float]:
    valid = scores.dropna().astype(float)
    if valid.empty:
        return (40.0, 55.0, 70.0)
    moderate_cut = float(valid.quantile(0.4))
    high_cut = float(valid.quantile(0.6))
    very_high_cut = float(valid.quantile(0.8))
    return (moderate_cut, high_cut, very_high_cut)


def validate_required_outputs() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED_OUTPUT_FILES if not path.exists()]
    if missing:
        joined = "\n".join(f"- `{item}`" for item in missing)
        raise FileNotFoundError(
            "Dashboard outputs are missing. Run the pipeline first:\n"
            "`python3 src/run_pipeline.py --data-root . --focus-year 2022 --min-school-n 20 --seed 42`\n\n"
            f"Missing files:\n{joined}"
        )


def apply_page_style() -> None:
    st.set_page_config(
        page_title="SDG 4 Equity Diagnostic",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root {
            --paper: #f6efe5;
            --ink: #17332d;
            --muted: #5c5148;
            --accent: #cf5c36;
            --accent-deep: #9f3d1d;
            --teal: #2f6f77;
            --sage: #dbe6df;
            --panel: rgba(255, 255, 255, 0.80);
        }
        .stApp {
            background:
                radial-gradient(circle at 100% 0%, rgba(207, 92, 54, 0.16), transparent 26%),
                radial-gradient(circle at 0% 20%, rgba(47, 111, 119, 0.12), transparent 22%),
                linear-gradient(180deg, #f3ecdf 0%, #faf7f1 48%, #ffffff 100%);
            color: var(--ink);
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }
        .hero-panel {
            padding: 1.5rem 1.75rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #17332d 0%, #2f6f77 100%);
            color: #f8f4ed;
            box-shadow: 0 18px 40px rgba(23, 51, 45, 0.18);
            margin-bottom: 1rem;
        }
        .hero-kicker {
            font-size: 0.85rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            opacity: 0.78;
            margin-bottom: 0.4rem;
        }
        .hero-title {
            font-size: 2.2rem;
            line-height: 1.05;
            font-weight: 700;
            margin-bottom: 0.6rem;
            font-family: "Avenir Next", "Trebuchet MS", sans-serif;
        }
        .hero-copy {
            font-size: 1rem;
            line-height: 1.55;
            max-width: 72rem;
            color: rgba(248, 244, 237, 0.94);
        }
        .metric-card {
            background: var(--panel);
            border: 1px solid rgba(23, 51, 45, 0.08);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 28px rgba(23, 51, 45, 0.08);
            min-height: 7.5rem;
        }
        .metric-label {
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 0.45rem;
        }
        .metric-value {
            font-size: 2rem;
            line-height: 1.0;
            color: var(--accent-deep);
            font-weight: 700;
            margin-bottom: 0.45rem;
            font-family: "Avenir Next", "Trebuchet MS", sans-serif;
        }
        .metric-caption {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.35;
        }
        .panel-note {
            background: rgba(219, 230, 223, 0.76);
            border-left: 4px solid #2c6e49;
            padding: 0.95rem 1rem;
            border-radius: 14px;
            margin: 0.5rem 0 1rem 0;
            color: var(--ink);
        }
        .warning-note {
            background: rgba(207, 92, 54, 0.10);
            border-left: 4px solid #cf5c36;
            padding: 0.95rem 1rem;
            border-radius: 14px;
            margin: 0.5rem 0 1rem 0;
            color: var(--ink);
        }
        .small-muted {
            color: var(--muted);
            font-size: 0.9rem;
        }
        div[data-testid="stTabs"] button {
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def parse_v7_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8", errors="ignore")
    patterns = {
        "grouped_cv_r2": r"Grouped 10-fold CV R2\s*:\s*([0-9.]+)\s*\+/-\s*([0-9.]+)",
        "holdout_r2": r"Hold-out R2\s*:\s*([0-9.]+)",
        "valid_clusters": r"With valid slope\s*:\s*([0-9]+)",
        "countries": r"Countries\s*:\s*([0-9]+)",
    }

    result: dict[str, object] = {"raw_text": text}
    grouped = re.search(patterns["grouped_cv_r2"], text)
    if grouped:
        result["grouped_cv_r2"] = float(grouped.group(1))
        result["grouped_cv_sd"] = float(grouped.group(2))
    holdout = re.search(patterns["holdout_r2"], text)
    if holdout:
        result["holdout_r2"] = float(holdout.group(1))
    valid = re.search(patterns["valid_clusters"], text)
    if valid:
        result["valid_clusters"] = int(valid.group(1))
    countries = re.search(patterns["countries"], text)
    if countries:
        result["countries"] = int(countries.group(1))
    return result


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> dict[str, object]:
    validate_required_outputs()
    latest = pd.read_csv(OUTPUTS / "03_country_trajectories" / "country_rankings_latest.csv")
    trend = pd.read_csv(OUTPUTS / "03_country_trajectories" / "country_rankings_trend.csv")
    typology = pd.read_csv(OUTPUTS / "03_country_trajectories" / "country_typology.csv")
    gap_series = pd.read_csv(OUTPUTS / "02_equity_definition" / "computed_equity_gap_by_country_year.csv")
    segment_profiles = pd.read_csv(OUTPUTS / "08_segmentation" / "segment_profiles.csv")
    intervention_map = pd.read_csv(OUTPUTS / "09_intervention_mapping" / "segment_to_intervention_map.csv")
    prototype_examples = pd.read_csv(OUTPUTS / "10_prototype" / "prototype_examples.csv")
    school_view = pd.read_csv(OUTPUTS / "10_prototype" / "dashboard_school_view.csv")
    correlates = pd.read_csv(OUTPUTS / "06_correlates" / "candidate_driver_rankings.csv")
    intervention_library = safe_read_csv(ROOT / "intervention_library.csv")
    v7_shap = safe_read_csv(V7_DIR / "shap_importance_v7.csv")
    v7_interventions = safe_read_csv(V7_DIR / "intervention_ranking_v7.csv")
    v7_summary = parse_v7_summary(V7_DIR / "v7_results")
    narrative = (OUTPUTS / "11_competition_fit" / "competition_narrative.md").read_text(encoding="utf-8")
    scope_limits = (OUTPUTS / "11_competition_fit" / "scope_and_limits.md").read_text(encoding="utf-8")
    final_report = (OUTPUTS / "12_final" / "FINAL_REPORT.md").read_text(encoding="utf-8")

    for frame in [school_view, prototype_examples]:
        if "school_key" in frame.columns:
            frame["school_key"] = frame["school_key"].astype("string")
    for frame in [latest, trend, typology, gap_series, school_view]:
        if "CNT" in frame.columns:
            frame["CNT"] = frame["CNT"].astype("string")

    country_summary = typology.merge(
        trend[["CNT", "annual_change", "trend_rank"]],
        on="CNT",
        how="left",
    )
    risk_thresholds = compute_risk_thresholds(school_view["EQUITY_RISK_SCORE"])
    school_view["risk_classification"] = school_view["EQUITY_RISK_SCORE"].map(lambda score: classify_risk(score, thresholds=risk_thresholds))

    intervention_summary = (
        intervention_map.sort_values(["segment", "category_rank"])
        .groupby("segment")
        .agg(
            interventions=("intervention", lambda s: " | ".join(s.astype(str))),
            categories=("recommended_category", lambda s: " | ".join(s.astype(str))),
        )
        .reset_index()
    )
    school_view = school_view.merge(
        intervention_summary,
        left_on="official_segment",
        right_on="segment",
        how="left",
    )

    return {
        "latest": latest,
        "trend": trend,
        "typology": typology,
        "country_summary": country_summary,
        "gap_series": gap_series,
        "school_view": school_view,
        "segment_profiles": segment_profiles,
        "intervention_map": intervention_map,
        "intervention_library": intervention_library,
        "prototype_examples": prototype_examples,
        "correlates": correlates,
        "narrative": narrative,
        "scope_limits": scope_limits,
        "final_report": final_report,
        "v7_shap": v7_shap,
        "v7_interventions": v7_interventions,
        "v7_summary": v7_summary,
        "risk_thresholds": risk_thresholds,
    }


def metric_card(label: str, value: str, caption: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-caption">{caption}</div>
    </div>
    """


def render_metric_row(cards: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(cards))
    for col, (label, value, caption) in zip(cols, cards):
        with col:
            st.markdown(metric_card(label, value, caption), unsafe_allow_html=True)


def sample_for_plot(df: pd.DataFrame, max_rows: int = 4000) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(max_rows, random_state=42)


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-kicker">SDG 4 Decision Support Prototype</div>
            <div class="hero-title">Equity diagnostic, school typology, and intervention prioritisation</div>
            <div class="hero-copy">
                This dashboard turns the completed pipeline into a competition-ready prototype:
                identify where SES-linked achievement gaps are largest or worsening, inspect school-level
                profiles, and translate those profiles into evidence-informed intervention priorities.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="warning-note">
            Strongest use: evidence-informed prioritisation. This app supports diagnosis and triage;
            it does not identify causal reasons for a given school's gap and it does not prove that a
            recommended intervention will work for one named school.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(data: dict[str, object]) -> None:
    country_summary = data["country_summary"]
    school_view = data["school_view"]
    st.sidebar.markdown("## Dashboard scope")
    st.sidebar.markdown(
        f"""
        - Systems covered: `{country_summary['CNT'].nunique()}`
        - 2022 school profiles: `{school_view['school_key'].nunique():,}`
        - Published school segments: `{school_view['official_segment'].nunique()}`
        - Intervention profile rules: `{data['intervention_map']['segment'].nunique()}`
        """
    )
    st.sidebar.markdown("## Recommended flow")
    st.sidebar.markdown(
        """
        1. Check which countries are high-gap or worsening.
        2. Inspect school segments and their risk/performance pattern.
        3. Use the observed-profile recommender to see profile-aligned priorities.
        4. Use the scenario planner only for hypothetical planning.
        5. Read the evidence tab before making any stronger claim.
        """
    )
    st.sidebar.markdown("## Privacy and safe use")
    st.sidebar.markdown(
        """
        - School IDs shown here are anonymised study IDs.
        - No named student-level records are exposed in the interface.
        - Do not present any output as a guaranteed causal school prescription.
        """
    )


def render_overview(data: dict[str, object]) -> None:
    country_summary = data["country_summary"]
    school_view = data["school_view"]
    segment_profiles = data["segment_profiles"]
    intervention_map = data["intervention_map"]

    render_metric_row(
        [
            ("Systems", f"{country_summary['CNT'].nunique()}", "Countries or systems in the cross-country trend view."),
            ("Segmented schools", f"{school_view['school_key'].nunique():,}", "Schools with complete enough 2022 profile data for segmentation."),
            ("School segments", f"{segment_profiles['official_segment'].nunique()}", "Published interpretable archetypes used for targeting."),
            ("Recommendation rules", f"{intervention_map['segment'].nunique()}", "Distinct school profiles linked to tailored intervention categories."),
        ]
    )

    st.markdown(
        """
        <div class="panel-note">
            Final representable recommended for the competition: a dashboard prototype with interactive visuals inside it.
            A pure chart deck would show patterns, but this structure also supports a decision: where to prioritise,
            what profile a school resembles, and which intervention category is most plausible for that profile.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 0.95])
    with left:
        top_gap = country_summary.sort_values("latest_gap", ascending=False).head(12)
        fig = px.bar(
            top_gap,
            x="latest_gap",
            y="CNT",
            orientation="h",
            color="country_typology",
            color_discrete_map=TYPOLOGY_COLORS,
            labels={"latest_gap": "Latest SES-maths gap", "CNT": "System"},
            title="Highest observed current equity gaps",
        )
        fig.update_layout(height=430, yaxis={"categoryorder": "total ascending"}, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        segment_counts = school_view["official_segment"].value_counts().rename_axis("official_segment").reset_index(name="school_count")
        fig = px.bar(
            segment_counts,
            x="official_segment",
            y="school_count",
            color="official_segment",
            color_discrete_map=SEGMENT_COLORS,
            title="Published school segments",
            labels={"official_segment": "School segment", "school_count": "Schools"},
        )
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def render_country_view(data: dict[str, object]) -> None:
    country_summary = data["country_summary"].copy()
    gap_series = data["gap_series"].copy()

    st.subheader("Country diagnostic")
    st.caption("Use this view for the problem-diagnosis part of the narrative: where is the equity gap currently largest, and where is it worsening with enough evidence?")

    filter_col1, filter_col2, filter_col3 = st.columns([1.2, 1.2, 0.8])
    with filter_col1:
        typology_options = sorted(country_summary["country_typology"].dropna().unique().tolist())
        typology_filter = st.multiselect("Country typology", options=typology_options, default=typology_options)
    with filter_col2:
        selected_country = st.selectbox("Focus system", options=sorted(country_summary["CNT"].tolist()), index=0)
    with filter_col3:
        top_n = st.slider("Top N ranking rows", min_value=5, max_value=20, value=10)

    if not typology_filter:
        st.info("Select at least one country typology to populate this view.")
        return

    filtered = country_summary.loc[country_summary["country_typology"].isin(typology_filter)].copy()
    if filtered.empty:
        st.info("No countries match the current filter.")
        return
    selected_country_row = country_summary.loc[country_summary["CNT"] == selected_country].iloc[0]

    render_metric_row(
        [
            ("Focus system", selected_country, selected_country_row["country_typology"]),
            ("Latest gap", f"{selected_country_row['latest_gap']:.1f}", f"Observed in {int(selected_country_row['latest_year'])}."),
            ("Long-run change", f"{selected_country_row['long_run_change']:+.1f}", f"Trend label: {selected_country_row['trend_label']}."),
            ("Trend evidence", selected_country_row["trend_evidence"], f"Time points available: {int(selected_country_row['n_timepoints'])}."),
        ]
    )

    left, right = st.columns([1.05, 0.95])
    with left:
        scatter = px.scatter(
            filtered,
            x="latest_gap",
            y="long_run_change",
            color="country_typology",
            size="n_timepoints",
            hover_name="CNT",
            hover_data={
                "annual_change": ":.2f",
                "gap_volatility": ":.1f",
                "latest_avg_math": ":.1f",
                "country_typology": True,
            },
            color_discrete_map=TYPOLOGY_COLORS,
            title="Latest observed gap vs long-run change",
            labels={"latest_gap": "Latest SES-maths gap", "long_run_change": "Change since first observed year"},
        )
        scatter.add_hline(y=0, line_dash="dot", line_color="#5c5148")
        scatter.add_vline(x=float(filtered["latest_gap"].median()), line_dash="dot", line_color="#5c5148")
        scatter.update_layout(height=460, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(scatter, use_container_width=True)
    with right:
        country_line = gap_series.loc[gap_series["CNT"] == selected_country].sort_values("YEAR")
        line = px.line(
            country_line,
            x="YEAR",
            y="gap_recomputed",
            markers=True,
            title=f"{selected_country}: gap trajectory over available waves",
            labels={"gap_recomputed": "SES-maths gap", "YEAR": "PISA wave"},
        )
        line.update_traces(line_color="#cf5c36", marker_color="#17332d")
        line.update_layout(height=460, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(line, use_container_width=True)

    ranking_cols = st.columns(2)
    with ranking_cols[0]:
        st.markdown("**Highest current gaps**")
        st.dataframe(
            filtered.sort_values("latest_gap", ascending=False).head(top_n)[["CNT", "latest_year", "latest_gap", "trend_label", "country_typology"]],
            use_container_width=True,
            hide_index=True,
        )
    with ranking_cols[1]:
        st.markdown("**Largest worsening where trend evidence is strong**")
        worsening = filtered.loc[filtered["trend_label"] == "Worsening"].sort_values("long_run_change", ascending=False).head(top_n)
        st.dataframe(
            worsening[["CNT", "long_run_change", "gap_volatility", "trend_evidence", "country_typology"]],
            use_container_width=True,
            hide_index=True,
        )


def render_school_view(data: dict[str, object]) -> None:
    school_view = data["school_view"].copy()
    segment_profiles = data["segment_profiles"].copy()

    st.subheader("School segmentation")
    st.caption("This is the core prototype view for school-level targeting. Schools are anonymised study IDs, so this should be presented as a profile inspector rather than a named-school ranking tool.")
    st.caption("The performance and gap cut lines shown below are relative within-sample reference cuts from the published segmentation, not absolute policy benchmarks.")

    col1, col2, col3 = st.columns([1.0, 1.2, 0.9])
    with col1:
        country_options = ["All"] + sorted(school_view["CNT"].dropna().unique().tolist())
        country_filter = st.selectbox("Country filter", options=country_options, index=0)
    with col2:
        segment_options = sorted(school_view["official_segment"].dropna().unique().tolist())
        segment_filter = st.multiselect("School segments", options=segment_options, default=segment_options)
    with col3:
        risk_min = float(school_view["EQUITY_RISK_SCORE"].min())
        risk_max = float(school_view["EQUITY_RISK_SCORE"].max())
        risk_filter = st.slider("Risk score range", min_value=float(risk_min), max_value=float(risk_max), value=(float(risk_min), float(risk_max)))

    filtered = school_view.loc[
        school_view["official_segment"].isin(segment_filter)
        & school_view["EQUITY_RISK_SCORE"].between(risk_filter[0], risk_filter[1], inclusive="both")
    ].copy()
    if country_filter != "All":
        filtered = filtered.loc[filtered["CNT"] == country_filter].copy()
    if filtered.empty:
        st.info("No schools match the current filters. Expand the risk range or re-add segments.")
        return

    render_metric_row(
        [
            ("Filtered schools", f"{len(filtered):,}", "Segmentable schools matching current filters."),
            ("Median maths", f"{filtered['school_mean_math'].median():.1f}", "School mean maths within the filtered set."),
            ("Median within-school gap", f"{filtered['within_school_gap'].median():.1f}", "School-internal SES-maths spread."),
            ("Median risk score", f"{filtered['EQUITY_RISK_SCORE'].median():.1f}", "Supplied equity risk score within the filtered set."),
        ]
    )

    left, right = st.columns([1.05, 0.95])
    with left:
        plot_df = sample_for_plot(filtered, max_rows=5000)
        scatter = px.scatter(
            plot_df,
            x="school_mean_math",
            y="within_school_gap",
            color="official_segment",
            hover_name="school_key",
            hover_data={
                "CNT": True,
                "EQUITY_RISK_SCORE": ":.1f",
                "student_count": ":.0f",
            },
            color_discrete_map=SEGMENT_COLORS,
            title="School performance vs within-school SES gap",
            labels={"school_mean_math": "School mean maths", "within_school_gap": "Within-school SES gap"},
        )
        perf_cut = float(filtered["performance_cut"].dropna().median()) if "performance_cut" in filtered.columns else 0
        gap_cut = float(filtered["gap_cut"].dropna().median()) if "gap_cut" in filtered.columns else 0
        scatter.add_vline(x=perf_cut, line_dash="dot", line_color="#5c5148")
        scatter.add_hline(y=gap_cut, line_dash="dot", line_color="#5c5148")
        scatter.update_layout(height=500, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(scatter, use_container_width=True)
    with right:
        segment_counts = (
            filtered["official_segment"].value_counts().rename_axis("official_segment").reset_index(name="school_count")
        )
        fig = px.bar(
            segment_counts,
            x="official_segment",
            y="school_count",
            color="official_segment",
            color_discrete_map=SEGMENT_COLORS,
            title="Filtered segment distribution",
            labels={"official_segment": "Segment", "school_count": "Schools"},
        )
        fig.update_layout(height=500, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div class="panel-note">
            Published segments are intentionally interpretable. The project uses the rule-based typology as the official segmentation
            because it is easier to defend in a competition setting than opaque clustering output.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Published segment profiles**")
    show_cols = [
        "official_segment",
        "segment_size",
        "EQUITY_RISK_SCORE",
        "school_mean_math",
        "within_school_gap",
        "school_mean_escs",
        "segment_description",
    ]
    st.dataframe(segment_profiles[show_cols], use_container_width=True, hide_index=True)


def build_comparison_frame(selected_row: pd.Series, school_view: pd.DataFrame) -> pd.DataFrame:
    metrics = list(PROFILE_METRICS.keys())
    segment_frame = school_view.loc[school_view["official_segment"] == selected_row["official_segment"], metrics]
    country_frame = school_view.loc[school_view["CNT"] == selected_row["CNT"], metrics]

    rows = []
    for metric in metrics:
        rows.append(
            {
                "metric": PROFILE_METRICS[metric],
                "Selected school": selected_row.get(metric),
                "Segment median": segment_frame[metric].median(),
                "Country median": country_frame[metric].median(),
            }
        )
    comparison = pd.DataFrame(rows).melt(id_vars="metric", var_name="benchmark", value_name="value")
    return comparison


def infer_segment_from_profile(
    mean_math: float,
    gap: float,
    climate: float,
    digital_strain: float,
    performance_cut: float,
    gap_cut: float,
    climate_cut: float,
    digital_cut: float,
) -> str:
    if pd.isna(mean_math) or pd.isna(gap):
        return "Mixed profile"
    if mean_math >= performance_cut and gap < gap_cut:
        return "Resilient equitable performers"
    if mean_math >= performance_cut and gap >= gap_cut:
        return "High-achievement unequal schools"
    if mean_math < performance_cut and digital_strain >= digital_cut:
        return "Digitally constrained schools"
    if mean_math < performance_cut and gap >= gap_cut and climate >= climate_cut:
        return "Strained high-inequality schools"
    if mean_math < performance_cut:
        return "Low-achievement broad support need"
    return "Mixed profile"


def select_recommendations_for_segment(
    segment_name: str,
    intervention_map: pd.DataFrame,
    max_cost_rating: int | None = None,
) -> pd.DataFrame:
    recs = intervention_map.loc[intervention_map["segment"] == segment_name].sort_values("category_rank").copy()
    if max_cost_rating is not None:
        filtered = recs.loc[recs["cost_rating"] <= max_cost_rating].copy()
        if not filtered.empty:
            return filtered
    return recs


def render_observed_recommender_view(data: dict[str, object]) -> None:
    school_view = data["school_view"].copy()
    intervention_map = data["intervention_map"].copy()
    prototype_examples = data["prototype_examples"].copy()

    st.subheader("Observed profile recommender")
    st.caption("This pane uses observed school-profile metrics already produced by the pipeline. It is the safe recommendation mode for the competition narrative.")

    mode = st.radio("Selection mode", options=["Observed school profile", "Prototype example"], horizontal=True)

    if mode == "Observed school profile":
        country_options = sorted(school_view["CNT"].dropna().unique().tolist())
        country_pick = st.selectbox("Country", options=country_options, index=0)
        country_schools = school_view.loc[school_view["CNT"] == country_pick].sort_values("EQUITY_RISK_SCORE", ascending=False).copy()
        labels = (
            country_schools["school_key"].astype(str)
            + " | "
            + country_schools["official_segment"].astype(str)
            + " | risk "
            + country_schools["EQUITY_RISK_SCORE"].round(1).astype(str)
        )
        label_to_index = dict(zip(labels.tolist(), country_schools.index.tolist()))
        selected_label = st.selectbox("School profile ID", options=labels.tolist(), index=0)
        selected_row = school_view.loc[label_to_index.get(selected_label, country_schools.index[0])]
    else:
        labels = (
            prototype_examples["CNT"].astype(str)
            + " | "
            + prototype_examples["school_key"].astype(str)
            + " | "
            + prototype_examples["official_segment"].astype(str)
        )
        idx = st.selectbox("Prototype example", options=range(len(labels)), format_func=lambda i: labels.iloc[i])
        example = prototype_examples.iloc[int(idx)]
        match = school_view.loc[
            (school_view["CNT"] == str(example["CNT"]))
            & (school_view["school_key"] == str(example["school_key"]))
        ]
        selected_row = match.iloc[0] if not match.empty else school_view.iloc[0]

    render_metric_row(
        [
            ("School profile", str(selected_row["school_key"]), f"Country code: {selected_row['CNT']}."),
            ("Risk class", str(selected_row["risk_classification"]), f"Risk score {selected_row['EQUITY_RISK_SCORE']:.1f}."),
            ("Assigned segment", str(selected_row["official_segment"]), "Published typology from Step 8."),
            ("Trajectory context", str(selected_row.get("TRAJECTORY", "NA")), "Country-level trend label carried into the school profile view."),
        ]
    )

    comparison = build_comparison_frame(selected_row, school_view)
    left, right = st.columns([1.1, 0.9])
    with left:
        bar = px.bar(
            comparison,
            x="metric",
            y="value",
            color="benchmark",
            barmode="group",
            title="Selected profile vs segment and country medians",
            labels={"metric": "Metric", "value": "Value"},
            color_discrete_sequence=["#cf5c36", "#2f6f77", "#8c7a6b"],
        )
        bar.update_layout(height=440, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(bar, use_container_width=True)
    with right:
        recs = select_recommendations_for_segment(selected_row["official_segment"], intervention_map)
        st.markdown("**Recommended intervention mix**")
        for _, rec in recs.iterrows():
            st.markdown(
                metric_card(
                    rec["recommended_category"].replace("_", " ").title(),
                    rec["intervention"],
                    f"Evidence {int(rec['evidence'])}/5 | {COST_LABELS.get(int(rec['cost_rating']), 'Implementation burden not rated')} | Transparent rule-based match",
                ),
                unsafe_allow_html=True,
            )
        st.markdown(
            """
            <div class="panel-note">
                Recommendation logic is profile-based. It prioritises categories that fit the observed pattern
                of performance, within-school inequality, climate/resource strain, and overall risk.
                Numeric effect-size fields from the intervention library are not presented here as school-specific forecasts.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_scenario_planner_view(data: dict[str, object]) -> None:
    school_view = data["school_view"].copy()
    intervention_map = data["intervention_map"].copy()
    risk_thresholds = data["risk_thresholds"]

    st.subheader("Scenario planner")
    st.caption("This is a hypothetical planning mode. It is useful for demo conversations, but it should be presented separately from observed school diagnosis.")

    st.markdown(
        """
        <div class="warning-note">
            Scenario mode does not estimate causal impact. It classifies a hypothetical profile into the published typology
            and shows which intervention categories would be prioritised under the same transparent rule set.
        </div>
        """,
        unsafe_allow_html=True,
    )

    country_options = ["Global median"] + sorted(school_view["CNT"].dropna().unique().tolist())
    country_choice = st.selectbox("Reference context", options=country_options, index=0)
    context = school_view.copy() if country_choice == "Global median" else school_view.loc[school_view["CNT"] == country_choice].copy()
    if context.empty:
        context = school_view.copy()

    performance_cut = float(context["school_mean_math"].median())
    gap_cut = float(context["within_school_gap"].median())
    climate_cut = float(context["NEGSCLIM"].median()) if "NEGSCLIM" in context.columns else 0.0
    digital_cut = float(context["student_device_problem_mean"].median()) if "student_device_problem_mean" in context.columns else 0.0

    col1, col2, col3 = st.columns(3)
    with col1:
        mean_math = st.slider(
            "Hypothetical school mean maths",
            min_value=float(school_view["school_mean_math"].quantile(0.05)),
            max_value=float(school_view["school_mean_math"].quantile(0.95)),
            value=float(performance_cut),
        )
        mean_escs = st.slider(
            "Hypothetical school mean ESCS",
            min_value=float(school_view["school_mean_escs"].quantile(0.05)),
            max_value=float(school_view["school_mean_escs"].quantile(0.95)),
            value=float(context["school_mean_escs"].median()),
        )
    with col2:
        within_gap = st.slider(
            "Hypothetical within-school SES gap",
            min_value=float(school_view["within_school_gap"].quantile(0.05)),
            max_value=float(school_view["within_school_gap"].quantile(0.95)),
            value=float(gap_cut),
        )
        climate = st.slider(
            "Hypothetical climate strain",
            min_value=float(school_view["NEGSCLIM"].quantile(0.05)),
            max_value=float(school_view["NEGSCLIM"].quantile(0.95)),
            value=float(climate_cut),
        )
    with col3:
        risk_score = st.slider(
            "Hypothetical risk score",
            min_value=float(school_view["EQUITY_RISK_SCORE"].quantile(0.05)),
            max_value=float(school_view["EQUITY_RISK_SCORE"].quantile(0.95)),
            value=float(context["EQUITY_RISK_SCORE"].median()),
        )
        digital_strain = st.slider(
            "Hypothetical digital-access strain",
            min_value=float(school_view["student_device_problem_mean"].quantile(0.05)),
            max_value=float(school_view["student_device_problem_mean"].quantile(0.95)),
            value=float(digital_cut),
        )

    budget_limit = st.select_slider(
        "Maximum implementation burden to display",
        options=[1, 2, 3],
        value=2,
        format_func=lambda v: COST_LABELS[v],
    )

    inferred_segment = infer_segment_from_profile(
        mean_math=mean_math,
        gap=within_gap,
        climate=climate,
        digital_strain=digital_strain,
        performance_cut=performance_cut,
        gap_cut=gap_cut,
        climate_cut=climate_cut,
        digital_cut=digital_cut,
    )
    risk_class = classify_risk(risk_score, thresholds=risk_thresholds)
    recs = select_recommendations_for_segment(inferred_segment, intervention_map, max_cost_rating=budget_limit)

    render_metric_row(
        [
            ("Reference context", country_choice, "Medians from this context anchor the scenario thresholds."),
            ("Inferred segment", inferred_segment, "Rule-based classification under the published typology."),
            ("Risk class", risk_class, f"Hypothetical risk score {risk_score:.1f}."),
            ("Budget ceiling", COST_LABELS[budget_limit], "Planner filter only; not an affordability guarantee."),
        ]
    )

    comparison = pd.DataFrame(
        [
            {"metric": "Mean maths", "Scenario": mean_math, "Reference median": performance_cut},
            {"metric": "Within-school gap", "Scenario": within_gap, "Reference median": gap_cut},
            {"metric": "Climate strain", "Scenario": climate, "Reference median": climate_cut},
            {"metric": "Digital strain", "Scenario": digital_strain, "Reference median": digital_cut},
            {"metric": "Mean ESCS", "Scenario": mean_escs, "Reference median": float(context["school_mean_escs"].median())},
            {"metric": "Risk score", "Scenario": risk_score, "Reference median": float(context["EQUITY_RISK_SCORE"].median())},
        ]
    ).melt(id_vars="metric", var_name="series", value_name="value")

    left, right = st.columns([1.05, 0.95])
    with left:
        fig = px.bar(
            comparison,
            x="metric",
            y="value",
            color="series",
            barmode="group",
            title="Scenario values vs reference medians",
            labels={"metric": "Metric", "value": "Value"},
            color_discrete_sequence=["#cf5c36", "#8c7a6b"],
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("**Scenario-aligned intervention priorities**")
        if recs.empty:
            st.info("No interventions met the current budget ceiling for this scenario.")
        else:
            for _, rec in recs.iterrows():
                st.markdown(
                    metric_card(
                        rec["recommended_category"].replace("_", " ").title(),
                        rec["intervention"],
                        f"Evidence {int(rec['evidence'])}/5 | {COST_LABELS.get(int(rec['cost_rating']), 'Implementation burden not rated')} | Segment match: {inferred_segment}",
                    ),
                    unsafe_allow_html=True,
                )
        st.markdown(
            """
            <div class="panel-note">
                Use this tab only for hypothetical planning conversations. The output reflects transparent matching rules,
                not a prediction that a school will achieve a given outcome after an intervention.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_evidence_view(data: dict[str, object]) -> None:
    correlates = data["correlates"].copy()
    narrative = data["narrative"]
    scope_limits = data["scope_limits"]
    final_report = data["final_report"]
    v7_shap = data["v7_shap"].copy()
    v7_interventions = data["v7_interventions"].copy()
    v7_summary = data["v7_summary"]

    st.subheader("Evidence and limits")
    st.caption("This tab is the guardrail against overclaiming. It keeps the dashboard anchored to what is directly observed, what is model-based, and what remains uncertain.")

    outcome = st.selectbox(
        "Correlate ranking outcome",
        options=list(OUTCOME_LABELS.keys()),
        format_func=lambda key: OUTCOME_LABELS.get(key, key),
        index=0,
    )
    top = correlates.loc[correlates["outcome"] == outcome].head(12).copy()
    top["direction"] = top["standardized_coef"].apply(lambda v: "Positive" if v >= 0 else "Negative")
    top = top.sort_values("abs_standardized_coef", ascending=True)

    left, right = st.columns([1.05, 0.95])
    with left:
        fig = px.bar(
            top,
            x="abs_standardized_coef",
            y="feature",
            color="direction",
            orientation="h",
            hover_data={
                "standardized_coef": ":.3f",
                "permutation_importance": ":.3f",
                "bivariate_correlation": ":.3f",
                "feature_group": True,
            },
            color_discrete_map={"Positive": "#2C6E49", "Negative": "#B33F40"},
            title=f"Top correlates for {OUTCOME_LABELS[outcome]}",
            labels={"abs_standardized_coef": "Absolute standardized coefficient", "feature": "Feature"},
        )
        fig.update_layout(height=500, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        scatter = px.scatter(
            top,
            x="permutation_importance",
            y="abs_standardized_coef",
            color="feature_group",
            hover_name="feature",
            title="Model-based importance cross-check",
            labels={
                "permutation_importance": "Permutation importance",
                "abs_standardized_coef": "Absolute standardized coefficient",
            },
        )
        scatter.update_layout(height=500, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(scatter, use_container_width=True)

    st.markdown("### Secondary evidence layer: V7 model")
    st.markdown(
        """
        <div class="panel-note">
            V7 is treated as supporting evidence, not as the canonical recommendation engine. It is a 2022-only, cluster-level
            school-context model with grouped cross-validation. Its role is to corroborate which school-context features are worth
            paying attention to, not to produce deterministic school prescriptions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if v7_summary:
        render_metric_row(
            [
                ("V7 grouped CV R²", f"{v7_summary.get('grouped_cv_r2', float('nan')):.3f}", f"SD {v7_summary.get('grouped_cv_sd', float('nan')):.3f}; grouped by country."),
                ("V7 hold-out R²", f"{v7_summary.get('holdout_r2', float('nan')):.3f}", "Reference only; not the primary evaluation."),
                ("Valid clusters", str(v7_summary.get("valid_clusters", "NA")), "Cluster-level slopes retained after pooled-student thresholding."),
                ("Countries in V7", str(v7_summary.get("countries", "NA")), "2022 country coverage in the secondary model."),
            ]
        )

    if not v7_interventions.empty or not v7_shap.empty:
        if "is_policy" in v7_interventions.columns:
            policy_view = v7_interventions.loc[v7_interventions["is_policy"] == True].copy()
        else:
            policy_view = pd.DataFrame()
        if policy_view.empty and "is_policy" in v7_shap.columns:
            policy_view = v7_shap.loc[v7_shap["is_policy"] == True].copy()
        if not policy_view.empty:
            keep_cols = [c for c in ["feature", "direction", "mean_abs_shap", "cost_tier", "action"] if c in policy_view.columns]
            display = policy_view[keep_cols].head(8).copy()
            rename_map = {
                "feature": "Questionnaire-coded feature",
                "direction": "Association direction",
                "mean_abs_shap": "Mean |SHAP|",
                "cost_tier": "Cost tier",
                "action": "Action label",
            }
            st.markdown("**V7 policy-coded signals requiring codebook interpretation**")
            st.dataframe(display.rename(columns=rename_map), use_container_width=True, hide_index=True)
            st.caption("These are auxiliary questionnaire-coded signals. They should stay in the evidence layer, not in the headline user narrative.")

    with st.expander("Competition narrative", expanded=False):
        st.markdown(narrative)
    with st.expander("Scope and limits", expanded=True):
        st.markdown(scope_limits)
    with st.expander("Final synthesis", expanded=False):
        st.markdown(final_report)


def main() -> None:
    apply_page_style()
    try:
        data = load_dashboard_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    render_sidebar(data)
    render_header()

    overview_tab, country_tab, school_tab, observed_tab, scenario_tab, evidence_tab = st.tabs(
        ["Overview", "Country Diagnostic", "School Segments", "Observed Recommendation", "Scenario Planner", "Evidence & Limits"]
    )
    with overview_tab:
        render_overview(data)
    with country_tab:
        render_country_view(data)
    with school_tab:
        render_school_view(data)
    with observed_tab:
        render_observed_recommender_view(data)
    with scenario_tab:
        render_scenario_planner_view(data)
    with evidence_tab:
        render_evidence_view(data)


if __name__ == "__main__":
    main()
