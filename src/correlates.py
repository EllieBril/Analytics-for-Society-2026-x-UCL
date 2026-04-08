from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .utils import format_num, write_text


FEATURE_GROUPS = {
    "student_belong_mean": "student_attitudes",
    "student_math_anxiety_mean": "student_attitudes",
    "student_skipping_mean": "attendance",
    "student_tardy_mean": "attendance",
    "student_career_info_mean": "aspirations",
    "student_expected_edu_mean": "aspirations",
    "student_homework_mean": "learning_time",
    "student_math_motivation_mean": "student_attitudes",
    "student_math_selfefficacy_mean": "student_attitudes",
    "student_math_reasoning_selfefficacy_mean": "student_attitudes",
    "student_digital_efficacy_mean": "digital",
    "student_subject_ict_use_mean": "digital",
    "student_ict_regulation_mean": "digital",
    "student_current_parent_support_mean": "family",
    "student_parent_involvement_mean": "family",
    "student_parent_career_expectations_mean": "family",
    "student_device_problem_mean": "digital_access",
    "student_internet_problem_mean": "digital_access",
    "student_school_digital_resources_mean": "digital_access",
    "student_school_digital_internet_devices_mean": "digital_access",
    "student_school_internet_speed_mean": "digital_access",
    "STAFFSHORT": "school_resources",
    "EDUSHORT": "school_resources",
    "NEGSCLIM": "school_climate",
    "STUBEHA": "school_climate",
    "TEACHBEHA": "school_climate",
    "OPENCUL": "school_climate",
    "SCHAUTO": "school_governance",
    "ABGMATH": "grouping_tracking",
    "RATCMP1": "digital_access",
    "RATCMP2": "digital_access",
    "PROPSUPP": "support_staff",
    "SC061Q05TA": "school_climate",
    "SC017Q09JA": "digital_access",
    "SC017Q10JA": "digital_access",
    "SC201Q01JA": "leadership",
    "SC201Q03JA": "leadership",
    "SC201Q04JA": "leadership",
    "SC201Q05JA": "leadership",
    "SC201Q06JA": "leadership",
    "teacher_response_count": "teacher_context",
    "TRUST": "teacher_context",
    "TCDISCLIMA": "teacher_context",
    "AUTONOMY": "teacher_context",
    "FEEDBINSTR": "teacher_context",
    "ICTMATTC": "teacher_context",
    "school_mean_escs": "composition",
}


def analyze_correlates(school_profiles: pd.DataFrame, output_dir: Path, logger) -> dict[str, pd.DataFrame]:
    profiles = school_profiles.copy()
    feature_cols = [c for c in FEATURE_GROUPS if c in profiles.columns and profiles[c].notna().mean() >= 0.5]
    target_cols = {
        "EQUITY_RISK_SCORE": "school_risk_score",
        "within_school_gap": "within_school_gap",
        "school_mean_math": "school_mean_math",
    }
    school_weight = profiles["student_count"].fillna(1)
    results = []
    bivariate = []
    for target_col, target_name in target_cols.items():
        model_df = profiles[[target_col] + feature_cols].copy()
        model_df = model_df.loc[model_df[target_col].notna()].copy()
        y = model_df[target_col]
        X = model_df[feature_cols]
        numeric_features = feature_cols
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_features,
                )
            ]
        )
        elastic = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, random_state=42, max_iter=5000)),
            ]
        )
        elastic.fit(X, y, model__sample_weight=school_weight.loc[model_df.index])
        coef = elastic.named_steps["model"].coef_

        rf = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("rf", RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42, n_jobs=1)),
            ]
        )
        rf.fit(X, y)
        perm = permutation_importance(rf, X, y, n_repeats=5, random_state=42, n_jobs=1)

        for i, feature in enumerate(feature_cols):
            corr = model_df[[feature, target_col]].corr(numeric_only=True).iloc[0, 1]
            bivariate.append({"outcome": target_name, "feature": feature, "correlation": corr})
            results.append(
                {
                    "outcome": target_name,
                    "feature": feature,
                    "feature_group": FEATURE_GROUPS.get(feature, "other"),
                    "standardized_coef": coef[i],
                    "abs_standardized_coef": abs(coef[i]),
                    "permutation_importance": perm.importances_mean[i],
                    "bivariate_correlation": corr,
                }
            )

    ranking = pd.DataFrame(results).sort_values(["outcome", "abs_standardized_coef"], ascending=[True, False]).reset_index(drop=True)
    bivariate_df = pd.DataFrame(bivariate)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output_dir / "candidate_driver_rankings.csv", index=False)
    bivariate_df.to_csv(output_dir / "bivariate_correlates.csv", index=False)

    report_lines = [
        "# Correlates Report",
        "",
        "## Completion",
        "- Status: fully completed.",
        "",
        "## Method",
        "- Unit of analysis: school-level 2022 profiles built from student, school, and teacher data.",
        "- Methods used: weighted descriptive aggregation, bivariate correlations, Elastic Net for standardized coefficients, and shallow random-forest permutation importance as a secondary check.",
        "- Interpretation rule: all findings are correlational and observational.",
        "",
        "## Strongest correlates by outcome",
    ]
    for outcome in ranking["outcome"].unique():
        top = ranking.loc[ranking["outcome"] == outcome].head(8)
        report_lines.append(f"### {outcome}")
        for _, row in top.iterrows():
            report_lines.append(
                f"- `{row['feature']}` ({row['feature_group']}) coef={format_num(row['standardized_coef'], 3)}, perm_imp={format_num(row['permutation_importance'], 3)}, corr={format_num(row['bivariate_correlation'], 3)}"
            )
    report_lines += [
        "",
        "## Caution",
        "- These rankings indicate which observed features co-move with risk, within-school inequality, or average performance after partial regularisation.",
        "- They do not establish why the gaps exist and they are not suitable for claiming causal intervention effects.",
    ]
    write_text(output_dir / "correlates_report.md", "\n".join(report_lines))
    logger.info("Step 6 complete: correlates rows=%s", len(ranking))
    return {"ranking": ranking, "bivariate": bivariate_df}
