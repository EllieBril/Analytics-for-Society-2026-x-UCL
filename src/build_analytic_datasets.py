from __future__ import annotations

from pathlib import Path
from functools import reduce

import numpy as np
import pandas as pd

from .config import (
    CRT_COLS,
    FLT_QQQ_COLS,
    FLT_TIM_COLS,
    PV_CRT_MATH,
    PV_CRT_READ,
    PV_CRT_SCIE,
    PV_FLIT,
    PV_MATH,
    PV_READ,
    PV_SCIE,
    SCHOOL_QQQ_COLS,
    STU_COG_COLS,
    STU_TIM_COLS,
    STUDENT_QQQ_COLS,
    TEACHER_QQQ_COLS,
)
from .load_csv import load_escs_trend, load_multiyear_schools, load_multiyear_students, load_school_risk
from .load_sav import read_sav_columns
from .utils import (
    average_plausible_values,
    normalize_id_series,
    weighted_mean,
    weighted_quantile,
    weighted_share,
    weighted_slope,
    write_text,
)


def _weighted_group_mean(frame: pd.DataFrame, group_col: str, value_col: str, weight_col: str) -> pd.Series:
    return frame.groupby(group_col).apply(lambda g: weighted_mean(g[value_col], g[weight_col]))


def _school_internal_gap(group: pd.DataFrame, outcome: str = "math_score", ses: str = "ESCS", weight: str = "W_FSTUWT") -> pd.Series:
    valid = group[[outcome, ses, weight]].dropna()
    n_students = int(valid.shape[0])
    result = {
        "student_count": n_students,
        "school_mean_math": weighted_mean(valid[outcome], valid[weight]),
        "school_mean_escs": weighted_mean(valid[ses], valid[weight]),
        "within_school_gap": np.nan,
        "within_school_ses_slope": np.nan,
        "low_perf_share_400": weighted_share(valid[outcome] < 400, valid[weight], positive_value=True),
    }
    if n_students < 20:
        return pd.Series(result)
    q25, q75 = weighted_quantile(valid[ses].to_numpy(), valid[weight].to_numpy(), [0.25, 0.75])
    low = valid[valid[ses] <= q25]
    high = valid[valid[ses] >= q75]
    if low.shape[0] >= 5 and high.shape[0] >= 5:
        result["within_school_gap"] = weighted_mean(high[outcome], high[weight]) - weighted_mean(low[outcome], low[weight])
    result["within_school_ses_slope"] = weighted_slope(valid[ses], valid[outcome], valid[weight])
    return pd.Series(result)


def build_multiyear_light(root: Path, output_dir: Path, logger) -> dict[str, pd.DataFrame]:
    students = load_multiyear_students(root)
    schools = load_multiyear_schools(root)
    before = len(students)
    students_dedup = students.drop_duplicates().copy()
    removed = before - len(students_dedup)
    unmatched_exclusions = students_dedup.loc[
        (students_dedup["YEAR"] == 2015) & (students_dedup["CNT"].isin(["QUC", "QUD", "QUE"]))
    ].copy()
    students_for_merge = students_dedup.loc[~students_dedup.index.isin(unmatched_exclusions.index)].copy()
    analytic_multiyear_light = students_for_merge.merge(
        schools.drop(columns=["SCHTYPE"], errors="ignore"),
        on=["YEAR", "CNT", "SCHOOLID_KEY"],
        how="left",
        suffixes=("", "_SCHOOL"),
    )
    analytic_multiyear_light.rename(columns={"SCHOOLID": "SCHOOLID_STUDENT", "SCHOOLID_SCHOOL": "SCHOOLID"}, inplace=True)
    if "SCHOOLID_STUDENT" in analytic_multiyear_light.columns and "SCHOOLID" not in analytic_multiyear_light.columns:
        analytic_multiyear_light.rename(columns={"SCHOOLID_STUDENT": "SCHOOLID"}, inplace=True)
    match_rate = analytic_multiyear_light["SCHSIZE"].notna().mean()
    output_dir.mkdir(parents=True, exist_ok=True)
    analytic_multiyear_light.to_parquet(output_dir / "analytic_multiyear_light.parquet", index=False)
    unmatched_exclusions.to_csv(output_dir / "documented_unmatched_2015_cases.csv", index=False)
    logger.info("Deduplicated multiyear student file: removed %s exact duplicate rows", removed)
    logger.info("Multiyear light merge match rate %.4f", match_rate)
    return {
        "students_dedup": students_dedup,
        "schools": schools,
        "analytic_multiyear_light": analytic_multiyear_light,
        "removed_duplicates": pd.DataFrame([{"removed_duplicate_rows": removed, "merge_match_rate": match_rate}]),
    }


def compute_country_gap_rebuild(students_dedup: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (cnt, year), group in students_dedup.groupby(["CNT", "YEAR"], dropna=False):
        valid = group[["ESCS", "MATH", "READ", "SCIENCE", "STU_WEIGHT"]].dropna(subset=["ESCS", "MATH", "STU_WEIGHT"])
        if valid.empty:
            continue
        q25, q75 = weighted_quantile(valid["ESCS"].to_numpy(), valid["STU_WEIGHT"].to_numpy(), [0.25, 0.75])
        low = valid[valid["ESCS"] <= q25]
        high = valid[valid["ESCS"] >= q75]
        records.append(
            {
                "CNT": cnt,
                "YEAR": int(year),
                "gap_recomputed": weighted_mean(high["MATH"], high["STU_WEIGHT"]) - weighted_mean(low["MATH"], low["STU_WEIGHT"]),
                "avg_math_recomputed": weighted_mean(valid["MATH"], valid["STU_WEIGHT"]),
                "avg_read_recomputed": weighted_mean(valid["READ"], valid["STU_WEIGHT"]),
                "avg_science_recomputed": weighted_mean(valid["SCIENCE"], valid["STU_WEIGHT"]),
                "avg_escs_recomputed": weighted_mean(valid["ESCS"], valid["STU_WEIGHT"]),
                "n_students_recomputed": int(valid.shape[0]),
                "q25_escs": q25,
                "q75_escs": q75,
            }
        )
    return pd.DataFrame(records)


def build_2022_raw_datasets(root: Path, output_dir: Path, logger) -> dict[str, pd.DataFrame]:
    cache_dir = output_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def cached_read(name: str, cols: list[str], cache_name: str, required: bool = True) -> pd.DataFrame:
        cache_path = cache_dir / f"{cache_name}.parquet"
        if cache_path.exists():
            logger.info("Loading cached %s from %s", name, cache_path.name)
            return pd.read_parquet(cache_path)
        if not required:
            logger.info("Skipping uncached auxiliary SAV %s for this run", name)
            return pd.DataFrame()
        logger.info("Reading SAV %s", name)
        df = read_sav_columns(root, name, cols)
        df.to_parquet(cache_path, index=False)
        logger.info("Cached %s rows=%s cols=%s", name, len(df), len(df.columns))
        return df

    stu = cached_read("stu_qqq", STUDENT_QQQ_COLS, "stu_qqq_selected")
    sch = cached_read("sch_qqq", SCHOOL_QQQ_COLS, "sch_qqq_selected")
    tch = cached_read("tch_qqq", TEACHER_QQQ_COLS, "tch_qqq_selected")
    stu_cog = cached_read("stu_cog", STU_COG_COLS, "stu_cog_selected", required=False)
    stu_tim = cached_read("stu_tim", STU_TIM_COLS, "stu_tim_selected", required=False)
    crt = cached_read("crt_cog", CRT_COLS, "crt_selected", required=False)
    flt = cached_read("flt_qqq", FLT_QQQ_COLS, "flt_qqq_selected", required=False)
    flt_tim = cached_read("flt_tim", FLT_TIM_COLS, "flt_tim_selected", required=False)

    for df in [stu, sch, tch, stu_cog, stu_tim, crt, flt, flt_tim]:
        if "CNT" in df.columns:
            df["CNT"] = df["CNT"].astype("string")

    stu = average_plausible_values(stu, [c for c in PV_MATH if c in stu.columns], "math_score")
    stu = average_plausible_values(stu, [c for c in PV_READ if c in stu.columns], "read_score")
    stu = average_plausible_values(stu, [c for c in PV_SCIE if c in stu.columns], "science_score")
    stu["gender_boy"] = np.where(stu["ST004D01T"] == 2, 1, np.where(stu["ST004D01T"] == 1, 0, np.nan))
    stu.rename(columns={"CNTSCHID_KEY": "school_key", "CNTSTUID_KEY": "student_key"}, inplace=True)
    sch.rename(columns={"CNTSCHID_KEY": "school_key"}, inplace=True)
    tch.rename(columns={"CNTSCHID_KEY": "school_key"}, inplace=True)
    if not stu_cog.empty:
        stu_cog.rename(columns={"CNTSCHID_KEY": "school_key", "CNTSTUID_KEY": "student_key"}, inplace=True)
    if not stu_tim.empty:
        stu_tim.rename(columns={"CNTSCHID_KEY": "school_key", "CNTSTUID_KEY": "student_key", "EFFORTT": "stu_effort_time"}, inplace=True)
    if not crt.empty:
        crt.rename(columns={"CNTSCHID_KEY": "school_key", "CNTSTUID_KEY": "student_key"}, inplace=True)
    if not flt.empty:
        flt.rename(columns={"CNTSCHID_KEY": "school_key", "CNTSTUID_KEY": "student_key"}, inplace=True)
    if not flt_tim.empty:
        flt_tim.rename(columns={"CNTSCHID_KEY": "school_key", "CNTSTUID_KEY": "student_key", "EFFORTT": "flt_effort_time"}, inplace=True)

    if not crt.empty:
        crt = average_plausible_values(crt, [c for c in PV_CRT_MATH if c in crt.columns], "crt_math_score")
        crt = average_plausible_values(crt, [c for c in PV_CRT_READ if c in crt.columns], "crt_read_score")
        crt = average_plausible_values(crt, [c for c in PV_CRT_SCIE if c in crt.columns], "crt_science_score")
    if not flt.empty:
        flt = average_plausible_values(flt, [c for c in PV_FLIT if c in flt.columns], "financial_literacy_score")

    teacher_numeric = [c for c in tch.columns if c not in {"CNT", "school_key", "CNTTCHID", "CNTTCHID_KEY"}]
    teacher_agg = tch.groupby(["CNT", "school_key"], dropna=False)[teacher_numeric].mean(numeric_only=True).reset_index()
    teacher_agg["teacher_response_count"] = tch.groupby(["CNT", "school_key"]).size().values

    risk = load_school_risk(root)
    school_csv_2022 = load_multiyear_schools(root)
    school_csv_2022 = school_csv_2022.loc[school_csv_2022["YEAR"] == 2022].copy()
    school_csv_2022["school_key"] = school_csv_2022["SCHOOLID_KEY"]

    analytic_2022_core = stu.merge(
        sch.drop(columns=["CNTSCHID"], errors="ignore"),
        on=["CNT", "school_key"],
        how="left",
        suffixes=("", "_school"),
    )
    analytic_2022_core = analytic_2022_core.merge(
        school_csv_2022[["CNT", "school_key", "SCHSIZE", "STRATIO", "SCHTYPE"]],
        on=["CNT", "school_key"],
        how="left",
        suffixes=("", "_light"),
    )
    analytic_2022_core = analytic_2022_core.merge(
        risk[["CNT", "SCHOOLID_KEY", "EQUITY_RISK_SCORE", "COUNTRY_AVG_GAP", "TRAJECTORY"]],
        left_on=["CNT", "school_key"],
        right_on=["CNT", "SCHOOLID_KEY"],
        how="left",
    ).drop(columns=["SCHOOLID_KEY"])

    analytic_2022_extended = analytic_2022_core.copy()
    if not stu_cog.empty:
        analytic_2022_extended = analytic_2022_extended.merge(
            stu_cog[["CNT", "school_key", "student_key", "RCORE_PERF", "RCO1S_PERF", "MPATH", "RDESIGN", "BOOKID"]],
            on=["CNT", "school_key", "student_key"],
            how="left",
        )
    if not stu_tim.empty:
        analytic_2022_extended = analytic_2022_extended.merge(
            stu_tim[["CNT", "school_key", "student_key", "stu_effort_time"]],
            on=["CNT", "school_key", "student_key"],
            how="left",
        )
    if not crt.empty:
        analytic_2022_extended = analytic_2022_extended.merge(
            crt[["CNT", "school_key", "student_key", "crt_math_score", "crt_read_score", "crt_science_score"]],
            on=["CNT", "school_key", "student_key"],
            how="left",
        )
    if not flt.empty:
        analytic_2022_extended = analytic_2022_extended.merge(
            flt[["CNT", "school_key", "student_key", "financial_literacy_score", "FLSCHOOL", "FLMULTSB", "ACCESSFP", "FLCONFIN", "FLCONICT", "ATTCONFM"]],
            on=["CNT", "school_key", "student_key"],
            how="left",
        )
    if not flt_tim.empty:
        analytic_2022_extended = analytic_2022_extended.merge(
            flt_tim[["CNT", "school_key", "student_key", "flt_effort_time"]],
            on=["CNT", "school_key", "student_key"],
            how="left",
        )
    analytic_2022_extended = analytic_2022_extended.merge(teacher_agg, on=["CNT", "school_key"], how="left")
    logger.info("Merged 2022 extended dataset rows=%s cols=%s", len(analytic_2022_extended), len(analytic_2022_extended.columns))

    logger.info("Building school profiles")
    school_profiles = build_school_profiles(analytic_2022_extended)
    logger.info("Built school profiles rows=%s", len(school_profiles))
    analytic_2022_extended.to_parquet(output_dir / "analytic_dataset.parquet", index=False)
    analytic_2022_core.to_parquet(output_dir / "analytic_2022_core.parquet", index=False)
    school_profiles.to_parquet(output_dir / "school_profiles_2022.parquet", index=False)

    merge_diag = pd.DataFrame(
        [
            {
                "dataset": "student_questionnaire_to_school_questionnaire",
                "base_rows": len(stu),
                "matched_rows": int(analytic_2022_core["SC013Q01TA"].notna().sum()),
                "match_rate": analytic_2022_core["SC013Q01TA"].notna().mean(),
            },
            {
                "dataset": "teacher_school_aggregation_to_core",
                "base_rows": len(school_profiles),
                "matched_rows": int(school_profiles["teacher_response_count"].notna().sum()),
                "match_rate": school_profiles["teacher_response_count"].notna().mean(),
            },
            {
                "dataset": "student_cognitive_to_core",
                "base_rows": len(analytic_2022_core),
                "matched_rows": int(analytic_2022_extended["RCORE_PERF"].notna().sum()) if "RCORE_PERF" in analytic_2022_extended.columns else 0,
                "match_rate": analytic_2022_extended["RCORE_PERF"].notna().mean() if "RCORE_PERF" in analytic_2022_extended.columns else 0.0,
            },
            {
                "dataset": "critical_thinking_to_core",
                "base_rows": len(analytic_2022_core),
                "matched_rows": int(analytic_2022_extended["crt_math_score"].notna().sum()) if "crt_math_score" in analytic_2022_extended.columns else 0,
                "match_rate": analytic_2022_extended["crt_math_score"].notna().mean() if "crt_math_score" in analytic_2022_extended.columns else 0.0,
            },
            {
                "dataset": "financial_literacy_to_core",
                "base_rows": len(analytic_2022_core),
                "matched_rows": int(analytic_2022_extended["financial_literacy_score"].notna().sum()) if "financial_literacy_score" in analytic_2022_extended.columns else 0,
                "match_rate": analytic_2022_extended["financial_literacy_score"].notna().mean() if "financial_literacy_score" in analytic_2022_extended.columns else 0.0,
            },
        ]
    )
    merge_diag.to_csv(output_dir / "merge_diagnostics.csv", index=False)
    write_text(output_dir / "analytic_dataset_codebook.md", build_codebook(analytic_2022_extended, sch, tch))
    logger.info("Step 5 dataset build complete: core rows=%s extended rows=%s schools=%s", len(analytic_2022_core), len(analytic_2022_extended), len(school_profiles))
    return {
        "analytic_2022_core": analytic_2022_core,
        "analytic_2022_extended": analytic_2022_extended,
        "school_profiles": school_profiles,
        "merge_diagnostics": merge_diag,
        "teacher_agg": teacher_agg,
    }


def build_school_profiles(analytic_2022_extended: pd.DataFrame) -> pd.DataFrame:
    school_student = analytic_2022_extended.copy()
    school_student["W_FSTUWT"] = pd.to_numeric(school_student["W_FSTUWT"], errors="coerce")
    student_weight = "W_FSTUWT"
    gap_input = school_student[["CNT", "school_key", "math_score", "ESCS", student_weight]].copy()
    gap_input = gap_input.dropna(subset=["math_score", "ESCS", student_weight]).copy()
    key_cols = ["CNT", "school_key"]
    gap_input["w_math"] = gap_input["math_score"] * gap_input[student_weight]
    gap_input["w_escs"] = gap_input["ESCS"] * gap_input[student_weight]
    gap_input["w_xy"] = gap_input["math_score"] * gap_input["ESCS"] * gap_input[student_weight]
    gap_input["w_x2"] = (gap_input["ESCS"] ** 2) * gap_input[student_weight]
    gap_input["w_low_perf"] = (gap_input["math_score"] < 400).astype(float) * gap_input[student_weight]
    grouped = gap_input.groupby(key_cols, dropna=False)
    school_gap = grouped.agg(
        student_count=("math_score", "size"),
        w_sum=(student_weight, "sum"),
        w_math_sum=("w_math", "sum"),
        w_escs_sum=("w_escs", "sum"),
        w_xy_sum=("w_xy", "sum"),
        w_x2_sum=("w_x2", "sum"),
        w_low_perf_sum=("w_low_perf", "sum"),
    ).reset_index()
    school_gap["school_mean_math"] = school_gap["w_math_sum"] / school_gap["w_sum"]
    school_gap["school_mean_escs"] = school_gap["w_escs_sum"] / school_gap["w_sum"]
    school_gap["low_perf_share_400"] = school_gap["w_low_perf_sum"] / school_gap["w_sum"]
    school_gap["within_school_ses_slope"] = (
        (school_gap["w_xy_sum"] / school_gap["w_sum"]) - (school_gap["school_mean_escs"] * school_gap["school_mean_math"])
    ) / ((school_gap["w_x2_sum"] / school_gap["w_sum"]) - (school_gap["school_mean_escs"] ** 2))
    school_gap["within_school_ses_slope"] = school_gap["within_school_ses_slope"].replace([np.inf, -np.inf], np.nan)

    gap_input["escs_rank_pct"] = gap_input.groupby(key_cols)["ESCS"].rank(pct=True, method="first")
    low = gap_input.loc[gap_input["escs_rank_pct"] <= 0.25].copy()
    high = gap_input.loc[gap_input["escs_rank_pct"] > 0.75].copy()
    low_gap = (low.groupby(key_cols)["w_math"].sum() / low.groupby(key_cols)[student_weight].sum()).rename("low_escs_math")
    high_gap = (high.groupby(key_cols)["w_math"].sum() / high.groupby(key_cols)[student_weight].sum()).rename("high_escs_math")
    school_gap = school_gap.merge(low_gap.reset_index(), on=key_cols, how="left").merge(high_gap.reset_index(), on=key_cols, how="left")
    school_gap["within_school_gap"] = school_gap["high_escs_math"] - school_gap["low_escs_math"]
    school_gap.loc[school_gap["student_count"] < 20, ["within_school_gap", "within_school_ses_slope"]] = np.nan
    school_gap = school_gap.drop(columns=["w_sum", "w_math_sum", "w_escs_sum", "w_xy_sum", "w_x2_sum", "w_low_perf_sum", "low_escs_math", "high_escs_math"])

    agg_specs = {
        "BELONG": "student_belong_mean",
        "ANXMAT": "student_math_anxiety_mean",
        "SKIPPING": "student_skipping_mean",
        "TARDYSD": "student_tardy_mean",
        "INFOSEEK": "student_career_info_mean",
        "EXPECEDU": "student_expected_edu_mean",
        "STUDYHMW": "student_homework_mean",
        "MATHMOT": "student_math_motivation_mean",
        "MATHEFF": "student_math_selfefficacy_mean",
        "MATHEF21": "student_math_reasoning_selfefficacy_mean",
        "ICTEFFIC": "student_digital_efficacy_mean",
        "ICTSUBJ": "student_subject_ict_use_mean",
        "ICTREG": "student_ict_regulation_mean",
        "CURSUPP": "student_current_parent_support_mean",
        "PARINVOL": "student_parent_involvement_mean",
        "PAREXPT": "student_parent_career_expectations_mean",
        "ST352Q01JA": "student_device_problem_mean",
        "ST352Q02JA": "student_internet_problem_mean",
        "ST300Q07JA": "student_parent_encouragement_mean",
        "ST300Q09JA": "student_parent_future_talk_mean",
        "ST330Q04WA": "student_career_advisor_school_mean",
        "ST330Q07WA": "student_career_internet_search_mean",
        "IC172Q01JA": "student_school_digital_resources_mean",
        "IC172Q02JA": "student_school_digital_internet_devices_mean",
        "IC172Q03JA": "student_school_internet_speed_mean",
        "IC172Q07JA": "student_school_tech_support_mean",
        "IC172Q08JA": "student_teachers_digital_skills_mean",
        "IC172Q09JA": "student_teachers_willing_digital_mean",
        "IC173Q02JA": "student_math_ict_use_lessons_mean",
        "IC175Q01JA": "student_digital_feedback_mean",
        "stu_effort_time": "student_questionnaire_effort_time_mean",
        "financial_literacy_score": "financial_literacy_mean",
    }
    agg_source_cols = [c for c in agg_specs if c in school_student.columns]
    agg_df = school_student[["CNT", "school_key", student_weight] + agg_source_cols].copy()
    feature_tables = []
    for source in agg_source_cols:
        valid = agg_df.loc[agg_df[source].notna(), key_cols + [source, student_weight]].copy()
        if valid.empty:
            continue
        valid["_wx"] = valid[source] * valid[student_weight]
        num = valid.groupby(key_cols)["_wx"].sum().rename("_wx_sum")
        den = valid.groupby(key_cols)[student_weight].sum().rename("_w_sum")
        mean = (num / den).rename(agg_specs[source]).reset_index()
        feature_tables.append(mean)
    if feature_tables:
        school_student_agg = reduce(lambda left, right: left.merge(right, on=key_cols, how="outer"), feature_tables)
    else:
        school_student_agg = agg_df[key_cols].drop_duplicates().copy()

    school_context_cols = [
        "CNT",
        "school_key",
        "SC001Q01TA",
        "SC011Q01TA",
        "SC013Q01TA",
        "SCHLTYPE",
        "STAFFSHORT",
        "EDUSHORT",
        "NEGSCLIM",
        "STUBEHA",
        "TEACHBEHA",
        "OPENCUL",
        "SCHAUTO",
        "ABGMATH",
        "RATCMP1",
        "RATCMP2",
        "PROPSUPP",
        "SCSUPRTED",
        "SCSUPRT",
        "SC061Q05TA",
        "SC017Q09JA",
        "SC017Q10JA",
        "SC201Q01JA",
        "SC201Q03JA",
        "SC201Q04JA",
        "SC201Q05JA",
        "SC201Q06JA",
        "SC173Q06JA",
        "SC037Q08TA",
        "SC042Q01TA",
        "SC042Q02TA",
        "SC187Q03WA",
        "SC187Q04WA",
        "SCHSIZE",
        "STRATIO",
        "EQUITY_RISK_SCORE",
        "COUNTRY_AVG_GAP",
        "TRAJECTORY",
        "teacher_response_count",
        "TRUST",
        "TCDISCLIMA",
        "AUTONOMY",
        "FEEDBINSTR",
        "ICTMATTC",
        "TC241Q01JA",
        "TC253Q01JA",
        "TC253Q03JA",
        "TC253Q04JA",
        "TC255Q06JA",
        "TC216Q05JA",
        "TC232Q04JA",
        "TC232Q06JA",
        "TC220Q07JA",
    ]
    school_context = school_student[school_context_cols].drop_duplicates(subset=["CNT", "school_key"])
    school_profiles = school_gap.merge(school_student_agg, on=["CNT", "school_key"], how="left").merge(
        school_context, on=["CNT", "school_key"], how="left"
    )
    school_profiles["teacher_data_available"] = school_profiles["teacher_response_count"].notna().astype(int)
    school_profiles["country_school_count"] = school_profiles.groupby("CNT")["school_key"].transform("nunique")
    school_profiles["risk_quintile"] = pd.qcut(
        school_profiles["EQUITY_RISK_SCORE"].rank(method="first"), q=5, labels=False, duplicates="drop"
    ) + 1
    return school_profiles


def build_codebook(analytic_2022_extended: pd.DataFrame, sch_raw: pd.DataFrame, tch_raw: pd.DataFrame) -> str:
    columns = sorted(analytic_2022_extended.columns.tolist())
    important = [
        "CNT", "school_key", "student_key", "W_FSTUWT", "math_score", "read_score", "science_score", "ESCS",
        "within_school_gap", "EQUITY_RISK_SCORE", "STAFFSHORT", "EDUSHORT", "NEGSCLIM", "SCHAUTO", "teacher_response_count",
    ]
    lines = [
        "# Analytic Dataset Codebook",
        "",
        "This file documents the canonical 2022 merged dataset used for the later steps.",
        "",
        "## Key identifiers",
        "- `CNT`: country code.",
        "- `school_key`: normalised international school identifier from `CNTSCHID`.",
        "- `student_key`: normalised international student identifier from `CNTSTUID`.",
        "",
        "## Main derived outcomes",
        "- `math_score`, `read_score`, `science_score`: mean of the 10 plausible values for descriptive modelling.",
        "- `EQUITY_RISK_SCORE`: supplied 2022 school risk score from the local CSV.",
        "",
        "## Main feature families",
        "- Student SES and home background: `ESCS`, `HISEI`, `PAREDINT`, `HOMEPOS`.",
        "- Student attitudes and behaviours: `BELONG`, `ANXMAT`, `SKIPPING`, `TARDYSD`, `MATHMOT`, `MATHEFF`, `ICTEFFIC`.",
        "- School context: `STAFFSHORT`, `EDUSHORT`, `NEGSCLIM`, `STUBEHA`, `TEACHBEHA`, `OPENCUL`, `SCHAUTO`, `ABGMATH`.",
        "- Teacher aggregates: `TRUST`, `TCDISCLIMA`, `AUTONOMY`, `FEEDBINSTR`, `ICTMATTC` and related raw averages.",
        "",
        f"## Column count\n- {len(columns)} columns in canonical dataset.",
        "",
        "## Selected columns",
    ]
    for col in columns[:120]:
        marker = " (priority)" if col in important else ""
        lines.append(f"- `{col}`{marker}")
    if len(columns) > 120:
        lines.append(f"- Additional columns omitted here for brevity: {len(columns) - 120}")
    return "\n".join(lines)


def write_equity_definition(
    output_dir: Path,
    rebuilt_gap: pd.DataFrame,
    supplied_gap: pd.DataFrame,
    escs_trend: pd.DataFrame,
) -> None:
    compare = supplied_gap.merge(rebuilt_gap, on=["CNT", "YEAR"], how="left")
    compare["abs_gap_diff"] = (compare["GAP"] - compare["gap_recomputed"]).abs()
    mean_abs_diff = compare["abs_gap_diff"].mean()
    max_abs_diff = compare["abs_gap_diff"].max()
    trend_years = sorted(escs_trend["cycle"].dropna().unique().tolist())
    lines = [
        "# Equity Gap Definition",
        "",
        "## Primary definition",
        "- Country-level primary metric: weighted mathematics score gap between the top and bottom ESCS quartiles within each country-year.",
        "- School-level primary metric: weighted mathematics score gap between the top and bottom ESCS quartiles within each school.",
        "- Secondary school-level estimate: weighted within-school SES slope in mathematics.",
        "",
        "## Why this definition",
        "- It is directly anchored in the available student SES measure (`ESCS`) and mathematics plausible values.",
        "- It is transparent enough for a competition narrative and easier to explain than a fully model-based index.",
        "",
        "## Supplied vs rebuilt metric",
        f"- Mean absolute difference between supplied `GAP` and rebuilt quartile gap: {mean_abs_diff:.2f} points.",
        f"- Maximum absolute difference: {max_abs_diff:.2f} points.",
        "- Interpretation: the supplied gap behaves like a SES-based mathematics inequality measure, but the rebuilt definition is retained for transparency.",
        "",
        "## Secondary operationalisations considered",
        "- Regression-based SES slope: useful for within-school inequality and less sensitive to quartile cut points.",
        "- Between-school vs within-school decomposition: useful conceptually, but not taken as the headline metric in v1 because the competition deliverable needs a simpler explanation.",
        "",
        "## ESCS comparability note",
        f"- The local `escs_trend.csv` provides a cross-cycle ESCS re-scaling for cycles {trend_years}, which supports cautious comparability for 2012-2018.",
        "- For the main multi-year country gap story, the supplied and rebuilt CSV-based series are used; for the rich driver work, the focus narrows to 2022.",
    ]
    write_text(output_dir / "equity_gap_definition.md", "\n".join(lines))
