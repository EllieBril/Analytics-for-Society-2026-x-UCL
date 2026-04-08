from __future__ import annotations

from pathlib import Path


CSV_FILES = {
    "country_trajectories": "country_trajectories.csv",
    "equity_gap_by_country_year": "equity_gap_by_country_year.csv",
    "intervention_library": "intervention_library.csv",
    "pisa_school_all_years": "pisa_school_all_years.csv",
    "pisa_student_all_years": "pisa_student_all_years.csv",
    "school_risk_scores": "school_risk_scores.csv",
    "escs_trend": "escs_trend/escs_trend.csv",
}

SAV_FILES = {
    "stu_qqq": "CY08MSP_STU_QQQ.SAV",
    "stu_cog": "CY08MSP_STU_COG.SAV",
    "stu_tim": "CY08MSP_STU_TIM.SAV",
    "sch_qqq": "CY08MSP_SCH_QQQ.SAV",
    "tch_qqq": "CY08MSP_TCH_QQQ.SAV",
    "crt_cog": "CY08MSP_CRT_COG.SAV",
    "flt_qqq": "FLT_SPSS/CY08MSP_FLT_QQQ.SAV",
    "flt_cog": "FLT_SPSS/CY08MSP_FLT_COG.SAV",
    "flt_tim": "FLT_SPSS/CY08MSP_FLT_TIM.SAV",
}

PV_MATH = [f"PV{i}MATH" for i in range(1, 11)]
PV_READ = [f"PV{i}READ" for i in range(1, 11)]
PV_SCIE = [f"PV{i}SCIE" for i in range(1, 11)]
PV_CRT_MATH = [f"PV{i}MATC" for i in range(1, 11)]
PV_CRT_READ = [f"PV{i}REAC" for i in range(1, 11)]
PV_CRT_SCIE = [f"PV{i}SCIC" for i in range(1, 11)]
PV_FLIT = [f"PV{i}FLIT" for i in range(1, 11)]

STUDENT_QQQ_COLS = [
    "CNT",
    "CNTSCHID",
    "CNTSTUID",
    "CNTRYID",
    "STRATUM",
    "SUBNATIO",
    "REGION",
    "OECD",
    "ADMINMODE",
    "W_FSTUWT",
    "ST004D01T",
    "ESCS",
    "BELONG",
    "ANXMAT",
    "HISEI",
    "PAREDINT",
    "HOMEPOS",
    "SKIPPING",
    "TARDYSD",
    "INFOSEEK",
    "EXPECEDU",
    "STUDYHMW",
    "MATHMOT",
    "MATHEFF",
    "MATHEF21",
    "ICTEFFIC",
    "ICTSUBJ",
    "ICTREG",
    "CURSUPP",
    "PARINVOL",
    "PAREXPT",
    "ST062Q01TA",
    "ST062Q02TA",
    "ST062Q03TA",
    "ST300Q07JA",
    "ST300Q09JA",
    "ST330Q04WA",
    "ST330Q07WA",
    "ST352Q01JA",
    "ST352Q02JA",
    "ST324Q04JA",
    "ST324Q14JA",
    "IC172Q01JA",
    "IC172Q02JA",
    "IC172Q03JA",
    "IC172Q07JA",
    "IC172Q08JA",
    "IC172Q09JA",
    "IC173Q02JA",
    "IC175Q01JA",
] + PV_MATH + PV_READ + PV_SCIE

SCHOOL_QQQ_COLS = [
    "CNT",
    "CNTSCHID",
    "CNTRYID",
    "STRATUM",
    "SUBNATIO",
    "REGION",
    "OECD",
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
    "SC202Q01JA",
    "SC202Q02JA",
    "SC202Q03JA",
    "SC202Q04JA",
    "SC202Q06JA",
    "SC042Q01TA",
    "SC042Q02TA",
    "SC187Q03WA",
    "SC187Q04WA",
    "SC004Q02TA",
    "SC004Q03TA",
    "SC004Q07NA",
    "SC018Q01TA01",
    "SC018Q01TA02",
    "SC018Q02TA01",
    "SC018Q02TA02",
    "SC182Q01WA01",
    "SC182Q01WA02",
    "SC182Q06WA01",
    "SC182Q06WA02",
]

TEACHER_QQQ_COLS = [
    "CNT",
    "CNTSCHID",
    "CNTTCHID",
    "TCHTYPE",
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

STU_COG_COLS = [
    "CNT",
    "CNTSCHID",
    "CNTSTUID",
    "BOOKID",
    "RDESIGN",
    "RCORE_PERF",
    "RCO1S_PERF",
    "MPATH",
    "MCORE_TEST",
    "RCORE_TEST",
]

STU_TIM_COLS = [
    "CNT",
    "CNTSCHID",
    "CNTSTUID",
    "EFFORTT",
]

CRT_COLS = ["CNT", "CNTSCHID", "CNTSTUID"] + PV_CRT_MATH + PV_CRT_READ + PV_CRT_SCIE

FLT_QQQ_COLS = [
    "CNT",
    "CNTSCHID",
    "CNTSTUID",
    "W_FSTUWT",
    "FLSCHOOL",
    "FLMULTSB",
    "ACCESSFP",
    "FLCONFIN",
    "FLCONICT",
    "ATTCONFM",
] + PV_FLIT

FLT_TIM_COLS = [
    "CNT",
    "CNTSCHID",
    "CNTSTUID",
    "EFFORTT",
]

ID_COLS = ["CNT", "CNTSCHID", "CNTSTUID", "CNTTCHID", "YEAR", "SCHOOLID", "STUDENTID", "CNTRYID", "CYC"]

OUTPUT_STEPS = {
    1: "01_data_inventory",
    2: "02_equity_definition",
    3: "03_country_trajectories",
    4: "04_school_risk",
    5: "05_analytic_dataset",
    6: "06_correlates",
    7: "07_within_school_gap",
    8: "08_segmentation",
    9: "09_intervention_mapping",
    10: "10_prototype",
    11: "11_competition_fit",
    12: "12_final",
}


def output_dir(root: Path, step: int) -> Path:
    return root / "outputs" / OUTPUT_STEPS[step]
