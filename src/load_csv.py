from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CSV_FILES
from .utils import normalize_id_series


def read_csv(root: Path, name: str, **kwargs) -> pd.DataFrame:
    path = root / CSV_FILES[name]
    return pd.read_csv(path, **kwargs)


def load_multiyear_students(root: Path) -> pd.DataFrame:
    df = read_csv(root, "pisa_student_all_years")
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
    df["SCHOOLID_KEY"] = normalize_id_series(df["SCHOOLID"])
    df["STUDENTID_KEY"] = normalize_id_series(df["STUDENTID"])
    return df


def load_multiyear_schools(root: Path) -> pd.DataFrame:
    df = read_csv(root, "pisa_school_all_years")
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
    df["SCHOOLID_KEY"] = normalize_id_series(df["SCHOOLID"])
    return df


def load_country_trajectories(root: Path) -> pd.DataFrame:
    return read_csv(root, "country_trajectories")


def load_equity_gap(root: Path) -> pd.DataFrame:
    return read_csv(root, "equity_gap_by_country_year")


def load_school_risk(root: Path) -> pd.DataFrame:
    df = read_csv(root, "school_risk_scores")
    df["SCHOOLID_KEY"] = normalize_id_series(df["SCHOOLID"])
    return df


def load_interventions(root: Path) -> pd.DataFrame:
    return read_csv(root, "intervention_library")


def load_escs_trend(root: Path) -> pd.DataFrame:
    df = read_csv(root, "escs_trend")
    df["schoolid_key"] = normalize_id_series(df["schoolid"])
    df["studentid_key"] = normalize_id_series(df["studentid"])
    return df
