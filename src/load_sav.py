from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import pyreadstat

from .config import SAV_FILES
from .utils import normalize_id_series


def sav_path(root: Path, name: str) -> Path:
    return root / SAV_FILES[name]


def read_sav_metadata(root: Path, name: str):
    return pyreadstat.read_sav(sav_path(root, name), metadataonly=True)[1]


def read_sav_columns(root: Path, name: str, columns: Iterable[str]) -> pd.DataFrame:
    path = sav_path(root, name)
    meta = read_sav_metadata(root, name)
    requested = [c for c in dict.fromkeys(columns) if c in meta.column_names]
    df, _ = pyreadstat.read_sav(path, usecols=requested)
    for col in ["CNTSCHID", "CNTSTUID", "CNTTCHID"]:
        if col in df.columns:
            df[f"{col}_KEY"] = normalize_id_series(df[col])
    return df


def read_sav_key_sample(root: Path, name: str) -> pd.DataFrame:
    meta = read_sav_metadata(root, name)
    cols = [c for c in ["CNT", "CNTSCHID", "CNTSTUID", "CNTTCHID", "CYC", "W_FSTUWT"] if c in meta.column_names]
    return read_sav_columns(root, name, cols)
