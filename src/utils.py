from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sdg4_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def normalize_id(value):
    if pd.isna(value):
        return pd.NA
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.endswith(".0"):
            cleaned = cleaned[:-2]
        return cleaned
    try:
        as_float = float(value)
        if as_float.is_integer():
            return str(int(as_float))
        return str(as_float)
    except Exception:
        return str(value)


def normalize_id_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_id).astype("string")


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    if mask.sum() == 0:
        return np.nan
    v = values[mask].astype(float).to_numpy()
    w = weights[mask].astype(float).to_numpy()
    total = w.sum()
    if total == 0:
        return np.nan
    return float(np.average(v, weights=w))


def weighted_share(values: pd.Series, weights: pd.Series, positive_value=1) -> float:
    mask = values.notna() & weights.notna()
    if mask.sum() == 0:
        return np.nan
    indicator = (values[mask] == positive_value).astype(float)
    return weighted_mean(pd.Series(indicator), weights[mask])


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: Iterable[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]
    if values.size == 0:
        return np.array([np.nan] * len(list(quantiles)))
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    quantiles = np.asarray(list(quantiles), dtype=float)
    return np.interp(quantiles, cumulative, values)


def average_plausible_values(df: pd.DataFrame, cols: list[str], out_col: str) -> pd.DataFrame:
    available = [c for c in cols if c in df.columns]
    if not available:
        df[out_col] = np.nan
        return df
    df[out_col] = df[available].mean(axis=1)
    return df


def weighted_slope(x: pd.Series, y: pd.Series, w: pd.Series) -> float:
    mask = x.notna() & y.notna() & w.notna()
    if mask.sum() < 3:
        return np.nan
    x_arr = x[mask].astype(float).to_numpy()
    y_arr = y[mask].astype(float).to_numpy()
    w_arr = w[mask].astype(float).to_numpy()
    if np.nanstd(x_arr) == 0 or w_arr.sum() == 0:
        return np.nan
    x_bar = np.average(x_arr, weights=w_arr)
    y_bar = np.average(y_arr, weights=w_arr)
    cov = np.average((x_arr - x_bar) * (y_arr - y_bar), weights=w_arr)
    var = np.average((x_arr - x_bar) ** 2, weights=w_arr)
    if var == 0:
        return np.nan
    return float(cov / var)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.1%}"


def format_num(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def rank_bands(series: pd.Series, labels: list[str]) -> pd.Series:
    valid = series.dropna()
    if valid.nunique() < len(labels):
        bins = pd.qcut(valid.rank(method="first"), q=len(labels), labels=labels)
    else:
        bins = pd.qcut(valid, q=len(labels), labels=labels, duplicates="drop")
    out = pd.Series(pd.NA, index=series.index, dtype="string")
    out.loc[bins.index] = bins.astype("string")
    return out
