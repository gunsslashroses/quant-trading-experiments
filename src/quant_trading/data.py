"""Data loading, cleaning, and preprocessing utilities.

Handles JKP dataset loading from CSV (or WRDS SQL), date alignment,
winsorization, rank-standardization, and one-hot encoding.
"""

from __future__ import annotations

import gc
from typing import Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

START_DATE = "1970-01-01"
END_DATE = "2026-12-31"

NUMERIC_FEATURES: list[str] = [
    "beta_60m",
    "market_equity",
    "be_me",
    "op_at",
    "at_gr1",
    "qmj",
    "ret_1_0",
    "ret_6_1",
    "oaccruals_at",
    "dbnetis_at",
    "debt_at",
    "niq_at_chg1",
    "ni_be",
]

NUMERIC_FEATURES_ML: list[str] = NUMERIC_FEATURES + ["rvol_252d"]

CATEGORICAL_FEATURES: list[str] = ["ff49"]

TARGET_COL = "ret_exc_lead1m_w"

# Expected direction of each characteristic's return premium
CHAR_DIRECTIONS: dict[str, bool] = {
    "market_equity": False,  # size: small > big
    "be_me": True,  # value: high B/M > low B/M
    "op_at": True,  # profitability: high > low
    "oaccruals_at": False,  # accruals: low > high
    "dbnetis_at": False,  # debt issuance: low > high
    "at_gr1": False,  # investment: low growth > high growth
    "debt_at": False,  # leverage: low > high
    "niq_at_chg1": True,  # profit growth: high > low
    "ni_be": True,  # ROE: high > low
    "qmj": True,  # quality: high > low
    "ret_1_0": False,  # short-term reversal: low > high
    "ret_6_1": True,  # medium-term momentum: high > low
    "ret_12_1": True,  # 12-month momentum: high > low
    "beta_60m": False,  # low beta: low > high
    "rvol_252d": False,  # low volatility
}


# SQL query template for WRDS JKP download
JKP_SQL_TEMPLATE = """
SELECT
    permno, permco, gvkey, iid, id, exch_main, primary_sec, source_crsp, eom,
    size_grp, gics, sic, ff49, ret_exc_lead1m,
    beta_60m, market_equity, be_me, op_at, at_gr1, qmj, rvol_252d,
    ret, ret_1_0, ret_3_1, ret_6_1, ret_12_1, ret_60_12,
    oaccruals_at, dbnetis_at, debt_at, niq_at_chg1, ni_be
FROM contrib.global_factor
WHERE date >= '{start}'
  AND date <= '{end}'
  AND common = 1
  AND excntry = 'USA'
"""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_jkp_csv(
    path: str,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """Load JKP data from a CSV file and apply standard date filters.

    Parameters
    ----------
    path : str
        Path to CSV file previously saved from WRDS.
    start_date, end_date : str
        ISO date strings for the sample period.

    Returns
    -------
    pd.DataFrame
        Cleaned panel with ``month_date`` column at month-start frequency.
    """
    header_cols = pd.read_csv(path, nrows=0).columns
    date_cols = ["eom"] if "eom" in header_cols else []
    df = pd.read_csv(path, parse_dates=date_cols)

    if "month_date" in df.columns:
        df["month_date"] = pd.to_datetime(df["month_date"]).values.astype("datetime64[M]")
    elif "eom" in df.columns:
        df["month_date"] = pd.to_datetime(df["eom"]).values.astype("datetime64[M]")
    else:
        raise ValueError("CSV must contain either 'month_date' or 'eom' column")

    df = df[
        (df["month_date"] >= pd.Timestamp(start_date))
        & (df["month_date"] <= pd.Timestamp(end_date))
    ].copy()

    # Drop duplicate columns (e.g. 'qmj' appearing twice)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    gc.collect()
    return df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def winsorize_returns(
    df: pd.DataFrame,
    lower: float = 0.01,
    upper: float = 0.99,
    src_col: str = "ret_exc_lead1m",
    dst_col: str = "ret_exc_lead1m_w",
) -> pd.DataFrame:
    """Cross-sectional winsorization of returns each month.

    Creates *dst_col* by clipping *src_col* at the monthly [lower, upper]
    quantiles.
    """

    def _clip(x: pd.Series) -> pd.Series:
        lo, hi = x.quantile([lower, upper])
        return x.clip(lo, hi)

    df = df.copy()
    df[dst_col] = df.groupby("month_date")[src_col].transform(_clip)
    return df


def rank_standardize_xsec(ser: pd.Series) -> pd.Series:
    """Cross-sectional rank standardization to [-1, 1].

    Handles missing values by returning NaN for those positions.
    """
    valid = ser.notna()
    out = pd.Series(np.nan, index=ser.index, dtype="float64")
    if valid.sum() == 0:
        return out
    ranked = ser[valid].rank(method="average", pct=True)
    out.loc[valid] = 2.0 * ranked - 1.0
    return out


def prepare_ml_features(
    df: pd.DataFrame,
    numeric_features: Sequence[str] = NUMERIC_FEATURES_ML,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
    target_col: str = TARGET_COL,
    rank_standardize: bool = True,
    fill_missing_with_median: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Prepare features for ML models.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (numeric + one-hot encoded categoricals).
    meta : pd.DataFrame
        Metadata columns (id, month_date, target).
    y : pd.Series
        Target variable.
    """
    keep_cols = (
        ["id", "month_date", target_col] + list(numeric_features) + list(categorical_features)
    )
    keep_cols = list(dict.fromkeys(keep_cols))

    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    base = df[keep_cols].dropna(subset=["id", "month_date", target_col]).copy()

    if fill_missing_with_median:
        base[list(numeric_features)] = base.groupby("month_date")[list(numeric_features)].transform(
            lambda x: x.fillna(x.median())
        )

    if rank_standardize:
        for col in numeric_features:
            base[col] = base.groupby("month_date")[col].transform(rank_standardize_xsec)

    # One-hot encode categoricals
    dummy_frames = []
    for cat in categorical_features:
        base[cat] = base[cat].fillna(-1).astype(int).astype(str)
        dummies = pd.get_dummies(base[cat], prefix=cat, dtype=float)
        dummy_frames.append(dummies)

    X = pd.concat([base[list(numeric_features)].astype(float)] + dummy_frames, axis=1)
    meta = base[["id", "month_date", target_col]].copy()
    y = base[target_col].astype(float)

    return X, meta, y
