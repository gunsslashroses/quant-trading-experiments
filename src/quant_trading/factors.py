"""Fama-French factor utilities.

Provides helpers for downloading official FF factors and comparing
reconstructed factor series against them.

Note: ``build_ff_factors`` (2×3 sort) has been removed — it was incorrectly
implemented and will be rewritten later.
"""

from __future__ import annotations

import pandas as pd


def fetch_ff5_factors(
    start_date: str = "1970-01-01",
    end_date: str = "2026-12-31",
) -> pd.DataFrame:
    """Download the official Fama-French 5-factor monthly data.

    Returns a DataFrame indexed by monthly Timestamps with columns:
    Mkt-RF, SMB, HML, RMW, CMA, RF.
    """
    import pandas_datareader.data as web

    ff5_data = web.DataReader(
        "F-F_Research_Data_5_Factors_2x3",
        "famafrench",
        start_date,
        end_date,
    )
    ff5 = ff5_data[0] / 100  # convert from percent to decimal
    ff5.index = ff5.index.to_timestamp()
    return ff5


def compare_factors(
    reconstructed: pd.DataFrame,
    official: pd.DataFrame,
    factor_pairs: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compute correlations between reconstructed and official factor series.

    Parameters
    ----------
    reconstructed : pd.DataFrame
        Reconstructed factor return series.
    official : pd.DataFrame
        Official FF factors (e.g. from ``fetch_ff5_factors``).
    factor_pairs : dict
        Mapping from reconstructed column name to official column name.
        Defaults to ``{"SMB": "SMB", "HML": "HML"}``.

    Returns
    -------
    pd.DataFrame
        Correlation matrix of the merged factor series.
    """
    if factor_pairs is None:
        factor_pairs = {"SMB": "SMB", "HML": "HML"}

    recon_renamed = reconstructed.rename(columns={k: f"JKP_{k}" for k in factor_pairs})
    merged = recon_renamed.join(official[list(factor_pairs.values())], how="inner").dropna()
    return merged.corr(method="pearson")
