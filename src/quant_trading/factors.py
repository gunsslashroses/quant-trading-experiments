"""Fama-French style factor construction.

Implements the 2×3 sort methodology using NYSE breakpoints to build
SMB and HML factors from JKP data, and provides comparison utilities
against the official Fama-French factors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_ff_factors(
    df: pd.DataFrame,
    char_name: str = "be_me",
) -> pd.DataFrame:
    """Construct SMB and HML using the Fama-French 2×3 methodology.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``market_equity``, *char_name*, ``ret_exc_lead1m``,
        ``exch_main``, and ``month_date``.
    char_name : str
        Characteristic used for the value dimension (default ``be_me``).

    Returns
    -------
    pd.DataFrame
        Monthly series with columns ``SMB`` and ``HML``, indexed by
        ``month_date``.
    """
    nyse = df[df["exch_main"] == 1].copy()

    # NYSE breakpoints
    s_break = nyse.groupby("month_date")["market_equity"].median().rename("s_break")
    c_breaks = (
        nyse.groupby("month_date")[char_name]
        .quantile([0.3, 0.7])
        .unstack()
        .rename(columns={0.3: "c30", 0.7: "c70"})
    )

    df_p = df.merge(s_break, on="month_date").merge(c_breaks, on="month_date")
    df_p["S_Port"] = np.where(df_p["market_equity"] > df_p["s_break"], "B", "S")
    df_p["C_Port"] = np.where(
        df_p[char_name] > df_p["c70"],
        "H",
        np.where(df_p[char_name] < df_p["c30"], "L", "N"),
    )
    df_p["Portfolio"] = df_p["S_Port"] + df_p["C_Port"]

    valid = df_p.dropna(subset=["ret_exc_lead1m", "market_equity"])

    def wavg(g: pd.DataFrame) -> float:
        return (g["ret_exc_lead1m"] * g["market_equity"]).sum() / g["market_equity"].sum()

    port_rets = (
        valid.groupby(["month_date", "Portfolio"]).apply(wavg, include_groups=False).unstack()
    )

    port_rets["SMB"] = (port_rets["SH"] + port_rets["SN"] + port_rets["SL"]) / 3 - (
        port_rets["BH"] + port_rets["BN"] + port_rets["BL"]
    ) / 3
    port_rets["HML"] = (port_rets["SH"] + port_rets["BH"]) / 2 - (
        port_rets["SL"] + port_rets["BL"]
    ) / 2

    return port_rets[["SMB", "HML"]]


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
        Reconstructed factors (e.g. from ``build_ff_factors``).
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
