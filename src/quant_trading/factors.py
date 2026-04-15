"""Fama-French factor construction utilities.

This module focuses on the classic Fama-French 2x3 stock sorts used to build
SMB and HML. The implementation is intentionally explicit because the key ideas
are methodological rather than algorithmic:

* use a CRSP-like common-stock universe,
* compute breakpoints from NYSE stocks only,
* rebalance once each June, and
* hold the portfolios from July through the next June.

References
----------
Fama, E. F., & French, K. R. (1992). The Cross-Section of Expected Stock
Returns.

Fama, E. F., & French, K. R. (1993). Common Risk Factors in the Returns on
Stocks and Bonds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_2X3_PORTFOLIOS: tuple[str, ...] = ("SL", "SM", "SH", "BL", "BM", "BH")


@dataclass(frozen=True)
class FamaFrenchReplicationResult:
    """Intermediate outputs from the SMB/HML replication pipeline."""

    universe: pd.DataFrame
    june_breakpoints: pd.DataFrame
    june_assignments: pd.DataFrame
    monthly_assignments: pd.DataFrame
    portfolio_returns: pd.DataFrame
    factors: pd.DataFrame


def _require_columns(df: pd.DataFrame, required: list[str], step_name: str) -> None:
    """Raise a clear error when a pipeline step is missing inputs."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{step_name} requires columns {missing}, but they are missing.")


def prepare_ff_universe(
    df: pd.DataFrame,
    *,
    min_history_months: int = 24,
    allowed_exchanges: tuple[int, ...] = (1, 2, 3),
) -> pd.DataFrame:
    """Apply the stock-universe filters used for FF-style 2x3 sorts.

    Parameters
    ----------
    df : pd.DataFrame
        Security-month panel that includes the JKP/WRDS fields used in the
        notebook: ``primary_sec``, ``exch_main``, ``source_crsp``,
        ``market_equity``, and ``be_me``.
    min_history_months : int, default 24
        Approximation to the Fama-French requirement that firms have a minimum
        accounting history before entering the value sorts. With the JKP panel
        we do not reconstruct Compustat fiscal-year counts directly, so we use
        observed security-month history as a transparent proxy.
    allowed_exchanges : tuple[int, ...], default (1, 2, 3)
        Exchange codes to keep. In the WRDS/JKP extract these correspond to
        NYSE, AMEX, and NASDAQ.

    Returns
    -------
    pd.DataFrame
        Filtered panel with helper columns ``year`` and ``month``.
    """
    _require_columns(
        df,
        [
            "id",
            "month_date",
            "primary_sec",
            "exch_main",
            "source_crsp",
            "market_equity",
            "be_me",
        ],
        "prepare_ff_universe",
    )

    universe = df[
        (df["primary_sec"] == 1)
        & (df["exch_main"].isin(allowed_exchanges))
        & (df["source_crsp"] == 1)
        & (df["market_equity"] > 0)
        & (df["be_me"] > 0)
    ].copy()

    universe = universe.sort_values(["id", "month_date"]).copy()
    universe["obs_count"] = universe.groupby("id").cumcount()
    universe = universe[universe["obs_count"] >= min_history_months].copy()

    universe["year"] = universe["month_date"].dt.year
    universe["month"] = universe["month_date"].dt.month
    return universe


def compute_nyse_breakpoints(
    universe: pd.DataFrame,
    *,
    june_month: int = 6,
    nyse_code: int = 1,
) -> pd.DataFrame:
    """Compute annual June size and value breakpoints from NYSE stocks only.

    Fama and French compute breakpoints using NYSE firms rather than the full
    cross-section so that the many small NASDAQ names do not dominate the cut
    points.
    """
    _require_columns(
        universe,
        ["year", "month", "exch_main", "market_equity", "be_me"],
        "compute_nyse_breakpoints",
    )

    june_nyse = universe[
        (universe["month"] == june_month) & (universe["exch_main"] == nyse_code)
    ].copy()
    if june_nyse.empty:
        raise ValueError("No NYSE June observations found; cannot compute breakpoints.")

    size_breaks = (
        june_nyse.groupby("year")["market_equity"].median().rename("size_median").reset_index()
    )
    bm_breaks = (
        june_nyse.groupby("year")["be_me"]
        .quantile([0.3, 0.7])
        .unstack()
        .rename(columns={0.3: "bm30", 0.7: "bm70"})
        .reset_index()
    )
    return size_breaks.merge(bm_breaks, on="year", how="inner")


def assign_ff_portfolios(universe: pd.DataFrame, breakpoints: pd.DataFrame) -> pd.DataFrame:
    """Assign each June observation to one of the six Fama-French portfolios.

    The size split is a 2-way Small/Big sort, while book-to-market is a 3-way
    Low/Medium/High sort. The resulting portfolios are:
    ``SL, SM, SH, BL, BM, BH``.
    """
    _require_columns(
        universe,
        ["id", "year", "month", "market_equity", "be_me"],
        "assign_ff_portfolios",
    )
    _require_columns(
        breakpoints,
        ["year", "size_median", "bm30", "bm70"],
        "assign_ff_portfolios",
    )

    june_universe = universe[universe["month"] == 6].copy()
    assigned = june_universe.merge(breakpoints, on="year", how="left")
    if assigned[["size_median", "bm30", "bm70"]].isna().any().any():
        raise ValueError("Missing breakpoints for at least one June sort year.")

    assigned["size_portfolio"] = np.where(
        assigned["market_equity"] <= assigned["size_median"], "S", "B"
    )
    assigned["value_portfolio"] = np.where(
        assigned["be_me"] <= assigned["bm30"],
        "L",
        np.where(assigned["be_me"] <= assigned["bm70"], "M", "H"),
    )
    assigned["portfolio"] = assigned["size_portfolio"] + assigned["value_portfolio"]

    return assigned[
        [
            "id",
            "year",
            "size_portfolio",
            "value_portfolio",
            "portfolio",
            "size_median",
            "bm30",
            "bm70",
        ]
    ].rename(columns={"year": "sort_year"})


def attach_monthly_portfolios(
    universe: pd.DataFrame,
    june_assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Carry June assignments into the monthly holding-period panel.

    Timing detail
    -------------
    The project stores returns in ``ret_exc_lead1m``, i.e. the return realized
    from month ``t`` to ``t+1``. That means the row dated June already contains
    the July return, so June rows must map to the *current* June sort year. In
    other words, with lead returns the holding-period mapping is:

    * June-December month-date -> current sort year
    * January-May month-date -> prior sort year
    """
    _require_columns(universe, ["id", "year", "month"], "attach_monthly_portfolios")
    _require_columns(
        june_assignments,
        ["id", "sort_year", "portfolio"],
        "attach_monthly_portfolios",
    )

    monthly = universe.copy()
    monthly["ffyear"] = np.where(monthly["month"] >= 6, monthly["year"], monthly["year"] - 1)

    return monthly.merge(
        june_assignments,
        left_on=["id", "ffyear"],
        right_on=["id", "sort_year"],
        how="inner",
    )


def compute_value_weighted_portfolio_returns(
    monthly_assignments: pd.DataFrame,
    *,
    return_col: str = "ret_exc_lead1m",
    weight_col: str = "market_equity",
) -> pd.DataFrame:
    """Compute value-weighted monthly returns for the six 2x3 portfolios.

    ``weight_col`` is measured at the same month-date as the lead return, so it
    acts as the start-of-period market equity for the return that is realized in
    the next month.
    """
    _require_columns(
        monthly_assignments,
        ["month_date", "portfolio", return_col, weight_col],
        "compute_value_weighted_portfolio_returns",
    )

    def _vw_return(group: pd.DataFrame) -> float:
        valid = group.dropna(subset=[return_col, weight_col])
        if valid.empty or valid[weight_col].sum() == 0:
            return np.nan
        return (valid[return_col] * valid[weight_col]).sum() / valid[weight_col].sum()

    portfolio_returns = (
        monthly_assignments.groupby(["month_date", "portfolio"])
        .apply(_vw_return, include_groups=False)
        .unstack("portfolio")
    )

    portfolio_returns.index = portfolio_returns.index + pd.DateOffset(months=1)
    portfolio_returns.index.name = "date"

    missing = set(REQUIRED_2X3_PORTFOLIOS) - set(portfolio_returns.columns)
    if missing:
        raise ValueError(f"Missing 2x3 portfolios after aggregation: {sorted(missing)}")

    return portfolio_returns.loc[:, list(REQUIRED_2X3_PORTFOLIOS)]


def construct_smb_hml(portfolio_returns: pd.DataFrame) -> pd.DataFrame:
    """Construct SMB and HML from the six underlying 2x3 portfolio returns."""
    _require_columns(
        portfolio_returns,
        list(REQUIRED_2X3_PORTFOLIOS),
        "construct_smb_hml",
    )

    factors = pd.DataFrame(index=portfolio_returns.index)
    factors["SMB"] = (
        portfolio_returns[["SH", "SM", "SL"]].mean(axis=1)
        - portfolio_returns[["BH", "BM", "BL"]].mean(axis=1)
    )
    factors["HML"] = (
        portfolio_returns[["SH", "BH"]].mean(axis=1)
        - portfolio_returns[["SL", "BL"]].mean(axis=1)
    )
    return factors.dropna()


def build_smb_hml(
    df: pd.DataFrame,
    *,
    min_history_months: int = 24,
    return_col: str = "ret_exc_lead1m",
    weight_col: str = "market_equity",
) -> FamaFrenchReplicationResult:
    """Run the full FF-style 2x3 pipeline and return all intermediates."""
    universe = prepare_ff_universe(df, min_history_months=min_history_months)
    june_breakpoints = compute_nyse_breakpoints(universe)
    june_assignments = assign_ff_portfolios(universe, june_breakpoints)
    monthly_assignments = attach_monthly_portfolios(universe, june_assignments)
    portfolio_returns = compute_value_weighted_portfolio_returns(
        monthly_assignments,
        return_col=return_col,
        weight_col=weight_col,
    )
    factors = construct_smb_hml(portfolio_returns)

    return FamaFrenchReplicationResult(
        universe=universe,
        june_breakpoints=june_breakpoints,
        june_assignments=june_assignments,
        monthly_assignments=monthly_assignments,
        portfolio_returns=portfolio_returns,
        factors=factors,
    )


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
