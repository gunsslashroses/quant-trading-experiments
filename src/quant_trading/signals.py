"""Signal generation and combination methods.

Provides:
- ``generate_positions``: cross-sectional percentile-based long/short positions
- ``scale_signal_within_month``: rank or min-max scaling within each month
- ``build_per_signal_panels``: prepare per-characteristic vote and signal panels
- ``build_rolling_ic_weights``: rolling information-coefficient signal weights
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def generate_positions(
    df: pd.DataFrame,
    char_name: str,
    rebal_period: int = 1,
    lower_pct: float = 0.3,
    upper_pct: float = 0.7,
    long_high: bool = True,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Generate long/short positions from cross-sectional percentile ranks.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``id``, ``month_date``, and *char_name*.
    char_name : str
        Column to rank stocks on.
    rebal_period : int
        Rebalance every *N*-th month (1 = monthly).
    lower_pct, upper_pct : float
        Percentile cutoffs.  Stocks below *lower_pct* and above *upper_pct*
        are assigned to the short and long legs respectively (or vice versa
        depending on *long_high*).
    long_high : bool
        If True, long the top percentile; if False, long the bottom.
    group_col : str or None
        Optional column for conditional (double) sorting.

    Returns
    -------
    pd.DataFrame
        Columns: ``id``, ``month_date``, ``position`` (1.0 / -1.0 / 0.0).
    """
    cols_to_keep = ["id", "month_date", char_name]
    if group_col is not None:
        cols_to_keep.append(group_col)
    work_df = df[cols_to_keep].copy()

    if rebal_period > 1:
        month_seq = pd.Series(work_df["month_date"].drop_duplicates().sort_values().values)
        rebal_months = month_seq.iloc[::rebal_period]
        rebal_df = work_df[work_df["month_date"].isin(rebal_months)].copy()
    else:
        rebal_df = work_df.copy()

    group_keys = ["month_date", group_col] if group_col is not None else ["month_date"]
    rebal_df["percentile"] = rebal_df.groupby(group_keys)[char_name].rank(pct=True)

    rebal_df["position"] = 0.0
    if long_high:
        rebal_df.loc[rebal_df["percentile"] >= upper_pct, "position"] = 1.0
        rebal_df.loc[rebal_df["percentile"] <= lower_pct, "position"] = -1.0
    else:
        rebal_df.loc[rebal_df["percentile"] >= upper_pct, "position"] = -1.0
        rebal_df.loc[rebal_df["percentile"] <= lower_pct, "position"] = 1.0

    if rebal_period > 1:
        full_panel = df[["id", "month_date"]].drop_duplicates()
        merged_df = pd.merge(
            full_panel,
            rebal_df[["id", "month_date", "position"]],
            on=["id", "month_date"],
            how="left",
        )
        merged_df["position"] = merged_df.groupby("id")["position"].ffill(limit=rebal_period - 1)
        return merged_df[["id", "month_date", "position"]]

    return rebal_df[["id", "month_date", "position"]]


# ---------------------------------------------------------------------------
# Signal scaling
# ---------------------------------------------------------------------------


def scale_signal_within_month(
    df: pd.DataFrame,
    char_col: str,
    date_col: str = "month_date",
    method: str = "rank",
) -> pd.DataFrame:
    """Scale a characteristic to [0, 1] within each month.

    Parameters
    ----------
    method : {"rank", "minmax"}
    """
    d = df.copy()
    if method == "rank":
        d["_scaled_0_1"] = d.groupby(date_col)[char_col].rank(method="average", pct=True).clip(0, 1)
    elif method == "minmax":
        g = d.groupby(date_col)[char_col]
        minv = g.transform("min")
        maxv = g.transform("max")
        denom = maxv - minv
        d["_scaled_0_1"] = np.where(denom == 0, 0.5, (d[char_col] - minv) / denom)
        d["_scaled_0_1"] = pd.to_numeric(d["_scaled_0_1"], errors="coerce").clip(0, 1)
    else:
        raise ValueError("method must be one of {'rank', 'minmax'}")
    return d


def signed_signal_from_0_1(x: pd.Series, long_high: bool = True) -> pd.Series:
    """Map a [0, 1] scaled signal to [-1, 1], flipping sign if *long_high* is False."""
    s = 2 * x - 1
    return s if long_high else -s


# ---------------------------------------------------------------------------
# Multi-signal panel builders
# ---------------------------------------------------------------------------


def build_per_signal_panels(
    data: pd.DataFrame,
    char_specs: Sequence[dict],
    lower_pct: float,
    upper_pct: float,
    rebal_period: int,
    signal_scale_method: str,
    id_col: str = "id",
    date_col: str = "month_date",
    ret_col: str = "ret_exc_lead1m_w",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-characteristic vote and signal panels.

    Parameters
    ----------
    char_specs : list of dict
        Each dict has keys ``char_name`` (str) and ``long_high`` (bool).

    Returns
    -------
    vote_panel : pd.DataFrame
        Long-format with columns ``id``, ``month_date``, ``position``, ``char_name``.
    signal_panel : pd.DataFrame
        Long-format with scaled/signed signal columns.
    """
    vote_frames: list[pd.DataFrame] = []
    signal_frames: list[pd.DataFrame] = []

    for spec in char_specs:
        char_name = spec["char_name"]
        long_high = bool(spec["long_high"])

        needed = [id_col, date_col, char_name, ret_col]
        d = data[[c for c in needed if c in data.columns]].copy()
        d = d.dropna(subset=[char_name])
        if d.empty:
            continue

        # Votes (for consensus method)
        pos = generate_positions(
            d[[id_col, date_col, char_name]].copy(),
            char_name=char_name,
            lower_pct=lower_pct,
            upper_pct=upper_pct,
            long_high=long_high,
            rebal_period=rebal_period,
        )
        pos = pos[[id_col, date_col, "position"]].copy()
        pos["char_name"] = char_name
        vote_frames.append(pos)

        # Scaled / signed signal
        sig = scale_signal_within_month(
            d[[id_col, date_col, char_name]].copy(),
            char_col=char_name,
            date_col=date_col,
            method=signal_scale_method,
        )
        sig["base_signal"] = 2 * sig["_scaled_0_1"] - 1
        sig["signed_signal"] = signed_signal_from_0_1(sig["_scaled_0_1"], long_high=long_high)
        sig["char_name"] = char_name
        signal_frames.append(
            sig[[id_col, date_col, "char_name", "_scaled_0_1", "base_signal", "signed_signal"]]
        )

    if not vote_frames:
        raise ValueError("No per-signal vote panels generated; check char_specs and data coverage.")
    if not signal_frames:
        raise ValueError(
            "No per-signal signal panels generated; check char_specs and data coverage."
        )

    return pd.concat(vote_frames, ignore_index=True), pd.concat(signal_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Rolling IC weights (Method 3)
# ---------------------------------------------------------------------------


def _cross_sectional_ic(
    group: pd.DataFrame,
    signal_col: str = "signed_signal",
) -> float:
    """Spearman rank correlation between signal and next-period return."""
    g = group[[signal_col, "ret_next"]].dropna()
    if len(g) < 3:
        return np.nan
    return g[signal_col].corr(g["ret_next"], method="spearman")


def build_rolling_ic_weights(
    signal_panel: pd.DataFrame,
    return_panel: pd.DataFrame,
    lookback_months: int = 18,
    min_history_months: int = 6,
    use_abs_ic: bool = False,
    clip_negative_ic_to_zero: bool = True,
    signal_col: str = "signed_signal",
    allow_negative_weights: bool = False,
    id_col: str = "id",
    date_col: str = "month_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute rolling-window IC-based signal weights.

    Returns
    -------
    weight_panel : pd.DataFrame
        Columns ``month_date``, ``char_name``, ``signal_weight``.
    ic_diagnostics : pd.DataFrame
        Raw IC values.
    """
    merged = signal_panel.merge(return_panel, on=[id_col, date_col], how="left")
    merged = merged.rename(columns={"ret_exc_lead1m_w": "ret_next"})

    ic = (
        merged.groupby([date_col, "char_name"])
        .apply(lambda g: _cross_sectional_ic(g, signal_col=signal_col), include_groups=False)
        .rename("ic")
        .reset_index()
    )

    ic_wide = ic.pivot(index=date_col, columns="char_name", values="ic").sort_index()
    months = ic_wide.index.tolist()
    chars = ic_wide.columns.tolist()

    rows: list[dict] = []
    for m in months:
        hist = ic_wide.loc[ic_wide.index < m].tail(lookback_months)
        if hist.shape[0] < min_history_months:
            w = pd.Series(1.0 / len(chars), index=chars)
        else:
            raw = hist.mean(skipna=True)
            if use_abs_ic:
                raw = raw.abs()
            if clip_negative_ic_to_zero:
                raw = raw.clip(lower=0)
            raw = raw.fillna(0)

            if allow_negative_weights:
                denom = raw.abs().sum()
                w = pd.Series(1.0 / len(chars), index=chars) if denom <= 0 else raw / denom
            else:
                total = raw.sum()
                w = pd.Series(1.0 / len(chars), index=chars) if total <= 0 else raw / total

        for c in chars:
            rows.append({date_col: m, "char_name": c, "signal_weight": float(w[c])})

    return pd.DataFrame(rows), ic
