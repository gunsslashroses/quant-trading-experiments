"""Portfolio construction and return calculation.

Two main engines:

1. ``calculate_portfolio_returns`` — original Fama-French-style engine
   (equal, value, char_rank_weighted, char_minmax_weighted).
2. ``calculate_portfolio_returns_generalized`` — extended engine for
   composite-signal methods (adds signal_abs, signal_rank, signal_minmax
   weight schemes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Engine 1: Classic factor-portfolio returns
# ---------------------------------------------------------------------------


def _apply_leg_cap_with_fallback(norm_w: pd.Series, cap: float) -> pd.Series:
    """Apply a per-name cap while keeping each leg fully invested."""
    w = norm_w.to_numpy().astype(float)
    n = len(w)
    if n == 0:
        return norm_w

    if cap * n < 1.0:
        w_clipped = np.minimum(w, cap)
        s = w_clipped.sum()
        if s > 0:
            w_final = w_clipped / s
        else:
            w_final = np.full_like(w, 1.0 / n)
        return pd.Series(w_final, index=norm_w.index)

    w_current = w.copy()
    capped = np.zeros(n, dtype=bool)

    for _ in range(10):
        over = w_current > cap + 1e-12
        if not over.any():
            break

        excess = w_current[over] - cap
        w_current[over] = cap
        capped |= over

        total_excess = excess.sum()
        if total_excess <= 0:
            break

        alloc_mask = ~capped
        if not alloc_mask.any():
            break

        w_alloc = w_current[alloc_mask]
        s_alloc = w_alloc.sum()
        if s_alloc <= 0:
            w_current[alloc_mask] = total_excess / alloc_mask.sum()
        else:
            w_current[alloc_mask] = w_alloc + (w_alloc / s_alloc) * total_excess

    s_final = w_current.sum()
    if s_final > 0:
        w_current /= s_final
    else:
        w_current = np.full_like(w_current, 1.0 / n)

    return pd.Series(w_current, index=norm_w.index)


def calculate_portfolio_returns(
    positions: pd.DataFrame,
    data: pd.DataFrame,
    ret_col: str = "ret_exc_lead1m",
    weight_scheme: str = "equal",
    weight_col: str | None = None,
    char_col: str | None = None,
    drop_micro_caps: bool = False,
    long_high: bool = True,
    max_weight_per_leg: float | None = None,
) -> pd.DataFrame:
    """Compute long/short portfolio returns with various weighting schemes.

    Parameters
    ----------
    positions : pd.DataFrame
        Columns ``id``, ``month_date``, ``position``.
    data : pd.DataFrame
        Must contain ``id``, ``month_date``, *ret_col* and optionally
        ``market_equity``, *weight_col*, *char_col*.
    weight_scheme : str
        One of ``equal``, ``value``, ``char_rank_weighted``,
        ``char_minmax_weighted``.
    max_weight_per_leg : float or None
        Per-name cap for each leg. If the cap is feasible for a fully
        invested leg, enforce it via iterative clip-and-redistribute.
        Otherwise, clip once and renormalize.
    """
    merged = pd.merge(positions, data, on=["id", "month_date"], how="left")
    active = merged[(merged["position"] == 1.0) | (merged["position"] == -1.0)].copy()
    active = active.dropna(subset=[ret_col])

    if drop_micro_caps and "market_equity" in active.columns:
        p05 = active.groupby("month_date")["market_equity"].transform(lambda x: x.quantile(0.05))
        active = active[active["market_equity"] >= p05]

    # Raw weights
    if weight_scheme == "equal":
        active["raw_weight"] = 1.0

    elif weight_scheme == "value" and weight_col is not None:
        active["raw_weight"] = active[weight_col]

    elif weight_scheme == "char_rank_weighted" and char_col is not None:
        ranks = active.groupby(["month_date", "position"])[char_col].rank(pct=True)
        if long_high:
            intensity = np.where(active["position"] == 1.0, ranks, 1.0 - ranks)
        else:
            intensity = np.where(active["position"] == 1.0, 1.0 - ranks, ranks)
        active["raw_weight"] = np.clip(intensity, 1e-6, 1.0 - 1e-6)

    elif weight_scheme == "char_minmax_weighted" and char_col is not None:
        x = active[char_col]
        x_min = active.groupby(["month_date", "position"])[char_col].transform("min")
        x_max = active.groupby(["month_date", "position"])[char_col].transform("max")
        scaled = (x - x_min) / (x_max - x_min + 1e-8)
        if long_high:
            intensity = np.where(active["position"] == 1.0, scaled, 1.0 - scaled)
        else:
            intensity = np.where(active["position"] == 1.0, 1.0 - scaled, scaled)
        active["raw_weight"] = np.clip(intensity, 1e-6, 1.0 - 1e-6)

    else:
        raise ValueError(f"Invalid weight_scheme='{weight_scheme}' or missing weight_col/char_col.")

    active = active.dropna(subset=["raw_weight"])

    # Normalize within each leg
    active["norm_w"] = active["raw_weight"] / active.groupby(["month_date", "position"])[
        "raw_weight"
    ].transform("sum")

    # Optional per-name cap
    if max_weight_per_leg is not None:
        cap = float(max_weight_per_leg)
        active["norm_w_final"] = active.groupby(["month_date", "position"])["norm_w"].transform(
            lambda w: _apply_leg_cap_with_fallback(w, cap)
        )
    else:
        active["norm_w_final"] = active["norm_w"]

    active["w_ret"] = active["norm_w_final"] * active[ret_col]
    leg_rets = active.groupby(["month_date", "position"])["w_ret"].sum().unstack()
    leg_rets = leg_rets.rename(columns={1.0: "Long", -1.0: "Short"})
    leg_rets["Spread"] = leg_rets["Long"] - leg_rets["Short"]
    return leg_rets


# ---------------------------------------------------------------------------
# Helper: normalize weights by leg
# ---------------------------------------------------------------------------


def _normalize_leg_weights(
    raw_weights: pd.Series,
    positions: pd.Series,
    max_weight_per_leg: float | None = None,
) -> pd.Series:
    out = pd.Series(0.0, index=raw_weights.index)
    long_mask = positions == 1
    short_mask = positions == -1

    long_sum = raw_weights[long_mask].sum()
    short_sum = raw_weights[short_mask].sum()

    if long_sum > 0:
        out.loc[long_mask] = raw_weights[long_mask] / long_sum
    elif long_mask.any():
        out.loc[long_mask] = 1.0 / long_mask.sum()

    if short_sum > 0:
        out.loc[short_mask] = raw_weights[short_mask] / short_sum
    elif short_mask.any():
        out.loc[short_mask] = 1.0 / short_mask.sum()

    if max_weight_per_leg is not None:
        out = out.clip(upper=max_weight_per_leg)
        long_sum = out[long_mask].sum()
        short_sum = out[short_mask].sum()
        if long_sum > 0:
            out.loc[long_mask] = out.loc[long_mask] / long_sum
        if short_sum > 0:
            out.loc[short_mask] = out.loc[short_mask] / short_sum

    return out


# ---------------------------------------------------------------------------
# Engine 2: Generalized portfolio returns (for composite signals)
# ---------------------------------------------------------------------------


def calculate_portfolio_returns_generalized(
    positions: pd.DataFrame,
    data: pd.DataFrame,
    ret_col: str = "ret_exc_lead1m_w",
    weight_scheme: str = "equal",
    weight_col: str = "market_equity",
    signal_col: str | None = None,
    max_weight_per_leg: float = 0.20,
    id_col: str = "id",
    date_col: str = "month_date",
) -> pd.DataFrame:
    """Extended portfolio return engine supporting signal-based weighting.

    Additional weight schemes beyond Engine 1:
    ``signal_abs``, ``signal_rank``, ``signal_minmax``.
    """
    needed = [id_col, date_col, ret_col, weight_col]
    if signal_col is not None:
        needed.append(signal_col)
    d = data[[c for c in sorted(set(needed)) if c in data.columns]].copy()

    p_cols = [id_col, date_col, "position"]
    if signal_col is not None and signal_col in positions.columns:
        p_cols.append(signal_col)
    p = positions[p_cols].copy()

    merged = p.merge(d, on=[id_col, date_col], how="left", suffixes=("", "_data"))
    merged = merged.dropna(subset=[ret_col])

    def _month_portfolio(g: pd.DataFrame) -> pd.Series:
        g = g.copy()
        g = g[g["position"] != 0].copy()
        if g.empty:
            return pd.Series({"Long": np.nan, "Short": np.nan, "Spread": np.nan})

        if weight_scheme == "equal":
            g["_raw_w"] = 1.0
        elif weight_scheme == "value":
            g["_raw_w"] = g[weight_col].fillna(0).clip(lower=0)
        elif weight_scheme == "signal_abs":
            if signal_col is None or signal_col not in g.columns:
                raise ValueError("signal_col required for signal_abs")
            g["_raw_w"] = g[signal_col].abs().fillna(0)
        elif weight_scheme == "signal_rank":
            if signal_col is None or signal_col not in g.columns:
                raise ValueError("signal_col required for signal_rank")
            g["_raw_w"] = 0.0
            longs = g["position"] == 1
            shorts = g["position"] == -1
            if longs.any():
                g.loc[longs, "_raw_w"] = g.loc[longs, signal_col].rank(method="average", pct=True)
            if shorts.any():
                g.loc[shorts, "_raw_w"] = (-g.loc[shorts, signal_col]).rank(
                    method="average", pct=True
                )
        elif weight_scheme == "signal_minmax":
            if signal_col is None or signal_col not in g.columns:
                raise ValueError("signal_col required for signal_minmax")
            g["_raw_w"] = 0.0
            longs = g["position"] == 1
            if longs.any():
                s = g.loc[longs, signal_col]
                denom = s.max() - s.min()
                g.loc[longs, "_raw_w"] = 1.0 if denom == 0 else (s - s.min()) / denom + 1e-12
            shorts = g["position"] == -1
            if shorts.any():
                s = -g.loc[shorts, signal_col]
                denom = s.max() - s.min()
                g.loc[shorts, "_raw_w"] = 1.0 if denom == 0 else (s - s.min()) / denom + 1e-12
        else:
            raise ValueError(
                f"weight_scheme must be one of "
                f"{{'equal','value','signal_abs','signal_rank','signal_minmax'}}, "
                f"got '{weight_scheme}'"
            )

        g["_raw_w"] = g["_raw_w"].replace([np.inf, -np.inf], np.nan).fillna(0)
        if (g["_raw_w"] <= 0).all():
            g["_raw_w"] = 1.0

        g["_norm_w"] = _normalize_leg_weights(
            raw_weights=g["_raw_w"],
            positions=g["position"],
            max_weight_per_leg=max_weight_per_leg,
        )

        long_ret = (g.loc[g["position"] == 1, "_norm_w"] * g.loc[g["position"] == 1, ret_col]).sum()
        short_ret = (
            g.loc[g["position"] == -1, "_norm_w"] * g.loc[g["position"] == -1, ret_col]
        ).sum()
        return pd.Series({"Long": long_ret, "Short": short_ret, "Spread": long_ret - short_ret})

    return merged.groupby(date_col, as_index=True).apply(_month_portfolio, include_groups=False)
