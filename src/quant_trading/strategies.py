"""High-level strategy runners for combined-signal methods.

Method 1: Consensus voting (n-out-of-S threshold).
Method 2: Equal-weight composite signal.
Method 3: Rolling-IC-weighted composite signal.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from quant_trading.evaluation import annualized_metrics_from_rets
from quant_trading.portfolio import calculate_portfolio_returns_generalized
from quant_trading.signals import (
    build_per_signal_panels,
    build_rolling_ic_weights,
    generate_positions,
)


def run_method1_consensus_voting(
    data: pd.DataFrame,
    char_specs: Sequence[dict],
    lower_pct: float,
    upper_pct: float,
    rebal_period: int,
    min_net_votes: int,
    portfolio_weight_scheme: str = "equal",
    ret_col: str = "ret_exc_lead1m_w",
    weight_col: str = "market_equity",
    max_weight_per_leg: float = 0.20,
    id_col: str = "id",
    date_col: str = "month_date",
    freq: int = 12,
) -> dict[str, Any]:
    """Run Method 1: n-out-of-S consensus voting.

    Each characteristic casts a +1 (long) or -1 (short) vote.
    A stock enters the long leg when net votes >= *min_net_votes*,
    and the short leg when net votes <= -*min_net_votes*.
    """
    d0 = data.copy()
    d0[date_col] = pd.to_datetime(d0[date_col])

    vote_panel, _ = build_per_signal_panels(
        data=d0,
        char_specs=char_specs,
        lower_pct=lower_pct,
        upper_pct=upper_pct,
        rebal_period=rebal_period,
        signal_scale_method="rank",
        id_col=id_col,
        date_col=date_col,
        ret_col=ret_col,
    )

    S = len(char_specs)
    agg = vote_panel.groupby([id_col, date_col], as_index=False).agg(net_votes=("position", "sum"))

    universe = d0[[id_col, date_col]].drop_duplicates()
    consensus = universe.merge(agg, on=[id_col, date_col], how="left")
    consensus["net_votes"] = consensus["net_votes"].fillna(0)
    consensus["position"] = 0.0
    consensus.loc[consensus["net_votes"] >= min_net_votes, "position"] = 1.0
    consensus.loc[consensus["net_votes"] <= -min_net_votes, "position"] = -1.0

    returns = calculate_portfolio_returns_generalized(
        positions=consensus[[id_col, date_col, "position"]],
        data=d0.dropna(subset=[ret_col]).copy(),
        ret_col=ret_col,
        weight_scheme=portfolio_weight_scheme,
        weight_col=weight_col,
        signal_col=None,
        max_weight_per_leg=max_weight_per_leg,
        id_col=id_col,
        date_col=date_col,
    )

    label = f"M1 Consensus | {portfolio_weight_scheme}"
    metrics = annualized_metrics_from_rets(returns["Spread"], strategy_name=label, freq=freq)
    metrics.insert(0, "Method", "Method 1: Consensus Voting")
    metrics.insert(2, "Signals (S)", S)
    metrics.insert(3, "Min Net Votes (n)", min_net_votes)
    metrics.insert(4, "Portfolio Weight", portfolio_weight_scheme)

    return {"positions": consensus, "returns": returns, "metrics": metrics}


def run_composite_method(
    data: pd.DataFrame,
    char_specs: Sequence[dict],
    lower_pct: float,
    upper_pct: float,
    rebal_period: int,
    signal_scale_method: str,
    signal_weight_method: str,
    portfolio_weight_scheme: str,
    lookback_months: int = 24,
    min_history_months: int = 12,
    use_abs_ic: bool = False,
    clip_negative_ic_to_zero: bool = True,
    auto_direction_from_ic: bool = False,
    ret_col: str = "ret_exc_lead1m_w",
    weight_col: str = "market_equity",
    max_weight_per_leg: float = 0.20,
    id_col: str = "id",
    date_col: str = "month_date",
    freq: int = 12,
) -> dict[str, Any]:
    """Run Method 2 (equal-weight composite) or Method 3 (IC-weighted composite).

    Parameters
    ----------
    signal_weight_method : {"equal", "rolling_ic"}
        ``"equal"`` → Method 2; ``"rolling_ic"`` → Method 3.
    auto_direction_from_ic : bool
        Method 3 only: infer signal direction from IC sign each month
        (ignores ``long_high`` in char_specs).
    """
    d0 = data.copy()
    d0[date_col] = pd.to_datetime(d0[date_col])

    _, signal_panel = build_per_signal_panels(
        data=d0,
        char_specs=char_specs,
        lower_pct=lower_pct,
        upper_pct=upper_pct,
        rebal_period=rebal_period,
        signal_scale_method=signal_scale_method,
        id_col=id_col,
        date_col=date_col,
        ret_col=ret_col,
    )

    signal_for_composite = "signed_signal"
    ic_diag = pd.DataFrame()

    if signal_weight_method == "equal":
        w_panel = (
            signal_panel[[date_col, "char_name"]]
            .drop_duplicates()
            .assign(signal_weight=lambda x: 1.0 / len(char_specs))
        )
        method_label = "Method 2: Equal-Weight Composite"
        strat_label = f"M2 Composite Equal | {portfolio_weight_scheme}"

    elif signal_weight_method == "rolling_ic":
        return_panel = d0[[id_col, date_col, ret_col]].copy()

        if auto_direction_from_ic:
            signal_for_ic = "base_signal"
            signal_for_composite = "base_signal"
            local_use_abs_ic = False
            local_clip_negative_ic_to_zero = False
            allow_negative_weights = True
        else:
            signal_for_ic = "signed_signal"
            signal_for_composite = "signed_signal"
            local_use_abs_ic = use_abs_ic
            local_clip_negative_ic_to_zero = clip_negative_ic_to_zero
            allow_negative_weights = False

        w_panel, ic_diag = build_rolling_ic_weights(
            signal_panel=signal_panel,
            return_panel=return_panel,
            lookback_months=lookback_months,
            min_history_months=min_history_months,
            use_abs_ic=local_use_abs_ic,
            clip_negative_ic_to_zero=local_clip_negative_ic_to_zero,
            signal_col=signal_for_ic,
            allow_negative_weights=allow_negative_weights,
            id_col=id_col,
            date_col=date_col,
        )
        method_label = "Method 3: IC-Weighted Composite"
        strat_label = f"M3 Composite IC-Weighted | {portfolio_weight_scheme}"
    else:
        raise ValueError("signal_weight_method must be 'equal' or 'rolling_ic'")

    # Aggregate weighted signals → composite score per stock-month
    weighted = signal_panel.merge(w_panel, on=[date_col, "char_name"], how="left")
    weighted["signal_weight"] = weighted["signal_weight"].fillna(0)
    weighted["weighted_signal"] = weighted[signal_for_composite] * weighted["signal_weight"]

    composite = weighted.groupby([id_col, date_col], as_index=False).agg(
        composite_signal=("weighted_signal", "sum")
    )

    # Sort into long/short based on composite signal
    comp_positions = generate_positions(
        composite[[id_col, date_col, "composite_signal"]].copy(),
        char_name="composite_signal",
        lower_pct=lower_pct,
        upper_pct=upper_pct,
        long_high=True,
        rebal_period=rebal_period,
    )

    final_positions = comp_positions.merge(composite, on=[id_col, date_col], how="left")

    signal_col = (
        "composite_signal"
        if portfolio_weight_scheme in {"signal_rank", "signal_minmax", "signal_abs"}
        else None
    )

    returns = calculate_portfolio_returns_generalized(
        positions=final_positions[[id_col, date_col, "position", "composite_signal"]],
        data=d0.dropna(subset=[ret_col]).copy(),
        ret_col=ret_col,
        weight_scheme=portfolio_weight_scheme,
        weight_col=weight_col,
        signal_col=signal_col,
        max_weight_per_leg=max_weight_per_leg,
        id_col=id_col,
        date_col=date_col,
    )

    metrics = annualized_metrics_from_rets(returns["Spread"], strategy_name=strat_label, freq=freq)
    metrics.insert(0, "Method", method_label)
    metrics.insert(2, "Signal Weighting", signal_weight_method)
    metrics.insert(3, "Signal Scaling", signal_scale_method)
    metrics.insert(4, "Portfolio Weight", portfolio_weight_scheme)
    metrics.insert(5, "Signals (S)", len(char_specs))

    if signal_weight_method == "rolling_ic":
        metrics.insert(6, "IC Lookback (m)", lookback_months)
        metrics.insert(7, "IC Min Hist (m)", min_history_months)
        metrics.insert(8, "IC Auto Direction", bool(auto_direction_from_ic))
    else:
        metrics.insert(6, "IC Lookback (m)", np.nan)
        metrics.insert(7, "IC Min Hist (m)", np.nan)
        metrics.insert(8, "IC Auto Direction", np.nan)

    return {
        "signal_panel": signal_panel,
        "signal_weights": w_panel,
        "ic_diagnostics": ic_diag,
        "composite_signal": composite,
        "positions": final_positions,
        "returns": returns,
        "metrics": metrics,
    }
