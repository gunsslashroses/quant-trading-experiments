"""Performance evaluation and reporting utilities.

Provides annualized return stats, OOS R², decile-spread analysis,
and helper functions for comparing models and strategies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def perf_stats_annualized(rets: pd.Series, freq: int = 12) -> pd.Series:
    """Compute annualized performance statistics for a return series.

    Parameters
    ----------
    rets : pd.Series
        Periodic (typically monthly) returns.
    freq : int
        Periods per year (12 for monthly).

    Returns
    -------
    pd.Series
        Keys: ``Annualized Return``, ``Annualized Vol``, ``Sharpe``,
        ``Max Drawdown``.
    """
    rets = pd.Series(rets).dropna()
    if rets.empty:
        return pd.Series(
            {
                "Annualized Return": np.nan,
                "Annualized Vol": np.nan,
                "Sharpe": np.nan,
                "Max Drawdown": np.nan,
            }
        )

    ann_return = (1 + rets).prod() ** (freq / len(rets)) - 1
    ann_vol = rets.std(ddof=1) * np.sqrt(freq)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan

    cum = (1 + rets).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()

    return pd.Series(
        {
            "Annualized Return": ann_return,
            "Annualized Vol": ann_vol,
            "Sharpe": sharpe,
            "Max Drawdown": max_dd,
        }
    )


def annualized_metrics_from_rets(
    rets: pd.Series,
    strategy_name: str,
    freq: int = 12,
) -> pd.DataFrame:
    """Compute annualized metrics and return as a single-row DataFrame.

    Useful for accumulating results across multiple strategy configurations.
    """
    rets = pd.Series(rets).dropna()
    if rets.empty:
        return pd.DataFrame(
            [
                {
                    "Strategy": strategy_name,
                    "Months": 0,
                    "Annualized Mean": np.nan,
                    "Annualized Vol": np.nan,
                    "Sharpe": np.nan,
                    "Max Drawdown": np.nan,
                    "Hit Rate": np.nan,
                    "Best Month": np.nan,
                    "Worst Month": np.nan,
                }
            ]
        )

    ann_mean = rets.mean() * freq
    ann_vol = rets.std(ddof=1) * np.sqrt(freq)
    sharpe = ann_mean / ann_vol if ann_vol != 0 else np.nan
    cum_ret = (1 + rets).cumprod()
    drawdown = cum_ret / cum_ret.cummax() - 1

    return pd.DataFrame(
        [
            {
                "Strategy": strategy_name,
                "Months": len(rets),
                "Annualized Mean": ann_mean,
                "Annualized Vol": ann_vol,
                "Sharpe": sharpe,
                "Max Drawdown": drawdown.min(),
                "Hit Rate": (rets > 0).mean(),
                "Best Month": rets.max(),
                "Worst Month": rets.min(),
            }
        ]
    )


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from a return series."""
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns - running_max) / running_max
    return drawdowns.min()


# ---------------------------------------------------------------------------
# OOS prediction evaluation
# ---------------------------------------------------------------------------


def oos_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Out-of-sample R² (Gu, Kelly, Xiu 2020 definition).

    .. math::
        R^2_{OOS} = 1 - \\frac{\\sum (y - \\hat{y})^2}{\\sum y^2}

    Note: denominator is sum of squared actuals (not centered), which
    differs from sklearn's ``r2_score``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.sum(y_true**2)
    if denom == 0:
        return np.nan
    return 1 - np.sum((y_true - y_pred) ** 2) / denom


def sort_ret_eq_wgt(
    ret_df: pd.DataFrame,
    bins: int = 10,
) -> pd.DataFrame:
    """Bin stocks into quantile portfolios by prediction each month.

    Parameters
    ----------
    ret_df : pd.DataFrame
        Indexed by date with columns ``pred`` and ``ret``.
    bins : int
        Number of quantile bins.

    Returns
    -------
    pd.DataFrame
        Time-series of equal-weight bin returns (columns = bin labels).
    """
    tmp = ret_df.copy().dropna()

    def _assign_bins(g: pd.DataFrame) -> pd.DataFrame:
        if len(g) < bins:
            g["bin"] = np.nan
            return g
        try:
            g["bin"] = pd.qcut(g["pred"], bins, labels=False, duplicates="drop") + 1
        except Exception:
            g["bin"] = np.nan
        return g

    tmp = tmp.groupby(level=0, group_keys=False).apply(_assign_bins)
    tmp = tmp.dropna(subset=["bin"])
    if tmp.empty:
        return pd.DataFrame(dtype="float64")

    tmp["bin"] = tmp["bin"].astype(int)
    return tmp.groupby([tmp.index, "bin"])["ret"].mean().unstack("bin")


def evaluate_model_performance(
    y_true: pd.Series,
    y_pred: pd.Series | np.ndarray,
    model_name: str,
) -> dict:
    """Print and return regression evaluation metrics."""
    from sklearn.metrics import mean_squared_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    sign_acc = np.mean(np.sign(y_true) == np.sign(y_pred))

    print(f"\n{'=' * 40}")
    print(f"{model_name} Performance")
    print(f"{'=' * 40}")
    print(f"MSE:            {mse:.6f}")
    print(f"OOS R-squared:  {r2:.6f}")
    print(f"Sign Accuracy:  {sign_acc:.2%}")

    return {"mse": mse, "r2": r2, "sign_acc": sign_acc}
