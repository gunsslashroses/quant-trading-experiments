"""Plotting helpers for factor returns, strategy comparison, and diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_cumulative_log_returns(
    series_dict: dict[str, pd.Series],
    title: str = "Cumulative Log Returns",
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Plot cumulative log returns for one or more named return series."""
    fig, ax = plt.subplots(figsize=figsize)
    for label, rets in series_dict.items():
        cum_log = np.log(1 + rets).cumsum()
        ax.plot(cum_log.index, cum_log.values, label=label, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Log Return")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_cumulative_returns(
    return_dict: dict[str, pd.Series],
    title: str = "Cumulative Return Comparison",
    figsize: tuple[int, int] = (12, 7),
) -> plt.Figure:
    """Plot growth-of-$1 for multiple strategy return series."""
    fig, ax = plt.subplots(figsize=figsize)
    for label, rets in return_dict.items():
        rets = pd.Series(rets).dropna()
        if rets.empty:
            continue
        cum = (1 + rets).cumprod()
        ax.plot(cum.index, cum.values, label=label, linewidth=2)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_factor_comparison(
    reconstructed: pd.Series,
    official: pd.Series,
    factor_name: str,
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Side-by-side cumulative log return plot for reconstructed vs official."""
    fig, ax = plt.subplots(figsize=figsize)
    cum_recon = np.log(1 + reconstructed).cumsum()
    cum_off = np.log(1 + official).cumsum()
    ax.plot(cum_recon.index, cum_recon.values, label=f"Reconstructed {factor_name}", linewidth=2)
    ax.plot(
        cum_off.index,
        cum_off.values,
        linestyle="--",
        label=f"Official {factor_name}",
        alpha=0.8,
    )
    ax.set_title(f"Cumulative Log Returns: {factor_name} (Reconstructed vs Official)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Log Return")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig
