"""Simple moving average crossover strategy."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TradeSignal:
    timestamp: pd.Timestamp
    action: str  # "BUY" or "SELL"
    price: float


def moving_average(prices: pd.Series, window: int) -> pd.Series:
    """Compute simple moving average over a rolling window."""
    return prices.rolling(window=window).mean()


def generate_signals(
    prices: pd.DataFrame,
    short_window: int = 10,
    long_window: int = 50,
) -> list[TradeSignal]:
    """Generate buy/sell signals from a simple moving average crossover.

    A BUY signal is generated when the short MA crosses above the long MA.
    A SELL signal is generated when the short MA crosses below the long MA.
    """
    if "close" not in prices.columns:
        raise ValueError("DataFrame must contain a 'close' column")

    short_ma = moving_average(prices["close"], short_window)
    long_ma = moving_average(prices["close"], long_window)

    signals: list[TradeSignal] = []
    prev_short_above = None

    for i in range(len(prices)):
        if np.isnan(short_ma.iloc[i]) or np.isnan(long_ma.iloc[i]):
            continue

        short_above = short_ma.iloc[i] > long_ma.iloc[i]

        if prev_short_above is not None and short_above != prev_short_above:
            action = "BUY" if short_above else "SELL"
            signals.append(
                TradeSignal(
                    timestamp=prices.index[i],
                    action=action,
                    price=prices["close"].iloc[i],
                )
            )

        prev_short_above = short_above

    return signals
