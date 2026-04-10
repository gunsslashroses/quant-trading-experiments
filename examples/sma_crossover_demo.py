#!/usr/bin/env python3
"""Demo: Simple Moving Average crossover strategy on synthetic price data.

Generates synthetic price data, runs the SMA crossover strategy,
backtests the signals, and plots the results.
"""

import numpy as np
import pandas as pd

from quant_trading.backtest import run_backtest
from quant_trading.strategy import generate_signals, moving_average


def generate_synthetic_prices(n_days: int = 252, seed: int = 42) -> pd.DataFrame:
    """Generate one year of synthetic daily price data with realistic patterns."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")

    returns = rng.normal(loc=0.0003, scale=0.015, size=n_days)
    price = 100.0
    prices = []
    for r in returns:
        price *= 1 + r
        prices.append(price)

    return pd.DataFrame({"close": prices}, index=dates)


def main():
    print("=" * 60)
    print("  Quant Trading Experiments — SMA Crossover Demo")
    print("=" * 60)

    prices = generate_synthetic_prices()
    print(f"\nGenerated {len(prices)} days of synthetic price data")
    print(f"  Start price: ${prices['close'].iloc[0]:.2f}")
    print(f"  End price:   ${prices['close'].iloc[-1]:.2f}")
    print(f"  Date range:  {prices.index[0].date()} to {prices.index[-1].date()}")

    short_w, long_w = 10, 30
    signals = generate_signals(prices, short_window=short_w, long_window=long_w)
    print(f"\nSMA Crossover (short={short_w}, long={long_w})")
    print(f"  Signals generated: {len(signals)}")

    for s in signals[:5]:
        print(f"    {s.timestamp.date()} — {s.action} @ ${s.price:.2f}")
    if len(signals) > 5:
        print(f"    ... and {len(signals) - 5} more")

    result = run_backtest(signals, initial_capital=10_000.0)
    print("\nBacktest Results:")
    print(f"  Initial capital: ${result.initial_capital:,.2f}")
    print(f"  Final value:     ${result.final_value:,.2f}")
    print(f"  Total return:    {result.total_return_pct:+.2f}%")
    print(f"  Number of trades: {result.num_trades}")

    buy_hold_return = ((prices["close"].iloc[-1] / prices["close"].iloc[0]) - 1) * 100
    print(f"\n  Buy & Hold return: {buy_hold_return:+.2f}%")
    print(f"  Strategy alpha:    {result.total_return_pct - buy_hold_return:+.2f}%")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ax1.plot(prices.index, prices["close"], label="Close Price", alpha=0.7)
        short_ma = moving_average(prices["close"], short_w)
        long_ma = moving_average(prices["close"], long_w)
        ax1.plot(prices.index, short_ma, label=f"SMA({short_w})", linewidth=1)
        ax1.plot(prices.index, long_ma, label=f"SMA({long_w})", linewidth=1)

        buys = [s for s in signals if s.action == "BUY"]
        sells = [s for s in signals if s.action == "SELL"]
        ax1.scatter(
            [s.timestamp for s in buys],
            [s.price for s in buys],
            marker="^",
            color="green",
            s=100,
            label="Buy",
            zorder=5,
        )
        ax1.scatter(
            [s.timestamp for s in sells],
            [s.price for s in sells],
            marker="v",
            color="red",
            s=100,
            label="Sell",
            zorder=5,
        )
        ax1.set_title("SMA Crossover Strategy")
        ax1.set_ylabel("Price ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        daily_returns = prices["close"].pct_change().dropna()
        ax2.hist(daily_returns, bins=50, alpha=0.7, edgecolor="black")
        ax2.set_title("Daily Returns Distribution")
        ax2.set_xlabel("Return")
        ax2.set_ylabel("Frequency")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("sma_crossover_results.png", dpi=150)
        print("\n  Chart saved to sma_crossover_results.png")
    except ImportError:
        print("\n  (matplotlib not available — skipping chart)")

    print("\n" + "=" * 60)
    print("  Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
