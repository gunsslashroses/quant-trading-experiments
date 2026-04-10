"""Tests for the moving average crossover strategy."""

import numpy as np
import pandas as pd
import pytest

from quant_trading.strategy import TradeSignal, generate_signals, moving_average


def test_moving_average_basic():
    prices = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = moving_average(prices, window=3)
    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[3] == pytest.approx(3.0)
    assert result.iloc[4] == pytest.approx(4.0)


def test_generate_signals_requires_close_column():
    df = pd.DataFrame({"open": [1, 2, 3]})
    with pytest.raises(ValueError, match="close"):
        generate_signals(df)


def test_generate_signals_crossover():
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    trend_up = np.linspace(100, 150, n // 2)
    trend_down = np.linspace(150, 90, n // 2)
    prices_arr = np.concatenate([trend_up, trend_down])
    noise = np.random.normal(0, 1, n)
    prices_arr = prices_arr + noise

    df = pd.DataFrame({"close": prices_arr}, index=dates)
    signals = generate_signals(df, short_window=5, long_window=20)

    assert len(signals) > 0
    for s in signals:
        assert isinstance(s, TradeSignal)
        assert s.action in ("BUY", "SELL")
        assert s.price > 0
