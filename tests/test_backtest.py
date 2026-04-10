"""Tests for the backtesting engine."""

import pandas as pd

from quant_trading.backtest import BacktestResult, run_backtest
from quant_trading.strategy import TradeSignal


def test_run_backtest_no_signals():
    result = run_backtest([], initial_capital=10_000.0)
    assert result.final_value == 10_000.0
    assert result.total_return_pct == 0.0
    assert result.num_trades == 0


def test_run_backtest_buy_and_sell():
    signals = [
        TradeSignal(timestamp=pd.Timestamp("2024-01-10"), action="BUY", price=100.0),
        TradeSignal(timestamp=pd.Timestamp("2024-02-15"), action="SELL", price=120.0),
    ]
    result = run_backtest(signals, initial_capital=10_000.0)

    assert isinstance(result, BacktestResult)
    assert result.num_trades == 2
    assert result.final_value == 12_000.0
    assert result.total_return_pct == 20.0


def test_run_backtest_buy_hold():
    signals = [
        TradeSignal(timestamp=pd.Timestamp("2024-01-10"), action="BUY", price=100.0),
    ]
    result = run_backtest(signals, initial_capital=10_000.0)
    assert result.final_value == 10_000.0
    assert result.num_trades == 1
