"""Tests for evaluation metrics."""

import numpy as np
import pandas as pd
import pytest

from quant_trading.evaluation import (
    annualized_metrics_from_rets,
    calculate_max_drawdown,
    oos_r2,
    perf_stats_annualized,
)


class TestPerfStats:
    def test_positive_returns(self):
        rets = pd.Series([0.01] * 12)
        stats = perf_stats_annualized(rets, freq=12)
        assert stats["Annualized Return"] > 0
        assert stats["Annualized Vol"] > 0
        assert stats["Sharpe"] > 0
        assert stats["Max Drawdown"] == 0.0  # no drawdown with constant positive returns

    def test_empty_returns(self):
        stats = perf_stats_annualized(pd.Series(dtype=float))
        assert np.isnan(stats["Annualized Return"])

    def test_handles_nan(self):
        rets = pd.Series([0.01, np.nan, 0.02, np.nan, 0.03])
        stats = perf_stats_annualized(rets)
        assert not np.isnan(stats["Sharpe"])


class TestAnnualizedMetrics:
    def test_returns_dataframe(self):
        rets = pd.Series([0.01, -0.005, 0.02, 0.005])
        result = annualized_metrics_from_rets(rets, "test_strat")
        assert isinstance(result, pd.DataFrame)
        assert result.iloc[0]["Strategy"] == "test_strat"
        assert result.iloc[0]["Months"] == 4


class TestOosR2:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert oos_r2(y, y) == pytest.approx(1.0)

    def test_zero_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        pred = np.array([0.0, 0.0, 0.0])
        assert oos_r2(y, pred) == pytest.approx(0.0)

    def test_worse_than_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        pred = np.array([10.0, 10.0, 10.0])
        assert oos_r2(y, pred) < 0


class TestMaxDrawdown:
    def test_no_drawdown(self):
        rets = pd.Series([0.01, 0.02, 0.03, 0.01])
        dd = calculate_max_drawdown(rets)
        assert dd == 0.0  # monotonically increasing

    def test_drawdown_negative(self):
        rets = pd.Series([0.10, -0.20, 0.05, -0.15])
        dd = calculate_max_drawdown(rets)
        assert dd < 0
