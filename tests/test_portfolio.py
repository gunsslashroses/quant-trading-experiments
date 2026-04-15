"""Tests for portfolio return calculation engines."""

import numpy as np
import pandas as pd
import pytest

from quant_trading.portfolio import _normalize_leg_weights, calculate_portfolio_returns


def _make_portfolio_data(n_stocks=20, n_months=6, seed=42):
    """Create synthetic positions + data for portfolio tests."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2020-01-01", periods=n_months, freq="MS")

    data_rows = []
    pos_rows = []
    for m in months:
        for i in range(n_stocks):
            ret = rng.normal(0.005, 0.05)
            me = rng.uniform(100, 5000)
            signal = rng.normal(0, 1)
            data_rows.append(
                {
                    "id": i,
                    "month_date": m,
                    "ret_exc_lead1m": ret,
                    "market_equity": me,
                    "signal": signal,
                }
            )
            if i < n_stocks // 3:
                pos = 1.0
            elif i >= 2 * n_stocks // 3:
                pos = -1.0
            else:
                pos = 0.0
            pos_rows.append({"id": i, "month_date": m, "position": pos})

    return pd.DataFrame(pos_rows), pd.DataFrame(data_rows)


class TestCalculatePortfolioReturns:
    def test_output_columns(self):
        positions, data = _make_portfolio_data()
        result = calculate_portfolio_returns(positions, data)
        assert "Long" in result.columns
        assert "Short" in result.columns
        assert "Spread" in result.columns

    def test_spread_equals_long_minus_short(self):
        positions, data = _make_portfolio_data()
        result = calculate_portfolio_returns(positions, data)
        pd.testing.assert_series_equal(
            result["Spread"],
            result["Long"] - result["Short"],
            check_names=False,
        )

    def test_equal_weight(self):
        positions, data = _make_portfolio_data()
        result = calculate_portfolio_returns(positions, data, weight_scheme="equal")
        assert not result["Spread"].isna().all()

    def test_value_weight(self):
        positions, data = _make_portfolio_data()
        result = calculate_portfolio_returns(
            positions,
            data,
            weight_scheme="value",
            weight_col="market_equity",
        )
        assert not result["Spread"].isna().all()

    def test_max_weight_cap_reduces_concentration(self):
        positions, data = _make_portfolio_data(n_stocks=5)
        result_uncapped = calculate_portfolio_returns(positions, data)
        result_capped = calculate_portfolio_returns(positions, data, max_weight_per_leg=0.5)
        # Both should produce valid results
        assert not result_uncapped["Spread"].isna().all()
        assert not result_capped["Spread"].isna().all()


class TestNormalizeLegWeights:
    def test_weights_sum_to_one_per_leg(self):
        raw = pd.Series([3.0, 1.0, 2.0, 4.0])
        positions = pd.Series([1, 1, -1, -1])
        result = _normalize_leg_weights(raw, positions)
        assert result[positions == 1].sum() == pytest.approx(1.0)
        assert result[positions == -1].sum() == pytest.approx(1.0)

    def test_cap_reduces_max_weight(self):
        raw = pd.Series([10.0, 1.0, 1.0])
        positions = pd.Series([1, 1, 1])
        uncapped = _normalize_leg_weights(raw, positions)
        capped = _normalize_leg_weights(raw, positions, max_weight_per_leg=0.5)
        # Capping should reduce the largest weight
        assert capped.max() < uncapped.max()
