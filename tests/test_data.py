"""Tests for data loading and preprocessing utilities."""

import numpy as np
import pandas as pd

from quant_trading.data import rank_standardize_xsec, winsorize_returns


def _make_panel(n_stocks=20, n_months=3, seed=42):
    """Create a small synthetic panel for testing."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    rows = []
    for m in months:
        for i in range(n_stocks):
            rows.append(
                {
                    "id": i,
                    "month_date": m,
                    "ret_exc_lead1m": rng.normal(0, 0.05),
                    "market_equity": rng.uniform(100, 10000),
                    "be_me": rng.uniform(0.1, 3.0),
                    "exch_main": 1 if i < n_stocks // 2 else 2,
                }
            )
    return pd.DataFrame(rows)


class TestWinsorizeReturns:
    def test_creates_new_column(self):
        df = _make_panel()
        result = winsorize_returns(df, lower=0.05, upper=0.95)
        assert "ret_exc_lead1m_w" in result.columns

    def test_does_not_modify_original(self):
        df = _make_panel()
        original_vals = df["ret_exc_lead1m"].copy()
        winsorize_returns(df, lower=0.05, upper=0.95)
        pd.testing.assert_series_equal(df["ret_exc_lead1m"], original_vals)

    def test_winsorized_within_bounds(self):
        df = _make_panel(n_stocks=100, n_months=5)
        result = winsorize_returns(df, lower=0.1, upper=0.9)
        for _, group in result.groupby("month_date"):
            lo = group["ret_exc_lead1m"].quantile(0.1)
            hi = group["ret_exc_lead1m"].quantile(0.9)
            assert result.loc[group.index, "ret_exc_lead1m_w"].min() >= lo - 1e-10
            assert result.loc[group.index, "ret_exc_lead1m_w"].max() <= hi + 1e-10


class TestRankStandardize:
    def test_output_range(self):
        s = pd.Series([10, 20, 30, 40, 50])
        result = rank_standardize_xsec(s)
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_handles_nan(self):
        s = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        result = rank_standardize_xsec(s)
        assert np.isnan(result.iloc[1])
        assert np.isnan(result.iloc[3])
        assert not np.isnan(result.iloc[0])

    def test_all_nan_returns_all_nan(self):
        s = pd.Series([np.nan, np.nan, np.nan])
        result = rank_standardize_xsec(s)
        assert result.isna().all()
