"""Tests for signal generation and scaling."""

import numpy as np
import pandas as pd
import pytest

from quant_trading.signals import (
    generate_positions,
    scale_signal_within_month,
    signed_signal_from_0_1,
)


def _make_signal_data(n_stocks=50, n_months=3, seed=42):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    rows = []
    for m in months:
        for i in range(n_stocks):
            rows.append(
                {
                    "id": i,
                    "month_date": m,
                    "signal": rng.normal(0, 1),
                    "ret_exc_lead1m_w": rng.normal(0, 0.05),
                }
            )
    return pd.DataFrame(rows)


class TestGeneratePositions:
    def test_output_columns(self):
        df = _make_signal_data()
        result = generate_positions(df, char_name="signal", lower_pct=0.3, upper_pct=0.7)
        assert set(result.columns) == {"id", "month_date", "position"}

    def test_positions_are_valid(self):
        df = _make_signal_data()
        result = generate_positions(df, char_name="signal")
        assert set(result["position"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_long_high_true(self):
        df = _make_signal_data()
        result = generate_positions(
            df,
            char_name="signal",
            long_high=True,
            lower_pct=0.3,
            upper_pct=0.7,
        )
        merged = df.merge(result, on=["id", "month_date"])
        longs = merged[merged["position"] == 1.0]
        shorts = merged[merged["position"] == -1.0]
        # On average, longs should have higher signal than shorts
        assert longs["signal"].mean() > shorts["signal"].mean()

    def test_long_high_false_reverses(self):
        df = _make_signal_data()
        result = generate_positions(
            df,
            char_name="signal",
            long_high=False,
            lower_pct=0.3,
            upper_pct=0.7,
        )
        merged = df.merge(result, on=["id", "month_date"])
        longs = merged[merged["position"] == 1.0]
        shorts = merged[merged["position"] == -1.0]
        assert longs["signal"].mean() < shorts["signal"].mean()


class TestScaleSignal:
    def test_rank_scaling_range(self):
        df = _make_signal_data()
        result = scale_signal_within_month(df, char_col="signal", method="rank")
        assert result["_scaled_0_1"].min() >= 0.0
        assert result["_scaled_0_1"].max() <= 1.0

    def test_minmax_scaling_range(self):
        df = _make_signal_data()
        result = scale_signal_within_month(df, char_col="signal", method="minmax")
        assert result["_scaled_0_1"].min() >= 0.0
        assert result["_scaled_0_1"].max() <= 1.0

    def test_invalid_method_raises(self):
        df = _make_signal_data()
        with pytest.raises(ValueError, match="method"):
            scale_signal_within_month(df, char_col="signal", method="bad")


class TestSignedSignal:
    def test_long_high_true(self):
        x = pd.Series([0.0, 0.5, 1.0])
        result = signed_signal_from_0_1(x, long_high=True)
        assert result.iloc[0] == pytest.approx(-1.0)
        assert result.iloc[1] == pytest.approx(0.0)
        assert result.iloc[2] == pytest.approx(1.0)

    def test_long_high_false_flips(self):
        x = pd.Series([0.0, 0.5, 1.0])
        result = signed_signal_from_0_1(x, long_high=False)
        assert result.iloc[0] == pytest.approx(1.0)
        assert result.iloc[1] == pytest.approx(0.0)
        assert result.iloc[2] == pytest.approx(-1.0)
