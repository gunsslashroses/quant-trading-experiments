"""Tests for Fama-French factor construction helpers."""

import numpy as np
import pandas as pd
import pytest

from quant_trading.factors import (
    attach_monthly_portfolios,
    compute_nyse_breakpoints,
    compute_value_weighted_portfolio_returns,
    construct_smb_hml,
    prepare_ff_universe,
)


def test_prepare_ff_universe_filters_rows_and_adds_calendar_fields():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "month_date": pd.to_datetime(["2020-06-01"] * 4),
            "primary_sec": [1, 0, 1, 1],
            "exch_main": [1, 1, 4, 2],
            "source_crsp": [1, 1, 1, 0],
            "market_equity": [100.0, 200.0, 300.0, 400.0],
            "be_me": [0.5, 0.7, 0.9, 1.1],
        }
    )

    result = prepare_ff_universe(df, min_history_months=0)

    assert result["id"].tolist() == [1]
    assert result["year"].tolist() == [2020]
    assert result["month"].tolist() == [6]


def test_compute_nyse_breakpoints_ignore_non_nyse_stocks():
    universe = pd.DataFrame(
        {
            "year": [2020] * 8,
            "month": [6] * 8,
            "exch_main": [1, 1, 1, 1, 1, 3, 3, 3],
            "market_equity": [10.0, 20.0, 30.0, 40.0, 50.0, 1_000.0, 2_000.0, 3_000.0],
            "be_me": [0.2, 0.4, 0.6, 0.8, 1.0, 10.0, 11.0, 12.0],
        }
    )

    result = compute_nyse_breakpoints(universe)
    nyse = universe[universe["exch_main"] == 1]

    assert result.loc[0, "size_median"] == pytest.approx(nyse["market_equity"].median())
    assert result.loc[0, "bm30"] == pytest.approx(nyse["be_me"].quantile(0.3))
    assert result.loc[0, "bm70"] == pytest.approx(nyse["be_me"].quantile(0.7))


def test_attach_monthly_portfolios_maps_june_to_current_sort_year():
    universe = pd.DataFrame(
        {
            "id": [1, 1, 1, 1, 1],
            "year": [2020, 2020, 2021, 2021, 2021],
            "month": [6, 7, 1, 5, 6],
            "month_date": pd.to_datetime(
                ["2020-06-01", "2020-07-01", "2021-01-01", "2021-05-01", "2021-06-01"]
            ),
        }
    )
    june_assignments = pd.DataFrame(
        {
            "id": [1, 1],
            "sort_year": [2020, 2021],
            "portfolio": ["SH", "BL"],
        }
    )

    result = attach_monthly_portfolios(universe, june_assignments)
    mapped = result.set_index("month_date")["sort_year"]

    assert mapped[pd.Timestamp("2020-06-01")] == 2020
    assert mapped[pd.Timestamp("2020-07-01")] == 2020
    assert mapped[pd.Timestamp("2021-01-01")] == 2020
    assert mapped[pd.Timestamp("2021-05-01")] == 2020
    assert mapped[pd.Timestamp("2021-06-01")] == 2021


def test_compute_value_weighted_portfolio_returns_shift_to_realized_month():
    monthly_assignments = pd.DataFrame(
        {
            "month_date": [pd.Timestamp("2020-06-01")] * 6,
            "portfolio": ["SL", "SM", "SH", "BL", "BM", "BH"],
            "ret_exc_lead1m": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            "market_equity": [1, 1, 1, 1, 1, 1],
        }
    )

    result = compute_value_weighted_portfolio_returns(monthly_assignments)

    assert result.index.tolist() == [pd.Timestamp("2020-07-01")]
    assert result.loc[pd.Timestamp("2020-07-01"), "SH"] == pytest.approx(0.03)
    assert result.loc[pd.Timestamp("2020-07-01"), "BL"] == pytest.approx(0.04)


def test_construct_smb_hml_matches_two_by_three_formulas():
    portfolio_returns = pd.DataFrame(
        {
            "SL": [0.01],
            "SM": [0.02],
            "SH": [0.03],
            "BL": [0.04],
            "BM": [0.05],
            "BH": [0.06],
        },
        index=[pd.Timestamp("2020-07-01")],
    )

    result = construct_smb_hml(portfolio_returns)

    expected_smb = np.mean([0.03, 0.02, 0.01]) - np.mean([0.06, 0.05, 0.04])
    expected_hml = np.mean([0.03, 0.06]) - np.mean([0.01, 0.04])

    assert result.loc[pd.Timestamp("2020-07-01"), "SMB"] == pytest.approx(expected_smb)
    assert result.loc[pd.Timestamp("2020-07-01"), "HML"] == pytest.approx(expected_hml)
