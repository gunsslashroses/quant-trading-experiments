"""Tests for iterative coarse-to-fine hyperparameter tuning."""

import numpy as np
from sklearn.linear_model import Ridge

from quant_trading.tuning import _build_fine_grid, iterative_grid_search


class TestBuildFineGrid:
    def test_log_scale(self):
        best = {"alpha": 1.0}
        specs = {"alpha": {"type": "log", "factor": 2.0, "points": 5, "min": 0.01}}
        grid = _build_fine_grid(best, specs)
        assert "alpha" in grid
        vals = grid["alpha"]
        assert len(vals) == 5
        assert min(vals) >= 0.01
        assert vals[0] < 1.0 < vals[-1]

    def test_int_linear(self):
        best = {"n_estimators": 100}
        specs = {"n_estimators": {"type": "int_linear", "step": 20, "points": 5, "min": 10}}
        grid = _build_fine_grid(best, specs)
        vals = grid["n_estimators"]
        assert all(isinstance(v, (int, np.integer)) for v in vals)
        assert 100 in vals

    def test_respects_bounds(self):
        best = {"alpha": 0.001}
        specs = {"alpha": {"type": "log", "factor": 10.0, "min": 0.0005, "max": 1.0}}
        grid = _build_fine_grid(best, specs)
        assert min(grid["alpha"]) >= 0.0005
        assert max(grid["alpha"]) <= 1.0


class TestIterativeGridSearch:
    def test_converges_on_ridge(self):
        rng = np.random.default_rng(42)
        n = 500
        X = rng.normal(0, 1, (n, 5))
        y = X @ rng.normal(0, 1, 5) + rng.normal(0, 0.1, n)

        initial_grid = {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
        param_specs = {
            "alpha": {"type": "log", "factor": 3.0, "points": 5, "min": 1e-6, "max": 1000}
        }

        result = iterative_grid_search(
            Ridge(),
            initial_grid=initial_grid,
            param_specs=param_specs,
            X=X,
            y=y,
            cv=3,
            max_rounds=4,
            verbose=0,
        )

        assert result["best_estimator"] is not None
        assert result["best_params"]["alpha"] > 0
        assert result["rounds"] >= 1
        assert len(result["history"]) >= 1
        assert result["best_score"] < 0  # neg MSE

    def test_stops_on_convergence(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 3))
        y = X[:, 0] + rng.normal(0, 0.01, 200)

        result = iterative_grid_search(
            Ridge(),
            initial_grid={"alpha": [0.01, 0.1, 1.0]},
            param_specs={"alpha": {"type": "log", "factor": 2.0, "points": 3, "min": 1e-6}},
            X=X,
            y=y,
            cv=3,
            max_rounds=10,
            min_score_improvement=1e-8,
            verbose=0,
        )
        # Should converge well before 10 rounds
        assert result["rounds"] < 10
