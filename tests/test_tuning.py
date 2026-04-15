"""Tests for Optuna-based hyperparameter tuning."""

import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

from quant_trading.tuning import tune_sklearn_model


class TestTuneSklearnModel:
    def _make_data(self, n=300, seed=42):
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (n, 5))
        y = X @ rng.normal(0, 1, 5) + rng.normal(0, 0.1, n)
        return X, y

    def test_ridge_finds_good_alpha(self):
        X, y = self._make_data()

        def ridge_objective(trial):
            alpha = trial.suggest_float("alpha", 1e-4, 100.0, log=True)
            return Ridge(alpha=alpha)

        result = tune_sklearn_model(
            ridge_objective,
            X,
            y,
            n_trials=20,
            cv=3,
            verbose=False,
        )
        assert result["best_params"]["alpha"] > 0
        assert result["best_estimator"] is not None
        assert result["best_score"] < 0  # neg MSE
        assert result["study"] is not None

    def test_adaboost_search(self):
        X, y = self._make_data()

        def ada_objective(trial):
            return AdaBoostRegressor(
                estimator=DecisionTreeRegressor(
                    max_depth=trial.suggest_int("max_depth", 1, 4),
                    random_state=42,
                ),
                n_estimators=trial.suggest_int("n_estimators", 10, 100, log=True),
                learning_rate=trial.suggest_float("lr", 0.01, 1.0, log=True),
                random_state=42,
            )

        result = tune_sklearn_model(
            ada_objective,
            X,
            y,
            n_trials=15,
            cv=3,
            verbose=False,
        )
        assert "max_depth" in result["best_params"]
        assert "n_estimators" in result["best_params"]
        assert "lr" in result["best_params"]
        assert result["best_estimator"] is not None

    def test_study_has_trials(self):
        X, y = self._make_data(n=100)

        def ridge_fn(trial):
            return Ridge(alpha=trial.suggest_float("alpha", 0.01, 10.0, log=True))

        result = tune_sklearn_model(
            ridge_fn,
            X,
            y,
            n_trials=10,
            cv=3,
            verbose=False,
        )
        assert len(result["study"].trials) == 10
