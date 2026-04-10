"""Bayesian hyperparameter tuning with Optuna.

Provides ready-to-use tuning functions for the ML models in this project:

- ``tune_sklearn_model``: Optuna-based tuning for any sklearn estimator
  (AdaBoost, Random Forest, Ridge, etc.) with temporal cross-validation.
- ``tune_keras_nn``: Optuna-based tuning for Keras neural networks with
  temporal cross-validation, handling session cleanup and early stopping.
- ``make_param_space``: convenience builder for Optuna search spaces from
  a declarative dict spec.

Why Optuna over GridSearchCV:
- **Sample-efficient**: TPE sampler explores promising regions automatically
  (same idea as manual coarse-to-fine, but principled and adaptive).
- **Early pruning**: MedianPruner kills unpromising trials before they finish.
- **Mixed types**: handles int, float (linear/log), categorical natively.
- **No manual grid definition**: just set bounds, let the algorithm search.
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Callable

import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


def tune_sklearn_model(
    estimator_fn: Callable[[optuna.Trial], Any],
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 100,
    cv: int | Any = 5,
    scoring: str = "neg_mean_squared_error",
    n_jobs_cv: int = 1,
    random_state: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Tune a scikit-learn estimator using Optuna.

    Parameters
    ----------
    estimator_fn : callable
        Function that takes an ``optuna.Trial`` and returns a configured
        estimator instance.  Use ``trial.suggest_*`` inside this function
        to define the search space.
    X, y : array-like
        Training data.
    n_trials : int
        Number of Optuna trials (default 100).
    cv : int or CV splitter
        Cross-validation strategy.  If int, uses ``TimeSeriesSplit(n_splits=cv)``.
    scoring : str
        sklearn scoring metric (higher is better).
    n_jobs_cv : int
        Parallelism *within* each CV call.  Optuna parallelism should be
        controlled via ``n_trials`` and study-level settings.
    random_state : int
        Seed for the TPE sampler.
    verbose : bool
        Print progress summary.

    Returns
    -------
    dict with keys:
        ``best_params``, ``best_score``, ``best_estimator`` (refit on full
        training data), ``study`` (the Optuna study object).

    Example
    -------
    >>> def objective_fn(trial):
    ...     return AdaBoostRegressor(
    ...         estimator=DecisionTreeRegressor(
    ...             max_depth=trial.suggest_int("max_depth", 1, 6),
    ...         ),
    ...         n_estimators=trial.suggest_int("n_estimators", 50, 500, log=True),
    ...         learning_rate=trial.suggest_float("learning_rate", 1e-4, 2.0, log=True),
    ...         random_state=42,
    ...     )
    >>> result = tune_sklearn_model(objective_fn, X_train, y_train, n_trials=80)
    """
    if isinstance(cv, int):
        cv = TimeSeriesSplit(n_splits=cv)

    def objective(trial: optuna.Trial) -> float:
        model = estimator_fn(trial)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs_cv)
        return scores.mean()

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    if verbose:
        print(f"  Optuna: {len(study.trials)} trials completed")
        print(f"  Best CV score: {study.best_value:.6f}")
        print(f"  Best params:   {study.best_params}")

    # Refit best model on full training data
    best_model = estimator_fn(study.best_trial)
    best_model.fit(X, y)

    return {
        "best_params": study.best_params,
        "best_score": study.best_value,
        "best_estimator": best_model,
        "study": study,
    }


def tune_keras_nn(
    build_fn: Callable[[optuna.Trial], Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 50,
    cv: int = 5,
    epochs: int = 100,
    batch_size: int = 512,
    patience: int = 10,
    random_state: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Tune a Keras model using Optuna with temporal cross-validation.

    Parameters
    ----------
    build_fn : callable
        Function ``(trial) -> compiled keras.Model``.  Use
        ``trial.suggest_*`` to define the HP search space.
    X_train, y_train : ndarray
        Training data (already scaled).
    n_trials : int
        Number of Optuna trials.
    cv : int
        Number of ``TimeSeriesSplit`` folds.
    epochs : int
        Max training epochs per fold (early stopping will cut short).
    batch_size : int
        Mini-batch size.
    patience : int
        Early-stopping patience.
    random_state : int
        Seed for the TPE sampler.
    verbose : bool
        Print progress summary.

    Returns
    -------
    dict with keys:
        ``best_params``, ``best_score`` (mean val loss across folds),
        ``study``.
    """
    import tensorflow as tf

    tscv = TimeSeriesSplit(n_splits=cv)

    def objective(trial: optuna.Trial) -> float:
        cv_losses = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr = y_train[train_idx].astype(np.float32)
            y_val = y_train[val_idx].astype(np.float32)

            with tf.device("/CPU:0"):
                model = build_fn(trial)
                cb = tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    restore_best_weights=True,
                )
                history = model.fit(
                    X_tr,
                    y_tr,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val),
                    callbacks=[cb],
                    verbose=0,
                )
                best_val_loss = min(history.history["val_loss"])
                cv_losses.append(best_val_loss)

            tf.keras.backend.clear_session()
            del model, history
            gc.collect()

        return np.mean(cv_losses)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    if verbose:
        print(f"  Optuna: {len(study.trials)} trials completed")
        print(f"  Best CV val loss: {study.best_value:.6f}")
        print(f"  Best params:      {study.best_params}")

    return {
        "best_params": study.best_params,
        "best_score": study.best_value,
        "study": study,
    }
