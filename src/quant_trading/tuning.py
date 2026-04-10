"""Iterative coarse-to-fine hyperparameter tuning.

Implements the successive refinement pattern:
1. Start with a broad parameter grid.
2. Run cross-validated grid search.
3. Build a finer grid centered on the best parameters found.
4. Repeat until convergence (score plateau, parameter stability, or max iterations).

Works with any scikit-learn estimator/pipeline via ``GridSearchCV``.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

logger = logging.getLogger(__name__)


def _build_fine_grid(
    best_params: dict[str, Any],
    param_specs: dict[str, dict],
) -> dict[str, list]:
    """Build the next finer grid centered on *best_params*.

    Parameters
    ----------
    best_params : dict
        Best parameters from the previous round.
    param_specs : dict
        For each parameter name, a dict with:
        - ``type``: ``"log"`` (geometric steps) or ``"linear"`` (arithmetic)
          or ``"int_linear"`` (integer arithmetic).
        - ``factor``: for log-scale, how much to multiply/divide by (e.g. 2.0).
        - ``step``: for linear/int_linear, step size around best value.
        - ``points``: number of points to generate (default 5).
        - ``min``, ``max``: optional hard bounds.

    Returns
    -------
    dict[str, list]
        Parameter grid suitable for ``GridSearchCV``.
    """
    fine_grid: dict[str, list] = {}

    for param_name, spec in param_specs.items():
        best_val = best_params[param_name]
        ptype = spec.get("type", "log")
        n_points = spec.get("points", 5)
        lo_bound = spec.get("min", None)
        hi_bound = spec.get("max", None)

        if ptype == "log":
            factor = spec.get("factor", 2.0)
            lo = best_val / factor
            hi = best_val * factor
            if lo_bound is not None:
                lo = max(lo, lo_bound)
            if hi_bound is not None:
                hi = min(hi, hi_bound)
            candidates = np.geomspace(lo, hi, n_points).tolist()

        elif ptype == "linear":
            step = spec.get("step", best_val * 0.25 if best_val != 0 else 0.1)
            half_range = step * (n_points // 2)
            lo = best_val - half_range
            hi = best_val + half_range
            if lo_bound is not None:
                lo = max(lo, lo_bound)
            if hi_bound is not None:
                hi = min(hi, hi_bound)
            candidates = np.linspace(lo, hi, n_points).tolist()

        elif ptype == "int_linear":
            step = spec.get("step", max(1, int(best_val * 0.2)))
            half_range = step * (n_points // 2)
            lo = int(best_val - half_range)
            hi = int(best_val + half_range)
            if lo_bound is not None:
                lo = max(lo, int(lo_bound))
            if hi_bound is not None:
                hi = min(hi, int(hi_bound))
            candidates = list(range(lo, hi + 1, step))
            if best_val not in candidates:
                candidates.append(int(best_val))
                candidates.sort()

        else:
            raise ValueError(f"Unknown param type '{ptype}' for '{param_name}'")

        fine_grid[param_name] = sorted(set(candidates))

    return fine_grid


def iterative_grid_search(
    estimator,
    initial_grid: dict[str, list],
    param_specs: dict[str, dict],
    X,
    y,
    cv: int | Any = 5,
    scoring: str = "neg_mean_squared_error",
    max_rounds: int = 5,
    min_score_improvement: float = 1e-6,
    n_jobs: int = -1,
    verbose: int = 1,
) -> dict[str, Any]:
    """Run iterative coarse-to-fine grid search.

    Parameters
    ----------
    estimator
        scikit-learn estimator or pipeline.
    initial_grid : dict
        Starting parameter grid (broad search space).
    param_specs : dict
        Refinement specs for each parameter (see ``_build_fine_grid``).
    X, y
        Training data.
    cv : int or CV splitter
        Cross-validation strategy. Defaults to 5-fold ``TimeSeriesSplit``.
    scoring : str
        Scoring metric (must be a "higher is better" metric for sklearn).
    max_rounds : int
        Maximum number of refinement rounds.
    min_score_improvement : float
        Stop if the best CV score improves by less than this between rounds.
    n_jobs : int
        Parallelism for GridSearchCV.
    verbose : int
        0 = silent, 1 = summary per round, 2 = detailed.

    Returns
    -------
    dict with keys:
        ``best_estimator``, ``best_params``, ``best_score``,
        ``rounds``, ``history`` (list of per-round results).
    """
    if isinstance(cv, int):
        cv = TimeSeriesSplit(n_splits=cv)

    current_grid = deepcopy(initial_grid)
    best_score_overall = -np.inf
    best_params_overall = None
    best_estimator_overall = None
    history: list[dict] = []

    for round_num in range(1, max_rounds + 1):
        grid_size = 1
        for v in current_grid.values():
            grid_size *= len(v)

        if verbose >= 1:
            print(
                f"  Round {round_num}/{max_rounds}: "
                f"{grid_size} combos, grid={_grid_summary(current_grid)}"
            )

        search = GridSearchCV(
            deepcopy(estimator),
            current_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=True,
        )
        search.fit(X, y)

        round_best_score = search.best_score_
        round_best_params = search.best_params_

        if verbose >= 1:
            print(f"         Best score: {round_best_score:.6f}, params: {round_best_params}")

        history.append(
            {
                "round": round_num,
                "grid": deepcopy(current_grid),
                "best_score": round_best_score,
                "best_params": round_best_params,
            }
        )

        improvement = round_best_score - best_score_overall
        if round_best_score > best_score_overall:
            best_score_overall = round_best_score
            best_params_overall = round_best_params
            best_estimator_overall = search.best_estimator_

        # Check convergence
        if round_num > 1 and improvement < min_score_improvement:
            if verbose >= 1:
                print(
                    f"  Converged: improvement={improvement:.2e} "
                    f"< threshold={min_score_improvement:.2e}"
                )
            break

        if round_num < max_rounds:
            current_grid = _build_fine_grid(round_best_params, param_specs)

    return {
        "best_estimator": best_estimator_overall,
        "best_params": best_params_overall,
        "best_score": best_score_overall,
        "rounds": len(history),
        "history": history,
    }


def _grid_summary(grid: dict[str, list]) -> str:
    parts = []
    for k, v in grid.items():
        short_key = k.split("__")[-1] if "__" in k else k
        if len(v) <= 3:
            parts.append(f"{short_key}={v}")
        else:
            parts.append(f"{short_key}=[{v[0]}..{v[-1]}]({len(v)})")
    return ", ".join(parts)
