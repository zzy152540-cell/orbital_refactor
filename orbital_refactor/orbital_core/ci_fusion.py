from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np


Array = np.ndarray
Posterior = tuple[str, Array, Array]


@dataclass(frozen=True)
class CIFusionResult:
    state: Array
    covariance: Array
    weights: dict[str, float]


def ci_objective(covariance: Array, mode: str = "trace") -> float:
    covariance = np.asarray(covariance, dtype=float)
    if mode == "trace":
        return float(np.trace(covariance))
    if mode == "logdet":
        sign, logdet = np.linalg.slogdet(covariance)
        return float(logdet) if sign > 0 else np.inf
    raise ValueError(f"Unsupported CI objective: {mode}")


def ci_fuse_pair(
    state_1: Array,
    covariance_1: Array,
    state_2: Array,
    covariance_2: Array,
    *,
    objective: str = "trace",
    grid_points: int = 101,
) -> tuple[Array, Array, float]:
    """Fuse two posteriors by CI using the legacy weight grid.

    The weight candidates are evaluated in one batched linear-algebra call.
    This preserves the original grid and first-minimum tie behavior while
    avoiding one SVD-based ``pinv`` per candidate weight.
    """
    if grid_points < 2:
        raise ValueError("grid_points must be at least 2.")
    _validate_objective(objective)

    state_1 = np.asarray(state_1, dtype=float)
    state_2 = np.asarray(state_2, dtype=float)
    covariance_1 = _symmetrize(np.asarray(covariance_1, dtype=float))
    covariance_2 = _symmetrize(np.asarray(covariance_2, dtype=float))

    # These two matrices are fixed during the entire weight search.
    information_1 = np.linalg.pinv(covariance_1)
    information_2 = np.linalg.pinv(covariance_2)

    weights = _pair_weight_grid(int(grid_points))
    fused_information = (
        weights[:, None, None] * information_1[None, :, :]
        + (1.0 - weights)[:, None, None] * information_2[None, :, :]
    )
    covariance_candidates = _invert_stack(fused_information)
    values = _objective_values(covariance_candidates, objective)
    best_index = int(np.argmin(values))
    if not np.isfinite(values[best_index]):
        raise RuntimeError("CI pair fusion failed to find a valid weight.")

    weight_1 = float(weights[best_index])
    covariance = _symmetrize(covariance_candidates[best_index])
    information_vector = (
        weight_1 * (information_1 @ state_1)
        + (1.0 - weight_1) * (information_2 @ state_2)
    )
    state = covariance @ information_vector
    return state, covariance, weight_1


def ci_fuse_three(
    state_1: Array,
    covariance_1: Array,
    state_2: Array,
    covariance_2: Array,
    state_3: Array,
    covariance_3: Array,
    *,
    objective: str = "trace",
    grid_points: int = 31,
) -> tuple[Array, Array, Array]:
    """Fuse three posteriors by CI using the legacy triangular grid.

    Candidate information matrices are assembled and inverted as a stack.
    The grid points and iteration order match the original nested loops, so
    the selected weight is unchanged except for insignificant floating-point
    differences between ``inv`` and ``pinv`` on nonsingular 6x6 matrices.
    """
    if grid_points < 2:
        raise ValueError("grid_points must be at least 2.")
    _validate_objective(objective)

    states = np.stack(
        [
            np.asarray(state_1, dtype=float),
            np.asarray(state_2, dtype=float),
            np.asarray(state_3, dtype=float),
        ],
        axis=0,
    )
    covariances = [
        _symmetrize(np.asarray(covariance_1, dtype=float)),
        _symmetrize(np.asarray(covariance_2, dtype=float)),
        _symmetrize(np.asarray(covariance_3, dtype=float)),
    ]

    # Local information matrices are invariant across all candidate weights.
    information = np.stack([np.linalg.pinv(covariance) for covariance in covariances])
    weights = _three_weight_grid(int(grid_points))

    # (candidate, modality) x (modality, row, column) -> (candidate, row, column)
    fused_information = np.einsum("km,mij->kij", weights, information, optimize=True)
    covariance_candidates = _invert_stack(fused_information)
    values = _objective_values(covariance_candidates, objective)
    best_index = int(np.argmin(values))
    if not np.isfinite(values[best_index]):
        raise RuntimeError("Three-way CI fusion failed to find valid weights.")

    best_weights = weights[best_index].copy()
    covariance = _symmetrize(covariance_candidates[best_index])

    # Compute the fused state only once, after the optimum has been selected.
    information_vectors = np.einsum("mij,mj->mi", information, states, optimize=True)
    fused_information_vector = best_weights @ information_vectors
    state = covariance @ fused_information_vector
    return state, covariance, best_weights


def ci_fuse_posteriors(
    posteriors: Iterable[Posterior],
    *,
    objective: str = "trace",
    grid_points: int = 31,
) -> CIFusionResult:
    items = list(posteriors)
    if not items:
        raise ValueError("CI fusion input is empty.")
    if len(items) == 1:
        name, state, covariance = items[0]
        return CIFusionResult(state.copy(), covariance.copy(), {name: 1.0})
    if len(items) == 2:
        (name_1, state_1, covariance_1), (name_2, state_2, covariance_2) = items
        state, covariance, weight_1 = ci_fuse_pair(
            state_1,
            covariance_1,
            state_2,
            covariance_2,
            objective=objective,
            grid_points=max(grid_points, 2),
        )
        return CIFusionResult(
            state,
            covariance,
            {name_1: weight_1, name_2: 1.0 - weight_1},
        )
    if len(items) == 3:
        names = [item[0] for item in items]
        state, covariance, weights = ci_fuse_three(
            items[0][1],
            items[0][2],
            items[1][1],
            items[1][2],
            items[2][1],
            items[2][2],
            objective=objective,
            grid_points=grid_points,
        )
        return CIFusionResult(
            state,
            covariance,
            {name: float(weight) for name, weight in zip(names, weights, strict=True)},
        )
    raise ValueError("Current simultaneous CI implementation supports one to three posteriors.")


@lru_cache(maxsize=16)
def _pair_weight_grid(grid_points: int) -> Array:
    weights = np.linspace(0.0, 1.0, grid_points, dtype=float)
    weights.setflags(write=False)
    return weights


@lru_cache(maxsize=16)
def _three_weight_grid(grid_points: int) -> Array:
    # Preserve the exact nested-loop grid and traversal order used previously.
    grid = np.linspace(0.0, 1.0, grid_points, dtype=float)
    candidates: list[tuple[float, float, float]] = []
    for weight_1 in grid:
        for weight_2 in grid:
            weight_3 = 1.0 - weight_1 - weight_2
            if weight_3 < 0.0:
                continue
            candidates.append((float(weight_1), float(weight_2), float(weight_3)))
    weights = np.asarray(candidates, dtype=float)
    weights.setflags(write=False)
    return weights


def _invert_stack(matrices: Array) -> Array:
    """Invert a stack of small square matrices, with a robust fallback.

    CI information matrices should be positive definite. ``np.linalg.inv`` is
    much faster than SVD-based ``pinv`` in that normal case. If a batch contains
    a singular matrix, only the fallback path uses ``pinv``.
    """
    matrices = np.asarray(matrices, dtype=float)
    try:
        return np.linalg.inv(matrices)
    except np.linalg.LinAlgError:
        return np.stack([_safe_inverse(matrix) for matrix in matrices], axis=0)


def _safe_inverse(matrix: Array) -> Array:
    matrix = _symmetrize(np.asarray(matrix, dtype=float))
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix)


def _objective_values(covariances: Array, mode: str) -> Array:
    if mode == "trace":
        return np.trace(covariances, axis1=-2, axis2=-1)
    if mode == "logdet":
        sign, logdet = np.linalg.slogdet(covariances)
        return np.where(sign > 0, logdet, np.inf)
    raise ValueError(f"Unsupported CI objective: {mode}")


def _validate_objective(mode: str) -> None:
    if mode not in {"trace", "logdet"}:
        raise ValueError(f"Unsupported CI objective: {mode}")


def _symmetrize(matrix: Array) -> Array:
    return 0.5 * (matrix + matrix.T)
