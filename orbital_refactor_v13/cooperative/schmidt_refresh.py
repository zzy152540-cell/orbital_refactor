from __future__ import annotations

from dataclasses import replace

import numpy as np

from cooperative.multi_neighbor_schmidt import MultiNeighborSchmidtState

Array = np.ndarray


def exact_transport_eligibility(
    state: MultiNeighborSchmidtState, *, neighbor_id: str,
    reference_covariance: Array, reference_mean: Array,
    covariance_rtol: float = 1e-8, covariance_atol: float = 1e-10,
) -> tuple[bool, str]:
    """Check that sender and receiver really share the advertised baseline."""
    if not np.allclose(
        state.neighbor_covariance(neighbor_id),
        np.asarray(reference_covariance, dtype=float).reshape(6, 6),
        rtol=covariance_rtol, atol=covariance_atol,
    ):
        return False, "reference_covariance_mismatch"
    if not np.allclose(
        state.neighbor_state_by_id[str(neighbor_id)],
        np.asarray(reference_mean, dtype=float).reshape(6),
        rtol=1e-10, atol=1e-7,
    ):
        return False, "reference_mean_mismatch"
    return True, "eligible"


def refresh_consider_neighbor(
    state: MultiNeighborSchmidtState,
    *,
    neighbor_id: str,
    neighbor_state: Array,
    neighbor_covariance: Array | None = None,
    mode: str = "safe_rescale",
    error_transition: Array | None = None,
    independent_process_noise: Array | None = None,
    correlation_margin: float = 1e-10,
) -> MultiNeighborSchmidtState:
    """Refresh one consider block without silently breaking joint PSD.

    ``safe_rescale`` keeps the old normalized cross-correlation (clamped to
    the PSD-admissible unit ball), ``zero_cross`` deliberately discards it,
    and ``exact_transport`` applies e_new = T e_old + w with w independent.
    """
    neighbor_id = str(neighbor_id)
    block = state.neighbor_slice(neighbor_id)
    new_mean = np.asarray(neighbor_state, dtype=float).reshape(6).copy()
    if mode == "propagate_only":
        return state
    if mode not in {"safe_rescale", "zero_cross", "exact_transport"}:
        raise ValueError("Unknown consider refresh mode.")

    keep = np.ones(state.dimension, dtype=bool)
    keep[block] = False
    rest = np.flatnonzero(keep)
    target = np.arange(block.start, block.stop)
    old = state.joint_covariance
    a = old[np.ix_(rest, rest)]
    c = old[np.ix_(rest, target)]
    b_old = old[np.ix_(target, target)]

    if mode == "exact_transport":
        if error_transition is None:
            raise ValueError("exact_transport requires error_transition.")
        transition = np.asarray(error_transition, dtype=float).reshape(6, 6)
        noise = (np.zeros((6, 6)) if independent_process_noise is None
                 else np.asarray(independent_process_noise, dtype=float).reshape(6, 6))
        b_new = transition @ b_old @ transition.T + noise
        c_new = c @ transition.T
    else:
        if neighbor_covariance is None:
            raise ValueError(f"{mode} requires neighbor_covariance.")
        b_new = _validate_psd(neighbor_covariance, "neighbor_covariance")
        if mode == "zero_cross":
            c_new = np.zeros_like(c)
        else:
            a_sqrt, a_inv_sqrt = _psd_sqrt_and_inverse(a)
            b_old_sqrt, b_old_inv_sqrt = _psd_sqrt_and_inverse(b_old)
            b_new_sqrt, _ = _psd_sqrt_and_inverse(b_new)
            normalized = a_inv_sqrt @ c @ b_old_inv_sqrt
            u, singular, vt = np.linalg.svd(normalized, full_matrices=False)
            limit = 1.0 - float(correlation_margin)
            normalized = (u * np.minimum(singular, limit)) @ vt
            c_new = a_sqrt @ normalized @ b_new_sqrt

    covariance = np.zeros_like(old)
    covariance[np.ix_(rest, rest)] = a
    covariance[np.ix_(target, target)] = b_new
    covariance[np.ix_(rest, target)] = c_new
    covariance[np.ix_(target, rest)] = c_new.T
    covariance = 0.5 * (covariance + covariance.T)
    if np.min(np.linalg.eigvalsh(covariance)) < -1e-7:
        raise ValueError("Refreshed joint covariance is not positive semidefinite.")
    means = {key: value.copy() for key, value in state.neighbor_state_by_id.items()}
    means[neighbor_id] = new_mean
    return replace(state, neighbor_state_by_id=means, joint_covariance=covariance)


def _validate_psd(matrix: Array, name: str) -> Array:
    result = np.asarray(matrix, dtype=float).reshape(6, 6)
    result = 0.5 * (result + result.T)
    if np.min(np.linalg.eigvalsh(result)) < -1e-9:
        raise ValueError(f"{name} must be positive semidefinite.")
    return result


def _psd_sqrt_and_inverse(matrix: Array) -> tuple[Array, Array]:
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    if values.min() < -1e-8:
        raise ValueError("Covariance block must be positive semidefinite.")
    values = np.maximum(values, 0.0)
    scale = max(float(values.max()), 1.0)
    inverse_values = np.where(values > 1e-12 * scale, 1.0 / np.sqrt(values), 0.0)
    square_root = (vectors * np.sqrt(values)) @ vectors.T
    inverse_square_root = (vectors * inverse_values) @ vectors.T
    return square_root, inverse_square_root
