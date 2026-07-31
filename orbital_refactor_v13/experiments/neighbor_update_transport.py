from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cooperative.multi_neighbor_schmidt import MultiNeighborSchmidtState
from cooperative.schmidt_refresh import refresh_consider_neighbor
from cooperative.exact_transport_protocol import (
    apply_exact_transport_state_message,
    build_exact_transport_state_message,
)

Array = np.ndarray


@dataclass(frozen=True)
class NeighborUpdateTransportSummary:
    strategy: str
    sample_count: int
    position_rmse: float
    mean_nees: float
    nees_95_coverage: float
    mean_nis: float
    minimum_covariance_eigenvalue: float


def run_neighbor_update_transport_monte_carlo(
    *, samples: int = 20000, seed: int = 20260812,
    absolute_sigma: float = 2.0, relative_sigma: float = 3.0,
) -> dict[str, NeighborUpdateTransportSummary]:
    """Isolate a neighbor-only measurement update and its covariance transport.

    The sender first performs a private absolute-position Kalman update.  The
    receiver then consumes a relative-position observation using a Schmidt
    active-only gain.  All strategies see exactly the same truth/noise draws.
    """
    if samples < 1:
        raise ValueError("samples must be at least one.")
    rng = np.random.default_rng(seed)
    pa = np.diag([10.0, 9.0, 8.0, 0.5, 0.4, 0.3]) ** 2
    pb = np.diag([12.0, 11.0, 9.0, 0.6, 0.5, 0.4]) ** 2
    scales = np.sqrt(np.diag(pa) * np.diag(pb))
    cross = np.diag(0.55 * scales)
    prior = np.block([[pa, cross], [cross.T, pb]])

    h_absolute = np.zeros((3, 6)); h_absolute[:, :3] = np.eye(3)
    r_absolute = np.eye(3) * absolute_sigma**2
    s_absolute = h_absolute @ pb @ h_absolute.T + r_absolute
    k_absolute = pb @ h_absolute.T @ np.linalg.inv(s_absolute)
    update_transition = np.eye(6) - k_absolute @ h_absolute
    update_noise = k_absolute @ r_absolute @ k_absolute.T
    pb_updated = update_transition @ pb @ update_transition.T + update_noise

    errors = rng.multivariate_normal(np.zeros(12), prior, size=samples)
    active_error = errors[:, :6]
    neighbor_error = errors[:, 6:]
    absolute_noise = rng.normal(0.0, absolute_sigma, size=(samples, 3))
    neighbor_updated_error = (
        neighbor_error @ update_transition.T + absolute_noise @ k_absolute.T
    )
    relative_noise = rng.normal(0.0, relative_sigma, size=(samples, 3))

    base_state = MultiNeighborSchmidtState(
        timestamp=0.0, active_node_id="receiver", neighbor_ids=("neighbor",),
        active_state=np.zeros(6), neighbor_state_by_id={"neighbor": np.zeros(6)},
        joint_covariance=prior,
    )
    exact_message = build_exact_transport_state_message(
        source_node_id="neighbor", timestamp=1.0, reference_timestamp=0.0,
        reference_state=np.zeros(6), reference_covariance=pb,
        updated_state=np.zeros(6), error_transition=update_transition,
        independent_process_noise=update_noise, lineage_id="neighbor:0",
        information_ids=("private_absolute_position",),
    )
    exact_received = apply_exact_transport_state_message(
        base_state, exact_message, expected_lineage_id="neighbor:0"
    )
    if not exact_received.accepted:
        raise RuntimeError(f"Exact transport message rejected: {exact_received.reason}")
    strategy_states = {
        "propagate_only": base_state,
        "safe_rescale": refresh_consider_neighbor(
            base_state, neighbor_id="neighbor", neighbor_state=np.zeros(6),
            neighbor_covariance=pb_updated, mode="safe_rescale",
        ),
        "zero_cross": refresh_consider_neighbor(
            base_state, neighbor_id="neighbor", neighbor_state=np.zeros(6),
            neighbor_covariance=pb_updated, mode="zero_cross",
        ),
        "exact_transport": exact_received.state,
    }
    result = {}
    h_relative = np.zeros((3, 12))
    h_relative[:, :3] = -np.eye(3); h_relative[:, 6:9] = np.eye(3)
    r_relative = np.eye(3) * relative_sigma**2
    for strategy, state in strategy_states.items():
        covariance = state.joint_covariance
        innovation_covariance = h_relative @ covariance @ h_relative.T + r_relative
        gain = np.zeros((12, 3))
        gain[:6] = covariance[:6] @ h_relative.T @ np.linalg.inv(innovation_covariance)
        residual = np.eye(12) - gain @ h_relative
        posterior = residual @ covariance @ residual.T + gain @ r_relative @ gain.T
        active_posterior = posterior[:6, :6]
        used_neighbor_error = (
            neighbor_error if strategy == "propagate_only" else neighbor_updated_error
        )
        innovation = active_error[:, :3] - used_neighbor_error[:, :3] + relative_noise
        updated_active_error = active_error + innovation @ gain[:6].T
        nees = np.einsum(
            "ni,ij,nj->n", updated_active_error,
            np.linalg.inv(active_posterior), updated_active_error,
        )
        nis = np.einsum(
            "ni,ij,nj->n", innovation,
            np.linalg.inv(innovation_covariance), innovation,
        )
        result[strategy] = NeighborUpdateTransportSummary(
            strategy=strategy, sample_count=samples,
            position_rmse=float(np.sqrt(np.mean(updated_active_error[:, :3] ** 2))),
            mean_nees=float(np.mean(nees)),
            nees_95_coverage=float(np.mean((nees >= 1.2373442458) & (nees <= 14.4493753354))),
            mean_nis=float(np.mean(nis)),
            minimum_covariance_eigenvalue=float(np.linalg.eigvalsh(posterior).min()),
        )
    return result
