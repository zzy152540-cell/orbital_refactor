from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from cooperative.cooperative_update import CooperativeUpdateResult, update_local_state
from interfaces.data_objects import ObservationMessage, StateMessage, TargetEstimate
from orbital_core.dynamics import (
    make_process_noise,
    numerical_jacobian_discrete,
    rk4_step_absolute,
)

Array = np.ndarray


@dataclass(frozen=True)
class DelayedCooperativeUpdateResult:
    estimate: TargetEstimate
    measurement_update: CooperativeUpdateResult
    observation_timestamp: float
    output_timestamp: float
    neighbor_propagation_dt: float
    posterior_propagation_dt: float


def align_state_message(
    message: StateMessage,
    target_timestamp: float,
    *,
    process_noise_acceleration: float = 1e-4,
) -> StateMessage:
    """Propagate a received state posterior to a later measurement epoch."""

    source_timestamp = _source_timestamp(message)
    target_timestamp = float(target_timestamp)
    dt = target_timestamp - source_timestamp
    if dt < -1e-12:
        raise ValueError("State messages cannot be propagated backward in time.")
    if dt <= 1e-12:
        return replace(
            message,
            timestamp=target_timestamp,
            source_timestamp=source_timestamp,
        )
    state, covariance = propagate_state_covariance(
        message.state_estimate,
        message.covariance,
        dt,
        process_noise_acceleration=process_noise_acceleration,
    )
    return replace(
        message,
        timestamp=target_timestamp,
        state_estimate=state,
        covariance=covariance,
        source_timestamp=source_timestamp,
    )


def apply_delayed_cooperative_update(
    *,
    local_estimate_at_observation: TargetEstimate,
    neighbor_state: StateMessage,
    observation: ObservationMessage,
    output_timestamp: float,
    process_noise_acceleration: float = 1e-4,
    regularization: float = 1e-9,
    gate_enable: bool = False,
    gate_threshold: float = np.inf,
    gate_mode: str = "soft",
    soft_scale: float = 20.0,
    quaternion_i2b_wxyz: Array | None = None,
) -> DelayedCooperativeUpdateResult:
    """Update at the observation epoch, then propagate the posterior forward."""

    observation_timestamp = float(observation.timestamp)
    output_timestamp = float(output_timestamp)
    if output_timestamp < observation_timestamp:
        raise ValueError("output_timestamp cannot precede the observation.")
    if not np.isclose(
        float(local_estimate_at_observation.timestamp),
        observation_timestamp,
    ):
        raise ValueError("Local estimate must be defined at the observation timestamp.")

    aligned_neighbor = align_state_message(
        neighbor_state,
        observation_timestamp,
        process_noise_acceleration=process_noise_acceleration,
    )
    update = update_local_state(
        local_estimate=local_estimate_at_observation,
        neighbor_state=aligned_neighbor,
        observation=observation,
        regularization=regularization,
        gate_enable=gate_enable,
        gate_threshold=gate_threshold,
        gate_mode=gate_mode,
        soft_scale=soft_scale,
        quaternion_i2b_wxyz=quaternion_i2b_wxyz,
    )
    posterior_dt = output_timestamp - observation_timestamp
    estimate = update.estimate
    if posterior_dt > 1e-12:
        state, covariance = propagate_state_covariance(
            estimate.state_estimate,
            estimate.covariance,
            posterior_dt,
            process_noise_acceleration=process_noise_acceleration,
        )
        estimate = replace(
            estimate,
            timestamp=output_timestamp,
            state_estimate=state,
            covariance=covariance,
        )
    else:
        estimate = replace(estimate, timestamp=output_timestamp)
    return DelayedCooperativeUpdateResult(
        estimate=estimate,
        measurement_update=update,
        observation_timestamp=observation_timestamp,
        output_timestamp=output_timestamp,
        neighbor_propagation_dt=observation_timestamp - _source_timestamp(neighbor_state),
        posterior_propagation_dt=posterior_dt,
    )


def propagate_state_covariance(
    state: Array,
    covariance: Array,
    dt: float,
    *,
    process_noise_acceleration: float = 1e-4,
) -> tuple[Array, Array]:
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if process_noise_acceleration < 0.0:
        raise ValueError("process_noise_acceleration cannot be negative.")
    vector = np.asarray(state, dtype=float).reshape(6)
    matrix = np.asarray(covariance, dtype=float).reshape(6, 6)
    propagate = lambda value: rk4_step_absolute(value, float(dt))
    transition = numerical_jacobian_discrete(propagate, vector)
    propagated_state = propagate(vector)
    propagated_covariance = (
        transition @ matrix @ transition.T
        + make_process_noise(float(dt), float(process_noise_acceleration))
    )
    return (
        propagated_state,
        0.5 * (propagated_covariance + propagated_covariance.T),
    )


def _source_timestamp(message: StateMessage) -> float:
    return float(
        message.timestamp
        if message.source_timestamp is None
        else message.source_timestamp
    )
