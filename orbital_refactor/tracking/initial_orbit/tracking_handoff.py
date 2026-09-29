from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from interfaces.data_objects import InitialState, ModuleInput, Observation
from orbital_core.dynamics import (
    make_process_noise,
    numerical_jacobian_discrete,
    rk4_step_absolute,
)

from tracking.data_contracts import TargetInitialState


@dataclass(frozen=True)
class IODTrackingHandoff:
    start_timestamp: float
    target_initial_states: Mapping[str, TargetInitialState]
    module_inputs: tuple[ModuleInput, ...]


def build_tracking_handoff(
    *,
    initialized_states: Mapping[str, TargetInitialState],
    timestamps,
    observer_state_history_by_id: Mapping[str, np.ndarray],
    q_eci2pri_history_by_id: Mapping[str, np.ndarray],
    observations_by_link: Mapping[tuple[str, str], list[Observation]],
    process_noise_acceleration_std: float = 1e-4,
) -> IODTrackingHandoff:
    """Build local continuous-filter tasks after successful known-target IOD."""

    times = np.asarray(timestamps, dtype=float).reshape(-1)
    if times.size < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("Tracking handoff requires at least two increasing epochs.")
    start = float(times[0])
    observers = set(observer_state_history_by_id)
    if observers != set(q_eci2pri_history_by_id):
        raise ValueError("Observer state and attitude histories must use the same IDs.")
    propagated_targets = {
        target_id: _propagate_initial_state(
            state, start,
            process_noise_acceleration_std=process_noise_acceleration_std,
        )
        for target_id, state in initialized_states.items()
    }
    nominal_dt = float(np.median(np.diff(times)))
    process_noise = make_process_noise(nominal_dt, process_noise_acceleration_std)
    module_inputs = []
    expected_links = {
        (observer_id, target_id)
        for observer_id in observers
        for target_id in propagated_targets
    }
    if set(observations_by_link) != expected_links:
        missing = expected_links - set(observations_by_link)
        extra = set(observations_by_link) - expected_links
        raise ValueError(f"Tracking observation links mismatch; missing={missing}, extra={extra}.")
    for observer_id in sorted(observers):
        observer_history = np.asarray(
            observer_state_history_by_id[observer_id], dtype=float,
        )
        quaternion_history = np.asarray(
            q_eci2pri_history_by_id[observer_id], dtype=float,
        )
        if observer_history.shape != (len(times), 6):
            raise ValueError(f"Observer {observer_id!r} state history must have shape (N, 6).")
        if quaternion_history.shape != (len(times), 4):
            raise ValueError(f"Observer {observer_id!r} attitude history must have shape (N, 4).")
        for target_id, target in propagated_targets.items():
            observations = list(observations_by_link[(observer_id, target_id)])
            if any(item.observer_id != observer_id for item in observations):
                raise ValueError("Tracking observation observer_id does not match its link.")
            if any(item.target_id != target_id for item in observations):
                raise ValueError("Tracking observation target_id does not match its link.")
            module_inputs.append(ModuleInput(
                initial_state=InitialState(
                    target_id=target_id,
                    timestamp=start,
                    state_estimate=target.state_eci - observer_history[0],
                    covariance=target.covariance_eci.copy(),
                ),
                sensor_measurements=observations,
                config={
                    "runtime": {
                        "timestamps": times.copy(),
                        "chief_state_history_eci": observer_history.copy(),
                        "q_eci2pri_history": quaternion_history.copy(),
                        "node_id": observer_id,
                    },
                    "filter": {
                        "architecture": "federated_ci",
                        "process_noise": process_noise.copy(),
                        "reset_feedback": True,
                        "ci_objective": "trace",
                        "ci_grid_points": 31,
                    },
                    "modalities": {},
                },
            ))
    return IODTrackingHandoff(
        start_timestamp=start,
        target_initial_states=propagated_targets,
        module_inputs=tuple(module_inputs),
    )


def _propagate_initial_state(
    value: TargetInitialState,
    timestamp: float,
    *,
    process_noise_acceleration_std: float,
) -> TargetInitialState:
    delta = float(timestamp - value.timestamp)
    if delta < 0.0:
        raise ValueError("Continuous tracking cannot begin before the IOD reference epoch.")
    if delta == 0.0:
        return value
    propagate = lambda state: rk4_step_absolute(state, delta)
    transition = numerical_jacobian_discrete(propagate, value.state_eci)
    covariance = (
        transition @ value.covariance_eci @ transition.T
        + make_process_noise(delta, process_noise_acceleration_std)
    )
    covariance = 0.5 * (covariance + covariance.T)
    return TargetInitialState(
        target_id=value.target_id,
        track_id=value.track_id,
        timestamp=timestamp,
        state_eci=propagate(value.state_eci),
        covariance_eci=covariance,
    )
