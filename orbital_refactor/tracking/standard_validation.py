from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from adapters.multimodal_sensor_simulator import (
    MultimodalSensorSimulationConfig,
    single_spri_observation_from_message,
    simulate_multimodal_sensor_history,
)
from cooperative.multi_sat_pipeline import build_module_inputs
from interfaces.data_objects import ModuleInput
from orbital_core.coordinates import dcm_to_quat_wxyz
from scenarios.multi_satellite_scenario import (
    CooperativeScenario,
    generate_cooperative_scenario,
)

from .data_contracts import MultiTargetInput, TargetInitialState
from .sequence_runner import KnownTargetSequenceHistory, run_known_target_sequence


Array = np.ndarray
DropoutKey = tuple[str, str, str]


@dataclass(frozen=True)
class StandardMultiTargetScenario:
    """Inputs and truth for a known-target multi-observer validation run."""

    scene_id: str
    timestamps: Array
    scenarios_by_target: Mapping[str, CooperativeScenario]
    module_inputs: tuple[ModuleInput, ...]
    initial_target_states: Mapping[str, TargetInitialState]


@dataclass(frozen=True)
class TargetValidationMetrics:
    target_id: str
    position_rmse_m: float
    velocity_rmse_mps: float
    max_position_error_m: float
    max_velocity_error_mps: float
    mean_nees: float
    mean_position_sigma_m: float
    mean_active_observers: float
    minimum_active_observers: int
    lifecycle_counts: Mapping[str, int]
    reacquisition_count: int


@dataclass(frozen=True)
class StandardValidationResult:
    scenario: StandardMultiTargetScenario
    history: KnownTargetSequenceHistory
    metrics_by_target: Mapping[str, TargetValidationMetrics]


def build_standard_multitarget_scenario(
    *,
    scene_id: str,
    timestamps: Array,
    observer_initial_states_eci: Mapping[str, Array],
    target_initial_states_eci: Mapping[str, Array],
    initial_error_by_link: Mapping[tuple[str, str], Array] | None = None,
    initial_covariance: Array | None = None,
    initial_covariance_by_target: Mapping[str, Array] | None = None,
    track_id_by_target: Mapping[str, str] | None = None,
    sensor_config: MultimodalSensorSimulationConfig | None = None,
    dropout_windows: Mapping[DropoutKey, tuple[tuple[float, float], ...]] | None = None,
    random_seed: int = 0,
) -> StandardMultiTargetScenario:
    """Build target-pointing raw RADAR/INFRARED/OPTICAL local filter tasks.

    ``dropout_windows`` keys are ``(observer_id, target_id, modality)`` and
    modality names use the public names RADAR, INFRARED, or OPTICAL.
    Initial relative states are derived from supplied absolute J2000 states;
    this is scenario initialization, not an initial-orbit-determination method.
    """

    times = np.asarray(timestamps, dtype=float).reshape(-1)
    if times.size < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("timestamps must contain at least two increasing values.")
    if not observer_initial_states_eci or not target_initial_states_eci:
        raise ValueError("At least one observer and one target are required.")
    if set(observer_initial_states_eci) & set(target_initial_states_eci):
        raise ValueError("Observer and target identifiers must be disjoint.")

    errors = dict(initial_error_by_link or {})
    covariance_by_target = dict(initial_covariance_by_target or {})
    track_ids = dict(track_id_by_target or {})
    dropouts = _normalize_dropout_windows(dropout_windows or {})
    scenarios = {}
    tasks = []
    initial_targets = {}
    target_items = list(target_initial_states_eci.items())
    observer_ids = tuple(observer_initial_states_eci)
    for target_index, (target_id, target_state) in enumerate(target_items):
        scenario = generate_cooperative_scenario(
            timestamps=times,
            target_id=target_id,
            target_initial_state_eci=target_state,
            observer_initial_states_eci=dict(observer_initial_states_eci),
        )
        scenarios[target_id] = scenario
        observations_by_node = {}
        for observer_index, observer_id in enumerate(observer_ids):
            relative_spri = scenario.relative_state_spri_by_node[observer_id]
            camera_attitude = np.asarray([
                dcm_to_quat_wxyz(_tracking_camera_basis(relative[:3]).T)
                for relative in relative_spri
            ])
            validity = {
                modality: _validity_history(
                    times,
                    dropouts.get((observer_id, target_id, modality), ()),
                )
                for modality in ("RADAR", "INFRARED", "OPTICAL")
            }
            raw = simulate_multimodal_sensor_history(
                timestamps=times,
                observer_id=observer_id,
                target_id=target_id,
                observer_state_history=np.zeros_like(relative_spri),
                target_state_history=relative_spri,
                quaternion_i2b_wxyz_history_by_modality={
                    "OPTICAL": camera_attitude,
                    "INFRARED": camera_attitude,
                },
                config=sensor_config,
                random_seed=(
                    int(random_seed) + 10_000 * target_index + 100 * observer_index
                ),
                valid_history_by_modality=validity,
            )
            observations_by_node[observer_id] = [
                single_spri_observation_from_message(message)
                for message in raw.messages
            ]
        per_node_errors = {
            observer_id: np.asarray(
                errors.get((observer_id, target_id), np.zeros(6)), dtype=float,
            )
            for observer_id in observer_ids
        }
        target_covariance = covariance_by_target.get(target_id, initial_covariance)
        tasks.extend(build_module_inputs(
            scenario=scenario,
            observations_by_node=observations_by_node,
            initial_error_by_node=per_node_errors,
            initial_covariance=target_covariance,
        ).values())
        initial_targets[target_id] = TargetInitialState(
            target_id=target_id,
            track_id=track_ids.get(target_id, f"track-{target_id}"),
            timestamp=float(times[0]),
            state_eci=np.asarray(target_state, dtype=float),
            covariance_eci=(
                np.diag([100.0, 100.0, 100.0, 0.2, 0.2, 0.2]) ** 2
                if target_covariance is None
                else np.asarray(target_covariance, dtype=float)
            ),
        )
    frozen_times = times.copy()
    frozen_times.setflags(write=False)
    return StandardMultiTargetScenario(
        scene_id=str(scene_id),
        timestamps=frozen_times,
        scenarios_by_target=scenarios,
        module_inputs=tuple(tasks),
        initial_target_states=initial_targets,
    )


def build_standard_multitarget_scenario_from_input(
    value: MultiTargetInput,
    *,
    sensor_config: MultimodalSensorSimulationConfig | None = None,
    dropout_windows: Mapping[DropoutKey, tuple[tuple[float, float], ...]] | None = None,
    random_seed: int = 0,
) -> StandardMultiTargetScenario:
    """Bridge one external role-assigned input into local filter tasks."""

    timestamps = np.asarray(value.config.get("timestamps"), dtype=float).reshape(-1)
    if timestamps.size < 2:
        raise ValueError("MultiTargetInput.config.timestamps must contain at least two epochs.")
    combined_dropouts = dict(dropout_windows or {})
    availability = value.config.get("sensorAvailabilityByTarget", {})
    for target_id, modality_flags in availability.items():
        for modality, enabled in modality_flags.items():
            if not bool(enabled):
                for observer_id in value.observer_states:
                    combined_dropouts[(observer_id, target_id, str(modality).upper())] = (
                        (float(timestamps[0]), float(timestamps[-1])),
                    )
    return build_standard_multitarget_scenario(
        scene_id=value.scene_id,
        timestamps=timestamps,
        observer_initial_states_eci={
            key: state.state_eci for key, state in value.observer_states.items()
        },
        target_initial_states_eci={
            key: state.state_eci for key, state in value.target_initial_states.items()
        },
        initial_covariance_by_target={
            key: state.covariance_eci
            for key, state in value.target_initial_states.items()
        },
        track_id_by_target={
            key: state.track_id for key, state in value.target_initial_states.items()
        },
        sensor_config=sensor_config,
        dropout_windows=combined_dropouts,
        random_seed=random_seed,
    )


def run_standard_multitarget_validation(
    scenario: StandardMultiTargetScenario,
    *,
    max_coast_epochs: int = 3,
) -> StandardValidationResult:
    history = run_known_target_sequence(
        scene_id=scenario.scene_id,
        module_inputs=scenario.module_inputs,
        initial_target_states=scenario.initial_target_states,
        max_coast_epochs=max_coast_epochs,
    )
    metrics = {
        target_id: _target_metrics(
            target_id,
            scenario.scenarios_by_target[target_id].target_trajectory.state_history_eci,
            history,
        )
        for target_id in scenario.scenarios_by_target
    }
    return StandardValidationResult(scenario, history, metrics)


def _target_metrics(target_id, truth, history):
    tracks = history.track_history_by_target[target_id]
    estimates = np.asarray([track.estimate.state_eci for track in tracks])
    covariances = np.asarray([track.estimate.covariance_eci for track in tracks])
    error = estimates - np.asarray(truth, dtype=float)
    position_norm = np.linalg.norm(error[:, :3], axis=1)
    velocity_norm = np.linalg.norm(error[:, 3:], axis=1)
    nees = np.asarray([
        float(delta @ np.linalg.pinv(covariance) @ delta)
        for delta, covariance in zip(error, covariances)
    ])
    active_counts = np.asarray([
        len(output.estimates_by_target[target_id].contributing_observer_ids)
        if target_id in output.estimates_by_target else 0
        for output in history.output_by_epoch
    ])
    lifecycles = [track.lifecycle.value for track in tracks]
    reacquisitions = sum(
        previous in {"COASTING", "LOST"} and current == "TRACKING"
        for previous, current in zip(lifecycles, lifecycles[1:])
    )
    return TargetValidationMetrics(
        target_id=target_id,
        position_rmse_m=float(np.sqrt(np.mean(position_norm ** 2))),
        velocity_rmse_mps=float(np.sqrt(np.mean(velocity_norm ** 2))),
        max_position_error_m=float(np.max(position_norm)),
        max_velocity_error_mps=float(np.max(velocity_norm)),
        mean_nees=float(np.mean(nees)),
        mean_position_sigma_m=float(np.mean(np.sqrt(
            np.maximum(np.trace(covariances[:, :3, :3], axis1=1, axis2=2), 0.0)
        ))),
        mean_active_observers=float(np.mean(active_counts)),
        minimum_active_observers=int(np.min(active_counts)),
        lifecycle_counts=dict(Counter(lifecycles)),
        reacquisition_count=int(reacquisitions),
    )


def _tracking_camera_basis(boresight):
    forward = np.array(boresight, dtype=float, copy=True).reshape(3)
    norm = np.linalg.norm(forward)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Camera boresight must be finite and nonzero.")
    forward /= norm
    reference = np.array([0.0, 0.0, 1.0])
    lateral = np.cross(reference, forward)
    if np.linalg.norm(lateral) < 1e-8:
        lateral = np.cross(np.array([0.0, 1.0, 0.0]), forward)
    lateral /= np.linalg.norm(lateral)
    vertical = np.cross(forward, lateral)
    return np.column_stack((forward, lateral, vertical))


def _normalize_dropout_windows(windows):
    result = {}
    for raw_key, raw_windows in windows.items():
        if len(raw_key) != 3:
            raise ValueError("Dropout keys must be (observer_id, target_id, modality).")
        observer_id, target_id, modality = map(str, raw_key)
        modality = modality.upper()
        if modality not in {"RADAR", "INFRARED", "OPTICAL"}:
            raise ValueError(f"Unsupported dropout modality {modality!r}.")
        normalized = []
        for start, end in raw_windows:
            if float(end) < float(start):
                raise ValueError("Dropout window end must not precede start.")
            normalized.append((float(start), float(end)))
        result[(observer_id, target_id, modality)] = tuple(normalized)
    return result


def _validity_history(timestamps, windows):
    valid = np.ones(len(timestamps), dtype=bool)
    for start, end in windows:
        valid[(timestamps >= start) & (timestamps <= end)] = False
    return valid
