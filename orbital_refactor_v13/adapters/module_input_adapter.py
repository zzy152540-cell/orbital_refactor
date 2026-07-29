from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from interfaces.data_objects import ModuleInput, Observation
from orbital_core.filters import LocalDynamicsEKF


Array = np.ndarray

_MODALITY_ALIASES = {
    "OPT": "opt",
    "OPTICAL": "opt",
    "TRADITIONAL_OPTICAL": "opt",
    "NN": "nn",
    "LEARNING": "nn",
    "LEARNING_OPTICAL": "nn",
    "INFRARED": "ir",
    "IR": "ir",
    "RADAR": "rad",
    "RAD": "rad",
}


@dataclass(frozen=True)
class FederatedAdapterResult:
    timestamps: Array
    chief_state_history_eci: Array
    q_eci2pri_history: Array
    measurements_by_modality: dict[str, Array]
    valid_flags_by_modality: dict[str, Array]
    local_filters: dict[str, LocalDynamicsEKF]
    initial_state: Array
    initial_covariance: Array
    node_id: str
    target_id: str
    reset_feedback: bool
    ci_objective: str
    ci_grid_points: int


def adapt_module_input(module_input: ModuleInput) -> FederatedAdapterResult:
    """Convert the documented ModuleInput object into the legacy-compatible arrays.

    The adapter is deliberately thin: it does not alter measurements, interpolate
    data, or change the filtering algorithm. Runtime histories such as the chief
    orbit and ECI-to-SPRI quaternion sequence remain configuration inputs because
    they are auxiliary model data rather than sensor observations.
    """
    config = module_input.config
    runtime = _section(config, "runtime")
    filter_config = _section(config, "filter")

    timestamps = np.asarray(
        _required(runtime, config, "timestamps"), dtype=float
    ).reshape(-1)
    chief_history = np.asarray(
        _required(runtime, config, "chief_state_history_eci"), dtype=float
    )
    q_history = np.asarray(
        _required(runtime, config, "q_eci2pri_history"), dtype=float
    )

    if timestamps.ndim != 1 or timestamps.size == 0:
        raise ValueError("timestamps must be a non-empty one-dimensional sequence.")
    if chief_history.shape != (timestamps.size, 6):
        raise ValueError("chief_state_history_eci must have shape (N, 6).")
    if q_history.shape != (timestamps.size, 4):
        raise ValueError("q_eci2pri_history must have shape (N, 4).")
    if not np.all(np.diff(timestamps) > 0.0):
        raise ValueError("timestamps must be strictly increasing.")

    grouped = _group_observations(module_input.sensor_measurements)
    if not grouped:
        raise ValueError("At least one supported observation modality is required.")

    measurements: dict[str, Array] = {}
    valid_flags: dict[str, Array] = {}
    filters: dict[str, LocalDynamicsEKF] = {}
    process_noise = np.asarray(
        _required(filter_config, config, "process_noise"), dtype=float
    ).reshape(6, 6)

    for modality, observations in grouped.items():
        measurement_array, flag_array, covariance = _align_modality(
            observations,
            timestamps,
            confidence_scaling=bool(filter_config.get("confidence_scaling", False)),
        )
        measurements[modality] = measurement_array
        valid_flags[modality] = flag_array
        modality_config = _modality_config(config, modality)
        filters[modality] = LocalDynamicsEKF(
            process_noise=process_noise,
            measurement_covariance=covariance,
            mode_name=modality,
            gate_enable=bool(modality_config.get("gate_enable", filter_config.get("gate_enable", False))),
            gate_threshold=float(modality_config.get("gate_threshold", np.inf)),
            gate_mode=str(modality_config.get("gate_mode", filter_config.get("gate_mode", "soft"))),
            soft_scale=float(modality_config.get("soft_scale", filter_config.get("soft_scale", 20.0))),
            regularization=float(filter_config.get("regularization", 1e-9)),
            legacy_fixed_jacobian_step=bool(filter_config.get("legacy_fixed_jacobian_step", True)),
            nn_meas_frame=str(modality_config.get("nn_meas_frame", "eci")),
            nn_use_pseudo_velocity=bool(modality_config.get("nn_use_pseudo_velocity", measurement_array.shape[1] == 6)),
        )

    node_id = str(runtime.get("node_id", config.get("node_id", "node_0")))
    target_id = str(module_input.initial_state.target_id)
    return FederatedAdapterResult(
        timestamps=timestamps,
        chief_state_history_eci=chief_history,
        q_eci2pri_history=q_history,
        measurements_by_modality=measurements,
        valid_flags_by_modality=valid_flags,
        local_filters=filters,
        initial_state=np.asarray(module_input.initial_state.state_estimate, dtype=float).reshape(6),
        initial_covariance=np.asarray(module_input.initial_state.covariance, dtype=float).reshape(6, 6),
        node_id=node_id,
        target_id=target_id,
        reset_feedback=bool(filter_config.get("reset_feedback", config.get("reset_feedback", False))),
        ci_objective=str(filter_config.get("ci_objective", config.get("ci_objective", "trace"))),
        ci_grid_points=int(filter_config.get("ci_grid_points", config.get("ci_grid_points", 101))),
    )


def _group_observations(observations: list[Observation]) -> dict[str, list[Observation]]:
    grouped: dict[str, list[Observation]] = {}
    for observation in observations:
        modality = _normalize_modality(observation)
        grouped.setdefault(modality, []).append(observation)
    for values in grouped.values():
        values.sort(key=lambda item: item.timestamp)
    return grouped


def _normalize_modality(observation: Observation) -> str:
    raw = str(observation.modality).upper()
    source = str(observation.source_type).upper()
    if raw in {"OPT", "OPTICAL"} and source == "LEARNING":
        return "nn"
    try:
        return _MODALITY_ALIASES[raw]
    except KeyError as exc:
        raise ValueError(f"Unsupported observation modality: {observation.modality}") from exc


def _align_modality(
    observations: list[Observation],
    timestamps: Array,
    *,
    confidence_scaling: bool,
) -> tuple[Array, Array, Array]:
    if not observations:
        raise ValueError("Observation group cannot be empty.")
    first_measurement = np.asarray(observations[0].measurement, dtype=float).reshape(-1)
    dimension = first_measurement.size
    base_covariance = np.asarray(observations[0].covariance, dtype=float).reshape(dimension, dimension)
    measurement_array = np.zeros((timestamps.size, dimension), dtype=float)
    valid_flags = np.zeros(timestamps.size, dtype=bool)
    timestamp_to_index = {float(value): index for index, value in enumerate(timestamps)}

    for observation in observations:
        timestamp = float(observation.timestamp)
        if timestamp not in timestamp_to_index:
            raise ValueError(
                f"Observation timestamp {timestamp} is not present in runtime timestamps."
            )
        index = timestamp_to_index[timestamp]
        if valid_flags[index]:
            raise ValueError(
                f"Duplicate valid observation for one modality at timestamp {timestamp}."
            )
        measurement = np.asarray(observation.measurement, dtype=float).reshape(-1)
        covariance = np.asarray(observation.covariance, dtype=float)
        if measurement.size != dimension or covariance.shape != (dimension, dimension):
            raise ValueError("Measurement dimensions must remain constant within a modality.")
        expected_covariance = base_covariance
        if confidence_scaling:
            confidence = float(np.clip(observation.confidence, 1e-6, 1.0))
            expected_covariance = covariance / confidence
        if not np.allclose(expected_covariance, base_covariance, rtol=1e-8, atol=1e-12):
            raise ValueError(
                "Current legacy-compatible adapter requires one constant covariance per modality."
            )
        measurement_array[index] = measurement
        valid_flags[index] = bool(observation.valid_flag)

    return measurement_array, valid_flags, base_covariance


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"config['{name}'] must be a mapping.")
    return dict(value)


def _required(section: Mapping[str, Any], root: Mapping[str, Any], key: str) -> Any:
    if key in section:
        return section[key]
    if key in root:
        return root[key]
    raise KeyError(f"Required configuration field missing: {key}")


def _modality_config(config: Mapping[str, Any], modality: str) -> dict[str, Any]:
    modalities = config.get("modalities", {})
    if not isinstance(modalities, Mapping):
        return {}
    aliases = {
        "opt": ("opt", "optical", "OPTICAL"),
        "nn": ("nn", "learning", "learning_optical", "NN"),
        "ir": ("ir", "infrared", "INFRARED"),
        "rad": ("rad", "radar", "RADAR"),
    }
    for key in aliases[modality]:
        value = modalities.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}

@dataclass(frozen=True)
class CentralizedAdapterResult:
    timestamps: Array
    chief_state_history_eci: Array
    q_eci2pri_history: Array
    measurements_by_modality: dict[str, Array]
    valid_flags_by_modality: dict[str, Array]
    centralized_filter: Any
    initial_state: Array
    initial_covariance: Array
    node_id: str
    target_id: str


def adapt_module_input_centralized(module_input: ModuleInput) -> CentralizedAdapterResult:
    """Build centralized-EKF inputs while reusing the same documented interface."""
    from orbital_core.centralized_filter import CentralizedDynamicsEKF

    base = adapt_module_input(module_input)
    config = module_input.config
    filter_config = _section(config, "filter")
    covariances = {
        name: filter_obj.R.copy()
        for name, filter_obj in base.local_filters.items()
    }
    thresholds = {
        name: float(filter_obj.gate_threshold)
        for name, filter_obj in base.local_filters.items()
    }
    nn_config = _modality_config(config, "nn") if "nn" in base.local_filters else {}
    centralized_filter = CentralizedDynamicsEKF(
        process_noise=next(iter(base.local_filters.values())).Q,
        measurement_covariances=covariances,
        gate_enable=bool(filter_config.get("gate_enable", False)),
        gate_thresholds=thresholds,
        gate_mode=str(filter_config.get("gate_mode", "soft")),
        soft_scale=float(filter_config.get("soft_scale", 20.0)),
        regularization=float(filter_config.get("regularization", 1e-9)),
        legacy_fixed_jacobian_step=bool(filter_config.get("legacy_fixed_jacobian_step", True)),
        nn_meas_frame=str(nn_config.get("nn_meas_frame", "eci")),
        nn_use_pseudo_velocity=bool(nn_config.get(
            "nn_use_pseudo_velocity",
            base.measurements_by_modality.get("nn", np.empty((0, 3))).shape[-1] == 6,
        )),
    )
    return CentralizedAdapterResult(
        timestamps=base.timestamps,
        chief_state_history_eci=base.chief_state_history_eci,
        q_eci2pri_history=base.q_eci2pri_history,
        measurements_by_modality=base.measurements_by_modality,
        valid_flags_by_modality=base.valid_flags_by_modality,
        centralized_filter=centralized_filter,
        initial_state=base.initial_state,
        initial_covariance=base.initial_covariance,
        node_id=base.node_id,
        target_id=base.target_id,
    )
