"""Canonical raw three-modality simulation shared by single and swarm paths."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Mapping

import numpy as np

from adapters.infrared_image_adapter import (
    InfraredCameraConfig,
    infrared_frame_to_observation_message,
    render_infrared_point_source_frame,
)
from adapters.optical_image_adapter import (
    OpticalCameraConfig,
    optical_frame_to_observation_message,
    render_optical_point_source_frame,
)
from adapters.radar_range_doppler_adapter import (
    RadarRangeDopplerConfig,
    radar_frame_to_observation_message,
    render_radar_range_doppler_frame,
)
from interfaces.data_objects import Observation, ObservationMessage
from interfaces.interface_contracts import validate_observation
from orbital_core.measurements import (
    measure_relative_range,
    measure_relative_range_rate,
)
from orbital_core.attitude import quat_to_dcm_i2b


@dataclass(frozen=True)
class MultimodalSensorSimulationConfig:
    optical: OpticalCameraConfig = field(default_factory=OpticalCameraConfig)
    infrared: InfraredCameraConfig = field(default_factory=InfraredCameraConfig)
    radar: RadarRangeDopplerConfig = field(default_factory=RadarRangeDopplerConfig)


class MeasurementSource(str, Enum):
    ANALYTIC = "analytic"
    RAW_FRONTEND = "raw_frontend"


@dataclass(frozen=True)
class MultimodalMeasurementSourceConfig:
    source: MeasurementSource | str = MeasurementSource.ANALYTIC
    sensors: MultimodalSensorSimulationConfig = field(
        default_factory=MultimodalSensorSimulationConfig
    )
    covariance_calibration_by_modality: Mapping[str, object] = field(
        default_factory=dict
    )

    def __post_init__(self):
        try:
            source = MeasurementSource(self.source)
        except ValueError as exc:
            raise ValueError(
                "measurement source must be 'analytic' or 'raw_frontend'."
            ) from exc
        object.__setattr__(self, "source", source)
        calibrations = {
            str(modality).upper(): table
            for modality, table in self.covariance_calibration_by_modality.items()
        }
        unknown = set(calibrations) - {"RADAR", "INFRARED", "OPTICAL"}
        if unknown:
            raise ValueError(
                f"Unknown calibration modalities: {sorted(unknown)}."
            )
        object.__setattr__(
            self, "covariance_calibration_by_modality", calibrations,
        )


@dataclass(frozen=True)
class MultimodalSensorEpoch:
    messages: tuple[ObservationMessage, ...]
    raw_frames: dict[str, object]


@dataclass(frozen=True)
class MultimodalSensorHistory:
    messages: tuple[ObservationMessage, ...]
    observations: tuple[Observation, ...]
    raw_frames_by_modality: dict[str, tuple[object, ...]]


def simulate_multimodal_sensor_epoch(
    *, timestamp, observer_id, target_id, observer_state, target_state,
    quaternion_i2b_wxyz=None, quaternion_i2b_wxyz_by_modality=None,
    config=None, rng_by_modality=None,
    radar_acquisition_center=None, covariance_calibration_by_modality=None,
    valid_by_modality=None,
):
    """Render one canonical RADAR/INFRARED/OPTICAL epoch.

    Camera modalities use BODY +X boresights but may have different mounting
    attitudes. A shared quaternion remains available for co-boresighted sensors.
    Radar remains the physical range/range-rate pair and defaults to a
    truth-centered acquisition window only when no explicit tracker center is
    supplied.
    """
    selected = config or MultimodalSensorSimulationConfig()
    rngs = dict(rng_by_modality or {})
    calibrations = dict(covariance_calibration_by_modality or {})
    observer = np.asarray(observer_state, dtype=float).reshape(6)
    target = np.asarray(target_state, dtype=float).reshape(6)
    attitudes = dict(quaternion_i2b_wxyz_by_modality or {})
    if quaternion_i2b_wxyz is not None:
        attitudes.setdefault("OPTICAL", quaternion_i2b_wxyz)
        attitudes.setdefault("INFRARED", quaternion_i2b_wxyz)
    missing_attitudes = {"OPTICAL", "INFRARED"} - set(attitudes)
    if missing_attitudes:
        raise ValueError(
            "Camera attitudes are required for "
            f"{sorted(missing_attitudes)}."
        )
    optical_quaternion = np.asarray(
        attitudes["OPTICAL"], dtype=float,
    ).reshape(4)
    infrared_quaternion = np.asarray(
        attitudes["INFRARED"], dtype=float,
    ).reshape(4)
    if radar_acquisition_center is None:
        center = np.array([
            measure_relative_range(observer, target),
            measure_relative_range_rate(observer, target),
        ])
        center_source = "truth_calibration"
    else:
        center = np.asarray(radar_acquisition_center, dtype=float).reshape(2)
        center_source = "external_prediction"

    optical_frame = render_optical_point_source_frame(
        timestamp=timestamp, observer_id=observer_id, target_id=target_id,
        observer_state=observer, target_state=target,
        quaternion_i2b_wxyz=optical_quaternion, config=selected.optical,
        rng=rngs.get("OPTICAL"),
    )
    infrared_frame = render_infrared_point_source_frame(
        timestamp=timestamp, observer_id=observer_id, target_id=target_id,
        observer_state=observer, target_state=target,
        quaternion_i2b_wxyz=infrared_quaternion, config=selected.infrared,
        rng=rngs.get("INFRARED"),
    )
    radar_frame = render_radar_range_doppler_frame(
        timestamp=timestamp, observer_id=observer_id, target_id=target_id,
        observer_state=observer, target_state=target,
        acquisition_range_m=center[0], acquisition_range_rate_mps=center[1],
        config=selected.radar, rng=rngs.get("RADAR"),
    )
    optical = optical_frame_to_observation_message(
        optical_frame, config=selected.optical,
        covariance_calibration=calibrations.get("OPTICAL"),
    )
    infrared = infrared_frame_to_observation_message(
        infrared_frame, config=selected.infrared,
        covariance_calibration=calibrations.get("INFRARED"),
    )
    radar = radar_frame_to_observation_message(
        radar_frame, config=selected.radar,
        covariance_calibration=calibrations.get("RADAR"),
    )
    radar.metadata["acquisition_center_mode"] = center_source
    externally_valid = dict(valid_by_modality or {})
    messages = tuple(
        replace(
            message,
            valid_flag=bool(
                message.valid_flag
                and externally_valid.get(message.modality, True)
            ),
            metadata={
                **message.metadata,
                "external_visibility_valid": bool(
                    externally_valid.get(message.modality, True)
                ),
            },
        )
        for message in (radar, infrared, optical)
    )
    return MultimodalSensorEpoch(
        messages=messages,
        raw_frames={
            "RADAR": radar_frame,
            "INFRARED": infrared_frame,
            "OPTICAL": optical_frame,
        },
    )


def simulate_multimodal_sensor_history(
    *, timestamps, observer_id, target_id,
    observer_state_history, target_state_history,
    quaternion_i2b_wxyz_history_by_modality,
    config=None, random_seed=0, valid_history_by_modality=None,
    radar_acquisition_center_history=None,
    covariance_calibration_by_modality=None,
):
    """Generate one paired history for both interface representations."""
    times = np.asarray(timestamps, dtype=float).reshape(-1)
    observer = np.asarray(observer_state_history, dtype=float)
    target = np.asarray(target_state_history, dtype=float)
    if observer.shape != (len(times), 6) or target.shape != (len(times), 6):
        raise ValueError("State histories must have shape (N, 6).")
    attitudes = {
        modality: np.asarray(values, dtype=float)
        for modality, values in quaternion_i2b_wxyz_history_by_modality.items()
    }
    for modality in ("OPTICAL", "INFRARED"):
        if attitudes.get(modality, np.empty((0, 4))).shape != (len(times), 4):
            raise ValueError(f"{modality} attitude history must have shape (N, 4).")
    valid_histories = {
        modality: np.asarray(values, dtype=bool).reshape(-1)
        for modality, values in (valid_history_by_modality or {}).items()
    }
    if any(values.shape != (len(times),) for values in valid_histories.values()):
        raise ValueError("Visibility histories must have shape (N,).")
    centers = (
        None if radar_acquisition_center_history is None
        else np.asarray(radar_acquisition_center_history, dtype=float)
    )
    if centers is not None and centers.shape != (len(times), 2):
        raise ValueError("Radar acquisition centers must have shape (N, 2).")
    seed_sequence = np.random.SeedSequence(int(random_seed))
    rngs = {
        modality: np.random.default_rng(child)
        for modality, child in zip(
            ("RADAR", "INFRARED", "OPTICAL"), seed_sequence.spawn(3)
        )
    }
    messages = []
    frames = {"RADAR": [], "INFRARED": [], "OPTICAL": []}
    for index, timestamp in enumerate(times):
        epoch = simulate_multimodal_sensor_epoch(
            timestamp=timestamp, observer_id=observer_id, target_id=target_id,
            observer_state=observer[index], target_state=target[index],
            quaternion_i2b_wxyz_by_modality={
                modality: attitudes[modality][index]
                for modality in ("OPTICAL", "INFRARED")
            },
            config=config, rng_by_modality=rngs,
            radar_acquisition_center=(
                None if centers is None else centers[index]
            ),
            covariance_calibration_by_modality=(
                covariance_calibration_by_modality
            ),
            valid_by_modality={
                modality: values[index]
                for modality, values in valid_histories.items()
            },
        )
        messages.extend(epoch.messages)
        for modality, frame in epoch.raw_frames.items():
            frames[modality].append(frame)
    message_tuple = tuple(messages)
    return MultimodalSensorHistory(
        messages=message_tuple,
        observations=tuple(observation_from_message(item) for item in message_tuple),
        raw_frames_by_modality={
            modality: tuple(values) for modality, values in frames.items()
        },
    )


def observation_from_message(message, *, source_type="RAW_SENSOR"):
    """Losslessly expose a network observation to the single-star interface."""
    observation = Observation(
        timestamp=float(message.timestamp), observer_id=str(message.observer_id),
        target_id=str(message.target_id), modality=str(message.modality),
        source_type=str(source_type),
        measurement=np.asarray(message.measurement, dtype=float).copy(),
        covariance=np.asarray(message.covariance, dtype=float).copy(),
        confidence=float(message.confidence), frame=str(message.frame),
        valid_flag=bool(message.valid_flag),
        metadata={
            **message.metadata,
            "observation_id": message.information_id,
            "source_timestamp": message.source_timestamp,
            "arrival_timestamp": message.arrival_timestamp,
        },
    )
    validate_observation(observation)
    return observation


def single_spri_observation_from_message(message, *, source_type="RAW_SENSOR"):
    """Adapt a raw network message to the legacy single-filter SPRI contract.

    The raw camera frontends report BODY +X pinhole coordinates, whereas the
    existing single-satellite filters consume SPRI az/el (infrared) or SPRI
    +Z normalized coordinates (optical).  This boundary conversion keeps the
    filter models unchanged and propagates the reported covariance through the
    coordinate transform.
    """
    base = observation_from_message(message, source_type=source_type)
    modality = str(message.modality).upper()
    if modality == "RADAR":
        return replace(base, frame="SPRI")
    if modality not in {"INFRARED", "OPTICAL"}:
        return base
    quaternion = message.metadata.get("quaternion_i2b_wxyz")
    if quaternion is None:
        raise ValueError(f"{modality} BODY measurement requires attitude metadata.")
    dcm_spri_to_body = quat_to_dcm_i2b(quaternion)

    def transform(values):
        first, second = np.asarray(values, dtype=float).reshape(2)
        if modality == "INFRARED":
            u = np.tan(first)
            ray_body = np.array([
                1.0, u, np.tan(second) * np.sqrt(1.0 + u * u),
            ])
            ray_spri = dcm_spri_to_body.T @ ray_body
            return np.array([
                np.arctan2(ray_spri[1], ray_spri[0]),
                np.arctan2(ray_spri[2], np.linalg.norm(ray_spri[:2])),
            ])
        ray_spri = dcm_spri_to_body.T @ np.array([1.0, first, second])
        return ray_spri[:2] / ray_spri[2]

    measurement = transform(message.measurement)
    epsilon = 1.0e-6
    jacobian = np.empty((2, 2))
    for axis in range(2):
        delta = np.zeros(2)
        delta[axis] = epsilon
        jacobian[:, axis] = (
            transform(np.asarray(message.measurement) + delta)
            - transform(np.asarray(message.measurement) - delta)
        ) / (2.0 * epsilon)
    covariance = jacobian @ base.covariance @ jacobian.T
    valid = bool(base.valid_flag)
    if modality == "OPTICAL":
        ray_spri = dcm_spri_to_body.T @ np.array([
            1.0, message.measurement[0], message.measurement[1],
        ])
        valid = valid and bool(ray_spri[2] > 1.0e-12)
    adapted = replace(
        base, measurement=measurement, covariance=covariance, frame="SPRI",
        valid_flag=valid,
        metadata={
            **base.metadata,
            "source_measurement_frame": str(message.frame),
            "single_filter_frame_adapter": "BODY_TO_SPRI",
        },
    )
    validate_observation(adapted)
    return adapted


def raw_frontend_observations_for_single_epoch(**kwargs):
    """Return canonical single-star observations from the shared raw epoch."""
    epoch = simulate_multimodal_sensor_epoch(**kwargs)
    return tuple(observation_from_message(message) for message in epoch.messages)


def select_single_epoch_observations(
    *, source_config, analytic_observations, raw_frontend_arguments=None,
):
    """Explicitly route one single-star epoch without touching analytic data."""
    selected = source_config or MultimodalMeasurementSourceConfig()
    analytic = tuple(analytic_observations)
    if selected.source is MeasurementSource.ANALYTIC:
        return analytic
    if raw_frontend_arguments is None:
        raise ValueError(
            "raw_frontend_arguments are required for raw_frontend mode."
        )
    return raw_frontend_observations_for_single_epoch(
        config=selected.sensors, **raw_frontend_arguments,
    )
