"""Single-host end-to-end acceptance for the online multi-target runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import time
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from adapters.infrared_image_adapter import InfraredCameraConfig
from adapters.multimodal_sensor_simulator import (
    MultimodalSensorSimulationConfig,
    simulate_multimodal_sensor_epoch,
)
from adapters.optical_image_adapter import OpticalCameraConfig
from adapters.radar_range_doppler_adapter import RadarRangeDopplerConfig
from brain_inspired.incremental_measurement_preprocessor import (
    IncrementalMultimodalCANNConfig,
    IncrementalMultimodalCANNPreprocessor,
)
from interfaces.online_multitarget_runtime import OnlineMultiTargetRuntime
from orbital_core.constants import R_EARTH
from orbital_core.coordinates import build_rtn_quaternion, dcm_to_quat_wxyz
from orbital_core.dynamics import propagate_absolute_orbit
from orbital_core.orbit_elements import keplerian_to_eci
from tracking.association import OnlineMultiTargetTracker


@dataclass(frozen=True)
class OnlineAcceptanceConfig:
    observer_count: int = 5
    target_count: int = 2
    duration_seconds: float = 20.0
    sample_interval_seconds: float = 1.0
    seed: int = 0
    image_size: int = 17
    cann_profile: str = "realtime"
    maximum_final_position_error_m: float = 2_000.0
    deadline_seconds: float = 1.0
    dropout_start_seconds: float | None = None
    dropout_end_seconds: float | None = None
    dropout_modality: str | None = None
    dropout_target_index: int | None = None

    def __post_init__(self):
        if self.observer_count < 2:
            raise ValueError("observer_count must be at least two.")
        if self.target_count < 1:
            raise ValueError("target_count must be positive.")
        if self.duration_seconds < 4.0 * self.sample_interval_seconds:
            raise ValueError("duration must cover IOD and post-IOD tracking.")
        if self.sample_interval_seconds <= 0.0:
            raise ValueError("sample_interval_seconds must be positive.")
        if self.image_size < 7:
            raise ValueError("image_size must be at least 7 pixels.")
        if self.cann_profile not in {"realtime", "research"}:
            raise ValueError("cann_profile must be realtime or research.")
        if self.maximum_final_position_error_m <= 0.0:
            raise ValueError("maximum_final_position_error_m must be positive.")
        if self.deadline_seconds <= 0.0:
            raise ValueError("deadline_seconds must be positive.")
        bounds = (self.dropout_start_seconds, self.dropout_end_seconds)
        if any(value is not None for value in bounds):
            if any(value is None for value in bounds):
                raise ValueError("dropout start and end must be supplied together.")
            if not (
                0.0 <= self.dropout_start_seconds
                <= self.dropout_end_seconds <= self.duration_seconds
            ):
                raise ValueError("dropout interval must lie inside the scenario.")
        modality = (
            None if self.dropout_modality is None
            else str(self.dropout_modality).upper()
        )
        if modality not in {None, "RADAR", "INFRARED", "OPTICAL", "ALL"}:
            raise ValueError("dropout_modality is unsupported.")
        object.__setattr__(self, "dropout_modality", modality)
        if (
            self.dropout_target_index is not None
            and not 0 <= self.dropout_target_index < self.target_count
        ):
            raise ValueError("dropout_target_index is outside the target range.")


@dataclass(frozen=True)
class OnlineAcceptanceArm:
    name: str
    frame_count: int
    initialized_target_count: int
    final_active_target_count: int
    false_birth_count: int
    identity_switch_count: int
    retired_target_count: int
    final_position_rmse_m: float | None
    maximum_final_position_error_m: float | None
    maximum_tracking_position_error_m: float | None
    processing_mean_ms: float
    processing_p50_ms: float
    processing_p95_ms: float
    processing_p99_ms: float
    processing_max_ms: float
    deadline_exceedance_fraction: float
    serialization_mean_ms: float
    trajectory_bytes_mean: float
    display_bytes_mean: float
    cann_preprocessed_observation_count: int
    cann_preprocessing_mean_ms: float
    cann_preprocessing_p95_ms: float
    algorithm_passed: bool
    timing_passed: bool
    passed: bool


@dataclass(frozen=True)
class OnlineLocalAcceptanceResult:
    config: dict[str, Any]
    input_generation_mean_ms: float
    input_generation_p95_ms: float
    input_frame_bytes_mean: float
    without_cann: OnlineAcceptanceArm
    incremental_cann: OnlineAcceptanceArm
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "inputGenerationMeanMs": self.input_generation_mean_ms,
            "inputGenerationP95Ms": self.input_generation_p95_ms,
            "inputFrameBytesMean": self.input_frame_bytes_mean,
            "withoutCann": _camel_arm(self.without_cann),
            "incrementalCann": _camel_arm(self.incremental_cann),
            "passed": self.passed,
        }


def run_online_multitarget_local_acceptance(
    config: OnlineAcceptanceConfig | None = None,
) -> OnlineLocalAcceptanceResult:
    """Run paired no-CANN and incremental-CANN arms on identical raw frames."""

    selected = config or OnlineAcceptanceConfig()
    frames, truths, generation_seconds, input_bytes = _build_frames(selected)
    without = _run_arm(selected, frames, truths, enable_cann=False)
    with_cann = _run_arm(selected, frames, truths, enable_cann=True)
    return OnlineLocalAcceptanceResult(
        config=asdict(selected),
        input_generation_mean_ms=_mean_ms(generation_seconds),
        input_generation_p95_ms=_percentile_ms(generation_seconds, 95),
        input_frame_bytes_mean=float(np.mean(input_bytes)),
        without_cann=without,
        incremental_cann=with_cann,
        passed=bool(without.passed and with_cann.passed),
    )


def _build_frames(config):
    times = np.arange(
        0.0,
        config.duration_seconds + 0.5 * config.sample_interval_seconds,
        config.sample_interval_seconds,
    )
    observer_histories = {
        f"observer-{index:02d}": propagate_absolute_orbit(
            _orbit_state(
                anomaly_deg=-2.0 + 0.35 * index,
                inclination_deg=54.5 + 0.12 * index,
                altitude_m=697_000.0 + 1_250.0 * index,
            ),
            times,
        )
        for index in range(config.observer_count)
    }
    target_histories = {
        f"physical-target-{index + 1:02d}": propagate_absolute_orbit(
            _orbit_state(
                anomaly_deg=1.0 + 2.0 * index,
                inclination_deg=55.2 + 0.08 * index,
                altitude_m=705_000.0 + 2_000.0 * index,
            ),
            times,
        )
        for index in range(config.target_count)
    }
    sensors = _sensor_config(config.image_size)
    streams = np.random.SeedSequence(config.seed).spawn(
        len(times) * config.observer_count * config.target_count * 3
    )
    stream_index = 0
    frames = []
    generation_seconds = []
    input_bytes = []
    for epoch_index, timestamp in enumerate(times):
        started = time.perf_counter()
        observers = []
        detections = []
        for observer_id, history in observer_histories.items():
            observer = history[epoch_index]
            observers.append({
                "observerId": observer_id,
                "stateEci": observer.tolist(),
                "qEci2PriWxyz": build_rtn_quaternion(observer).tolist(),
            })
            for target_index, (_physical_id, target_history) in enumerate(
                target_histories.items()
            ):
                target = target_history[epoch_index]
                attitude = _target_pointing_quaternion(target[:3] - observer[:3])
                epoch_streams = streams[stream_index:stream_index + 3]
                stream_index += 3
                sensor_epoch = simulate_multimodal_sensor_epoch(
                    timestamp=float(timestamp),
                    observer_id=observer_id,
                    target_id="opaque-detection-source",
                    observer_state=observer,
                    target_state=target,
                    quaternion_i2b_wxyz=attitude,
                    config=sensors,
                    rng_by_modality={
                        modality: np.random.default_rng(seed)
                        for modality, seed in zip(
                            ("RADAR", "INFRARED", "OPTICAL"),
                            epoch_streams,
                            strict=True,
                        )
                    },
                )
                group = f"detection-{observer_id}-{epoch_index}-{target_index}"
                for message in sensor_epoch.messages:
                    if not message.valid_flag:
                        continue
                    if _drop_detection(
                        config, timestamp=float(timestamp),
                        target_index=target_index, modality=message.modality,
                    ):
                        continue
                    detections.append({
                        "messageId": (
                            f"raw-{observer_id}-{epoch_index}-{target_index}-"
                            f"{message.modality.lower()}"
                        ),
                        "observerId": observer_id,
                        "modality": message.modality,
                        "detectionGroupId": group,
                        "measurement": message.measurement.tolist(),
                        "covariance": message.covariance.tolist(),
                        "confidence": float(message.confidence),
                        "valid": bool(message.valid_flag),
                        "metadata": _jsonable(message.metadata),
                    })
        frame = {
            "frameIndex": epoch_index,
            "timestamp": float(timestamp),
            "observers": observers,
            "detections": detections,
        }
        encoded = json.dumps(
            frame, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")
        frames.append(frame)
        generation_seconds.append(time.perf_counter() - started)
        input_bytes.append(len(encoded))
    truths = {
        physical_id: np.asarray(history, dtype=float)
        for physical_id, history in target_histories.items()
    }
    return frames, truths, generation_seconds, input_bytes


def _run_arm(config, frames, truths, *, enable_cann):
    tracker = OnlineMultiTargetTracker(
        scene_id="online-local-acceptance",
        measurement_preprocessor_factory=(
            (
                lambda _observer, _target: IncrementalMultimodalCANNPreprocessor(
                    IncrementalMultimodalCANNConfig(profile=config.cann_profile)
                )
            )
            if enable_cann else None
        ),
    )
    runtime = OnlineMultiTargetRuntime(
        scene_id="online-local-acceptance",
        time_base_id="TB-ONLINE-LOCAL-ACCEPTANCE",
        dataset_id="RAW-MULTIMODAL-PAIRED",
        start_time_ms=1_789_704_000_000,
        sample_interval_seconds=config.sample_interval_seconds,
        tracker=tracker,
    )
    processing = []
    serialization = []
    trajectory_bytes = []
    display_bytes = []
    cann_seconds = []
    cann_observations = 0
    created_ids = set()
    retired_ids = set()
    reference_physical_by_target = {}
    identity_switches = 0
    final_errors = []
    all_errors = []
    final_active_count = 0
    for epoch_index, frame in enumerate(frames):
        started = time.perf_counter()
        output = runtime.process_frame(frame)
        processing.append(time.perf_counter() - started)
        started = time.perf_counter()
        trajectory = json.dumps(
            output.trajectory_message, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")
        display = json.dumps(
            output.display_message, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")
        serialization.append(time.perf_counter() - started)
        trajectory_bytes.append(len(trajectory))
        display_bytes.append(len(display))
        created_ids.update(output.initialized_target_ids)
        retired_ids.update(output.retired_target_ids)
        diagnostics = output.display_message["estimation"]
        cann_observations += int(diagnostics["cannPreprocessedObservationCount"])
        cann_seconds.append(
            float(diagnostics["cannMeasurementPreprocessingMs"]) / 1_000.0
        )
        states = {
            item["satObj"]: np.asarray(
                [item["position"][key] for key in (
                    "x", "y", "z", "vx", "vy", "vz",
                )],
                dtype=float,
            )
            for item in output.trajectory_message["satelliteList"]
        }
        truth = {key: value[epoch_index] for key, value in truths.items()}
        matches = _match_states(states, truth)
        errors = []
        for target_id, physical_id in matches.items():
            expected = reference_physical_by_target.setdefault(target_id, physical_id)
            if physical_id != expected:
                identity_switches += 1
            errors.append(float(np.linalg.norm(
                states[target_id][:3] - truth[physical_id][:3],
            )))
        all_errors.extend(errors)
        final_errors = errors
        final_active_count = len(states)
    maximum_final_error = max(final_errors) if final_errors else None
    final_rmse = (
        float(np.sqrt(np.mean(np.square(final_errors))))
        if final_errors else None
    )
    maximum_tracking_error = max(all_errors) if all_errors else None
    false_birth_count = max(0, len(created_ids) - config.target_count)
    algorithm_passed = bool(
        len(created_ids) == config.target_count
        and final_active_count == config.target_count
        and false_birth_count == 0
        and identity_switches == 0
        and not retired_ids
        and maximum_final_error is not None
        and maximum_final_error <= config.maximum_final_position_error_m
    )
    total_processing = np.asarray(processing) + np.asarray(serialization)
    exceedance = float(np.mean(total_processing > config.deadline_seconds))
    timing_passed = bool(
        np.percentile(total_processing, 95) <= config.deadline_seconds
        and exceedance <= 0.05
    )
    return OnlineAcceptanceArm(
        name="incremental_cann" if enable_cann else "without_cann",
        frame_count=len(frames),
        initialized_target_count=len(created_ids),
        final_active_target_count=final_active_count,
        false_birth_count=false_birth_count,
        identity_switch_count=identity_switches,
        retired_target_count=len(retired_ids),
        final_position_rmse_m=final_rmse,
        maximum_final_position_error_m=maximum_final_error,
        maximum_tracking_position_error_m=maximum_tracking_error,
        processing_mean_ms=_mean_ms(processing),
        processing_p50_ms=_percentile_ms(processing, 50),
        processing_p95_ms=_percentile_ms(processing, 95),
        processing_p99_ms=_percentile_ms(processing, 99),
        processing_max_ms=_percentile_ms(processing, 100),
        deadline_exceedance_fraction=exceedance,
        serialization_mean_ms=_mean_ms(serialization),
        trajectory_bytes_mean=float(np.mean(trajectory_bytes)),
        display_bytes_mean=float(np.mean(display_bytes)),
        cann_preprocessed_observation_count=cann_observations,
        cann_preprocessing_mean_ms=_mean_ms(cann_seconds),
        cann_preprocessing_p95_ms=_percentile_ms(cann_seconds, 95),
        algorithm_passed=algorithm_passed,
        timing_passed=timing_passed,
        passed=bool(algorithm_passed and timing_passed),
    )


def _match_states(estimates, truths):
    estimate_items = tuple(sorted(estimates.items()))
    truth_items = tuple(sorted(truths.items()))
    if not estimate_items or not truth_items:
        return {}
    costs = np.asarray([
        [np.linalg.norm(estimate[:3] - truth[:3]) for _, truth in truth_items]
        for _, estimate in estimate_items
    ])
    rows, columns = linear_sum_assignment(costs)
    return {
        estimate_items[row][0]: truth_items[column][0]
        for row, column in zip(rows, columns)
    }


def _orbit_state(*, anomaly_deg, inclination_deg, altitude_m):
    return keplerian_to_eci(
        R_EARTH + altitude_m, 0.001, np.deg2rad(inclination_deg),
        np.deg2rad(15.0), 0.0, np.deg2rad(anomaly_deg),
    )


def _target_pointing_quaternion(relative_position):
    forward = np.asarray(relative_position, dtype=float).copy()
    forward /= np.linalg.norm(forward)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(forward @ reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    lateral = np.cross(reference, forward)
    lateral /= np.linalg.norm(lateral)
    vertical = np.cross(forward, lateral)
    return dcm_to_quat_wxyz(np.column_stack((forward, lateral, vertical)).T)


def _sensor_config(image_size):
    common = {
        "width": int(image_size), "height": int(image_size),
        "read_noise_sigma": 1.0,
    }
    return MultimodalSensorSimulationConfig(
        optical=OpticalCameraConfig(**common),
        infrared=InfraredCameraConfig(**common),
        radar=RadarRangeDopplerConfig(**common),
    )


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _drop_detection(config, *, timestamp, target_index, modality):
    if config.dropout_start_seconds is None or config.dropout_modality is None:
        return False
    if not (
        config.dropout_start_seconds
        <= timestamp <= config.dropout_end_seconds
    ):
        return False
    if (
        config.dropout_target_index is not None
        and target_index != config.dropout_target_index
    ):
        return False
    return bool(
        config.dropout_modality == "ALL"
        or str(modality).upper() == config.dropout_modality
    )


def _mean_ms(values):
    return float(np.mean(np.asarray(values, dtype=float)) * 1_000.0)


def _percentile_ms(values, percentile):
    return float(np.percentile(np.asarray(values, dtype=float), percentile) * 1_000.0)


def _camel_arm(arm):
    return {
        "name": arm.name,
        "frameCount": arm.frame_count,
        "initializedTargetCount": arm.initialized_target_count,
        "finalActiveTargetCount": arm.final_active_target_count,
        "falseBirthCount": arm.false_birth_count,
        "identitySwitchCount": arm.identity_switch_count,
        "retiredTargetCount": arm.retired_target_count,
        "finalPositionRmseM": arm.final_position_rmse_m,
        "maximumFinalPositionErrorM": arm.maximum_final_position_error_m,
        "maximumTrackingPositionErrorM": arm.maximum_tracking_position_error_m,
        "processingMeanMs": arm.processing_mean_ms,
        "processingP50Ms": arm.processing_p50_ms,
        "processingP95Ms": arm.processing_p95_ms,
        "processingP99Ms": arm.processing_p99_ms,
        "processingMaxMs": arm.processing_max_ms,
        "deadlineExceedanceFraction": arm.deadline_exceedance_fraction,
        "serializationMeanMs": arm.serialization_mean_ms,
        "trajectoryBytesMean": arm.trajectory_bytes_mean,
        "displayBytesMean": arm.display_bytes_mean,
        "cannPreprocessedObservationCount": arm.cann_preprocessed_observation_count,
        "cannPreprocessingMeanMs": arm.cann_preprocessing_mean_ms,
        "cannPreprocessingP95Ms": arm.cann_preprocessing_p95_ms,
        "algorithmPassed": arm.algorithm_passed,
        "timingPassed": arm.timing_passed,
        "passed": arm.passed,
    }
