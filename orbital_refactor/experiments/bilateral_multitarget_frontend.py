"""Raw three-modality frontend for the bilateral 15-friendly/15-hostile case.

The simulation/evaluation side knows which truth target generated a detection.
The algorithm-facing messages deliberately do not: identity is retained only
in a separate truth lookup used by tests and later scoring.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import time
from types import MappingProxyType
from typing import Mapping

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
from cooperative.topology import NetworkTopology
from orbital_core.coordinates import build_rtn_quaternion, dcm_to_quat_wxyz
from scenarios.bilateral_multitarget import (
    BilateralWalkerTruth,
    generate_bilateral_walker_truth,
)
from scenarios.measurement_visibility import (
    VisibilityConfig,
    evaluate_inter_satellite_visibility,
)
from tracking.association.data_contracts import UnlabeledObservationMessage
from tracking.association.frontend_adapter import (
    unlabeled_iod_observations_from_messages,
)
from tracking.association.online_tracker import OnlineMultiTargetTracker
from tracking.distributed_target_consensus import DistributedTargetConsensus


MODALITIES = ("RADAR", "INFRARED", "OPTICAL")


@dataclass(frozen=True)
class BilateralFrontendConfig:
    """Small deterministic Stage-B frontend configuration."""

    duration_seconds: float = 2.0
    sample_interval_seconds: float = 1.0
    maximum_observation_range_m: float = 6_000_000.0
    image_size: int = 17
    random_seed: int = 0
    retain_raw_frames: bool = False
    sensor_config: MultimodalSensorSimulationConfig | None = None
    dropout_modality: str | None = None
    dropout_start_seconds: float | None = None
    dropout_end_seconds: float | None = None
    dropout_target_indices: tuple[int, ...] | None = None
    observer_failure_start_seconds: float | None = None
    observer_failure_end_seconds: float | None = None
    failed_observer_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.duration_seconds) or self.duration_seconds <= 0.0:
            raise ValueError("duration_seconds must be positive and finite.")
        if (
            not np.isfinite(self.sample_interval_seconds)
            or self.sample_interval_seconds <= 0.0
        ):
            raise ValueError("sample_interval_seconds must be positive and finite.")
        if (
            not np.isfinite(self.maximum_observation_range_m)
            or self.maximum_observation_range_m <= 0.0
        ):
            raise ValueError(
                "maximum_observation_range_m must be positive and finite."
            )
        if int(self.image_size) < 9:
            raise ValueError("image_size must be at least 9 pixels.")
        modality = (
            None if self.dropout_modality is None
            else str(self.dropout_modality).upper()
        )
        if modality not in {None, *MODALITIES, "ALL"}:
            raise ValueError("dropout_modality is unsupported.")
        bounds = (self.dropout_start_seconds, self.dropout_end_seconds)
        if any(value is not None for value in bounds):
            if any(value is None for value in bounds):
                raise ValueError("Dropout start and end must be supplied together.")
            if not (
                0.0 <= self.dropout_start_seconds
                <= self.dropout_end_seconds <= self.duration_seconds
            ):
                raise ValueError("Dropout interval must lie inside the scenario.")
            if modality is None:
                raise ValueError("A dropout interval requires dropout_modality.")
        elif modality is not None:
            raise ValueError("dropout_modality requires a dropout interval.")
        indices = self.dropout_target_indices
        if indices is not None:
            indices = tuple(int(value) for value in indices)
            if len(set(indices)) != len(indices) or (
                indices and min(indices) < 0
            ):
                raise ValueError(
                    "dropout_target_indices must contain unique nonnegative values."
                )
        object.__setattr__(self, "dropout_modality", modality)
        object.__setattr__(self, "dropout_target_indices", indices)
        observer_bounds = (
            self.observer_failure_start_seconds,
            self.observer_failure_end_seconds,
        )
        failed_indices = self.failed_observer_indices
        if failed_indices is not None:
            failed_indices = tuple(int(value) for value in failed_indices)
            if len(set(failed_indices)) != len(failed_indices) or (
                failed_indices and min(failed_indices) < 0
            ):
                raise ValueError(
                    "failed_observer_indices must contain unique nonnegative values."
                )
        if any(value is not None for value in observer_bounds):
            if any(value is None for value in observer_bounds):
                raise ValueError(
                    "Observer failure start and end must be supplied together."
                )
            if not (
                0.0 <= self.observer_failure_start_seconds
                <= self.observer_failure_end_seconds <= self.duration_seconds
            ):
                raise ValueError(
                    "Observer failure interval must lie inside the scenario."
                )
            if failed_indices is None:
                raise ValueError(
                    "An observer failure interval requires failed_observer_indices."
                )
        elif failed_indices is not None:
            raise ValueError(
                "failed_observer_indices requires an observer failure interval."
            )
        if modality is not None and any(
            value is not None for value in observer_bounds
        ):
            raise ValueError(
                "Modality dropout and observer failure must be tested separately."
            )
        object.__setattr__(self, "failed_observer_indices", failed_indices)


@dataclass(frozen=True)
class BilateralFrontendResult:
    """Target-free message stream plus isolated evaluation truth."""

    config: BilateralFrontendConfig
    truth: BilateralWalkerTruth
    messages_by_timestamp: Mapping[float, tuple[UnlabeledObservationMessage, ...]]
    observer_state_by_epoch: Mapping[tuple[str, float], np.ndarray]
    truth_target_by_detection_group: Mapping[str, str]
    visible_observers_by_target_and_timestamp: Mapping[
        tuple[str, float], tuple[str, ...]
    ]
    raw_frames_by_detection_group: Mapping[str, Mapping[str, object]] = field(
        default_factory=dict
    )

    @property
    def messages(self) -> tuple[UnlabeledObservationMessage, ...]:
        return tuple(
            message
            for timestamp in self.truth.timestamps
            for message in self.messages_by_timestamp[float(timestamp)]
        )

    @property
    def minimum_visible_observers_per_target_epoch(self) -> int:
        counts = [
            len(observers)
            for observers in self.visible_observers_by_target_and_timestamp.values()
        ]
        return min(counts) if counts else 0

    @property
    def maximum_visible_observers_per_target_epoch(self) -> int:
        counts = [
            len(observers)
            for observers in self.visible_observers_by_target_and_timestamp.values()
        ]
        return max(counts) if counts else 0


@dataclass(frozen=True)
class BilateralOnlineSmokeResult:
    """Short IOD-to-tracking evidence for all hostile targets."""

    frame_count: int
    initialized_track_count: int
    final_active_track_count: int
    retired_track_count: int
    final_position_rmse_m: float
    final_maximum_position_error_m: float
    final_velocity_rmse_mps: float
    final_acceleration_rmse_mps2: float
    processing_mean_ms: float
    processing_p95_ms: float
    cann_preprocessed_observation_count: int
    cann_action_counts: Mapping[str, int]
    all_states_finite: bool
    phase_metrics: Mapping[str, "BilateralPhaseMetrics"]
    consensus_received_message_count: int
    consensus_recovered_link_count: int
    active_communication_node_count_by_timestamp: Mapping[float, int]
    active_communication_edge_count_by_timestamp: Mapping[float, int]


@dataclass(frozen=True)
class BilateralPhaseMetrics:
    """Fleet-wide error summary for one temporal phase."""

    epoch_count: int
    position_sample_count: int
    acceleration_sample_count: int
    position_rmse_m: float
    velocity_rmse_mps: float
    acceleration_rmse_mps2: float


@dataclass(frozen=True)
class BilateralOnlineComparisonResult:
    """Paired no-CANN/CANN arms evaluated on the same frontend frames."""

    without_cann: BilateralOnlineSmokeResult
    incremental_cann: BilateralOnlineSmokeResult
    message_count_by_timestamp: Mapping[float, int]


def build_bilateral_raw_frontend(
    config: BilateralFrontendConfig | None = None,
    *,
    truth: BilateralWalkerTruth | None = None,
) -> BilateralFrontendResult:
    """Render visible friendly-to-hostile detections and strip truth identity."""

    selected = config or BilateralFrontendConfig()
    scene_truth = truth or generate_bilateral_walker_truth(
        duration_seconds=selected.duration_seconds,
        sample_interval_seconds=selected.sample_interval_seconds,
    )
    if (
        selected.dropout_target_indices
        and max(selected.dropout_target_indices)
        >= len(scene_truth.manifest.hostile_target_ids)
    ):
        raise ValueError("dropout_target_indices is outside the hostile roster.")
    if (
        selected.failed_observer_indices
        and max(selected.failed_observer_indices)
        >= len(scene_truth.manifest.friendly_observer_ids)
    ):
        raise ValueError("failed_observer_indices is outside the friendly roster.")
    sensor_config = selected.sensor_config or _sensor_config(selected.image_size)
    visibility_config = VisibilityConfig(
        earth_occultation=True,
        maximum_range=selected.maximum_observation_range_m,
    )
    messages_by_timestamp: dict[
        float, tuple[UnlabeledObservationMessage, ...]
    ] = {}
    observer_state_by_epoch: dict[tuple[str, float], np.ndarray] = {}
    truth_by_group: dict[str, str] = {}
    visible_by_target_epoch: dict[tuple[str, float], tuple[str, ...]] = {}
    raw_frames: dict[str, Mapping[str, object]] = {}
    friendly_histories = scene_truth.friendly_truth_state_history_by_node
    hostile_histories = scene_truth.hostile_truth_state_history_by_target

    for epoch_index, timestamp_value in enumerate(scene_truth.timestamps):
        timestamp = float(timestamp_value)
        epoch_messages: list[UnlabeledObservationMessage] = []
        visible_observers: dict[str, list[str]] = {
            target_id: [] for target_id in scene_truth.manifest.hostile_target_ids
        }
        for observer_index, observer_id in enumerate(
            scene_truth.manifest.friendly_observer_ids
        ):
            observer_state = friendly_histories[observer_id][epoch_index]
            observer_state_by_epoch[(observer_id, timestamp)] = observer_state.copy()
            if _observer_failed(
                selected,
                timestamp=timestamp,
                observer_index=observer_index,
            ):
                continue
            for target_index, target_id in enumerate(
                scene_truth.manifest.hostile_target_ids
            ):
                scene_truth.manifest.require_observation_pair(observer_id, target_id)
                target_state = hostile_histories[target_id][epoch_index]
                visibility = evaluate_inter_satellite_visibility(
                    observer_state, target_state, visibility_config
                )
                if not visibility.visible:
                    continue
                visible_observers[target_id].append(observer_id)
                detection_group = (
                    f"det:{epoch_index:06d}:{observer_index:02d}:"
                    f"{target_index:02d}"
                )
                quaternion = _target_pointing_quaternion(
                    target_state[:3] - observer_state[:3]
                )
                sensor_epoch = simulate_multimodal_sensor_epoch(
                    timestamp=timestamp,
                    observer_id=observer_id,
                    target_id=target_id,
                    observer_state=observer_state,
                    target_state=target_state,
                    quaternion_i2b_wxyz=quaternion,
                    config=sensor_config,
                    rng_by_modality=_rng_by_modality(
                        selected.random_seed,
                        epoch_index,
                        observer_index,
                        target_index,
                    ),
                )
                truth_by_group[detection_group] = target_id
                if selected.retain_raw_frames:
                    raw_frames[detection_group] = MappingProxyType(
                        dict(sensor_epoch.raw_frames)
                    )
                for message in sensor_epoch.messages:
                    if _drop_message(
                        selected,
                        timestamp=timestamp,
                        target_index=target_index,
                        modality=message.modality,
                    ):
                        continue
                    epoch_messages.append(
                        _strip_truth_identity(message, detection_group)
                    )
        messages_by_timestamp[timestamp] = tuple(epoch_messages)
        for target_id, observers in visible_observers.items():
            visible_by_target_epoch[(target_id, timestamp)] = tuple(observers)

    return BilateralFrontendResult(
        config=selected,
        truth=scene_truth,
        messages_by_timestamp=MappingProxyType(messages_by_timestamp),
        observer_state_by_epoch=MappingProxyType(observer_state_by_epoch),
        truth_target_by_detection_group=MappingProxyType(truth_by_group),
        visible_observers_by_target_and_timestamp=MappingProxyType(
            visible_by_target_epoch
        ),
        raw_frames_by_detection_group=MappingProxyType(raw_frames),
    )


def run_bilateral_online_smoke(
    config: BilateralFrontendConfig | None = None,
    *,
    enable_cann: bool = False,
) -> BilateralOnlineSmokeResult:
    """Exercise raw frontend, autonomous IOD, association, and local fusion."""

    selected = config or BilateralFrontendConfig(duration_seconds=3.0)
    frontend = build_bilateral_raw_frontend(selected)
    return _run_bilateral_online_arm(frontend, enable_cann=enable_cann)


def run_bilateral_online_comparison(
    config: BilateralFrontendConfig | None = None,
) -> BilateralOnlineComparisonResult:
    """Run paired arms without regenerating or randomizing sensor input."""

    selected = config or BilateralFrontendConfig(duration_seconds=4.0)
    frontend = build_bilateral_raw_frontend(selected)
    return BilateralOnlineComparisonResult(
        without_cann=_run_bilateral_online_arm(frontend, enable_cann=False),
        incremental_cann=_run_bilateral_online_arm(frontend, enable_cann=True),
        message_count_by_timestamp=MappingProxyType({
            timestamp: len(messages)
            for timestamp, messages in frontend.messages_by_timestamp.items()
        }),
    )


def run_bilateral_distributed_online_comparison(
    config: BilateralFrontendConfig | None = None,
) -> BilateralOnlineComparisonResult:
    """Run paired CANN arms with fixed-topology target-level Consensus-CI."""

    selected = config or BilateralFrontendConfig(duration_seconds=4.0)
    frontend = build_bilateral_raw_frontend(selected)
    return BilateralOnlineComparisonResult(
        without_cann=_run_bilateral_online_arm(
            frontend, enable_cann=False, enable_consensus=True,
        ),
        incremental_cann=_run_bilateral_online_arm(
            frontend, enable_cann=True, enable_consensus=True,
        ),
        message_count_by_timestamp=MappingProxyType({
            timestamp: len(messages)
            for timestamp, messages in frontend.messages_by_timestamp.items()
        }),
    )


def _run_bilateral_online_arm(
    frontend: BilateralFrontendResult,
    *,
    enable_cann: bool,
    enable_consensus: bool = False,
) -> BilateralOnlineSmokeResult:
    tracker = OnlineMultiTargetTracker(
        scene_id=frontend.truth.manifest.scene_id,
        measurement_preprocessor_factory=(
            (
                lambda _observer, _target: IncrementalMultimodalCANNPreprocessor(
                    IncrementalMultimodalCANNConfig(profile="realtime")
                )
            )
            if enable_cann else None
        ),
    )
    consensus = (
        DistributedTargetConsensus(
            scene_id=frontend.truth.manifest.scene_id,
            topology=frontend.truth.manifest.friendly_topology,
        )
        if enable_consensus else None
    )
    base_topology = frontend.truth.manifest.friendly_topology
    initialized_ids: set[str] = set()
    retired_ids: set[str] = set()
    final_update = None
    processing_seconds: list[float] = []
    cann_observations = 0
    cann_actions: Counter[str] = Counter()
    consensus_received = 0
    consensus_recovered = 0
    active_node_counts: dict[float, int] = {}
    active_edge_counts: dict[float, int] = {}
    matched_history_by_truth: dict[str, list[tuple[float, np.ndarray]]] = {}
    state_errors_by_timestamp: dict[float, np.ndarray] = {}
    for epoch_index, timestamp_value in enumerate(frontend.truth.timestamps):
        timestamp = float(timestamp_value)
        observer_states = {
            observer_id: frontend.observer_state_by_epoch[(observer_id, timestamp)]
            for observer_id in frontend.truth.manifest.friendly_observer_ids
        }
        observations = unlabeled_iod_observations_from_messages(
            frontend.messages_by_timestamp[timestamp],
            observer_state_by_epoch=frontend.observer_state_by_epoch,
        )
        started = time.perf_counter()
        final_update = tracker.step(
            timestamp=timestamp,
            observations=observations,
            observer_states_eci=observer_states,
            q_eci2pri_by_observer={
                observer_id: build_rtn_quaternion(state)
                for observer_id, state in observer_states.items()
            },
        )
        if consensus is not None:
            active_observers = _active_observer_ids(
                frontend, timestamp=timestamp,
            )
            active_topology = _topology_with_inactive_nodes(
                base_topology,
                inactive_node_ids=(
                    set(base_topology.node_ids) - set(active_observers)
                ),
            )
            if active_topology.adjacency != consensus.topology.adjacency:
                consensus.update_topology(active_topology)
            consensus_step = consensus.step(
                timestamp=timestamp,
                local_reports=final_update.local_reports,
                active_source_ids=active_observers,
            )
            tracker.apply_consensus_feedback(consensus_step)
            consensus_received += sum(
                len(message_ids)
                for message_ids in (
                    consensus_step.received_message_ids_by_node.values()
                )
            )
            consensus_recovered += sum(
                len(source_ids)
                for source_ids in consensus_step.recovered_source_ids_by_node.values()
            )
            active_node_counts[timestamp] = len(active_observers)
            active_edge_counts[timestamp] = _undirected_edge_count(active_topology)
        processing_seconds.append(time.perf_counter() - started)
        cann_observations += final_update.cann_preprocessed_observation_count
        cann_actions.update(final_update.cann_action_counts)
        initialized_ids.update(final_update.initialized_states_by_target)
        retired_ids.update(final_update.retired_target_ids)
        state_errors_by_timestamp[timestamp] = _append_matched_history(
            matched_history_by_truth,
            timestamp=timestamp,
            tracks=final_update.tracks_by_target,
            truth_state_by_target={
                target_id: history[epoch_index]
                for target_id, history in (
                    frontend.truth.hostile_truth_state_history_by_target.items()
                )
            },
        )

    estimates = tuple(
        track.estimate.state_eci for track in final_update.tracks_by_target.values()
    )
    truths = tuple(
        history[-1]
        for history in frontend.truth.hostile_truth_state_history_by_target.values()
    )
    errors = _minimum_assignment_state_errors(estimates, truths)
    acceleration_errors = _final_acceleration_errors(
        matched_history_by_truth,
        frontend.truth.hostile_truth_state_history_by_target,
        frontend.truth.timestamps,
    )
    acceleration_errors_by_timestamp = _acceleration_errors_by_timestamp(
        matched_history_by_truth,
        frontend.truth.hostile_truth_state_history_by_target,
        frontend.truth.timestamps,
    )
    return BilateralOnlineSmokeResult(
        frame_count=len(frontend.truth.timestamps),
        initialized_track_count=len(initialized_ids),
        final_active_track_count=len(estimates),
        retired_track_count=len(retired_ids),
        final_position_rmse_m=float(np.sqrt(np.mean(np.square(errors[:, 0])))),
        final_maximum_position_error_m=float(np.max(errors[:, 0])),
        final_velocity_rmse_mps=float(np.sqrt(np.mean(np.square(errors[:, 1])))),
        final_acceleration_rmse_mps2=float(
            np.sqrt(np.mean(np.square(acceleration_errors)))
        ),
        processing_mean_ms=float(np.mean(processing_seconds) * 1_000.0),
        processing_p95_ms=float(
            np.percentile(processing_seconds, 95) * 1_000.0
        ),
        cann_preprocessed_observation_count=cann_observations,
        cann_action_counts=MappingProxyType(dict(sorted(cann_actions.items()))),
        all_states_finite=bool(
            estimates and all(np.all(np.isfinite(state)) for state in estimates)
        ),
        phase_metrics=MappingProxyType(_phase_metrics(
            frontend.config,
            state_errors_by_timestamp,
            acceleration_errors_by_timestamp,
        )),
        consensus_received_message_count=consensus_received,
        consensus_recovered_link_count=consensus_recovered,
        active_communication_node_count_by_timestamp=MappingProxyType(
            active_node_counts
        ),
        active_communication_edge_count_by_timestamp=MappingProxyType(
            active_edge_counts
        ),
    )


def _minimum_assignment_state_errors(estimates, truths) -> np.ndarray:
    if not estimates or not truths or len(estimates) != len(truths):
        raise ValueError(
            "Smoke-test scoring requires equal nonzero estimate and truth counts."
        )
    costs = np.asarray([
        [np.linalg.norm(estimate[:3] - truth[:3]) for truth in truths]
        for estimate in estimates
    ])
    rows, columns = linear_sum_assignment(costs)
    return np.asarray([
        (
            np.linalg.norm(estimates[row][:3] - truths[column][:3]),
            np.linalg.norm(estimates[row][3:] - truths[column][3:]),
        )
        for row, column in zip(rows, columns)
    ])


def _append_matched_history(
    history_by_truth,
    *,
    timestamp,
    tracks,
    truth_state_by_target,
) -> np.ndarray:
    if not tracks:
        return np.empty((0, 2), dtype=float)
    track_items = tuple(sorted(tracks.items()))
    truth_items = tuple(sorted(truth_state_by_target.items()))
    costs = np.asarray([
        [
            np.linalg.norm(track.estimate.state_eci[:3] - truth[:3])
            for _, truth in truth_items
        ]
        for _, track in track_items
    ])
    rows, columns = linear_sum_assignment(costs)
    errors = []
    for row, column in zip(rows, columns):
        truth_id = truth_items[column][0]
        estimate = track_items[row][1].estimate.state_eci
        truth = truth_items[column][1]
        history_by_truth.setdefault(truth_id, []).append(
            (timestamp, estimate.copy())
        )
        errors.append((
            np.linalg.norm(estimate[:3] - truth[:3]),
            np.linalg.norm(estimate[3:] - truth[3:]),
        ))
    return np.asarray(errors, dtype=float)


def _final_acceleration_errors(
    estimate_history_by_truth,
    truth_history_by_target,
    timestamps,
) -> np.ndarray:
    truth_index_by_timestamp = {
        float(timestamp): index for index, timestamp in enumerate(timestamps)
    }
    errors = []
    for target_id, estimate_history in estimate_history_by_truth.items():
        if len(estimate_history) < 2:
            continue
        (previous_time, previous), (current_time, current) = estimate_history[-2:]
        dt = current_time - previous_time
        previous_truth = truth_history_by_target[target_id][
            truth_index_by_timestamp[previous_time]
        ]
        current_truth = truth_history_by_target[target_id][
            truth_index_by_timestamp[current_time]
        ]
        estimated_acceleration = (current[3:] - previous[3:]) / dt
        truth_acceleration = (current_truth[3:] - previous_truth[3:]) / dt
        errors.append(np.linalg.norm(estimated_acceleration - truth_acceleration))
    if not errors:
        raise ValueError("At least two tracked epochs are required for acceleration.")
    return np.asarray(errors)


def _acceleration_errors_by_timestamp(
    estimate_history_by_truth,
    truth_history_by_target,
    timestamps,
) -> dict[float, np.ndarray]:
    truth_index_by_timestamp = {
        float(timestamp): index for index, timestamp in enumerate(timestamps)
    }
    errors_by_timestamp: dict[float, list[float]] = {}
    for target_id, estimate_history in estimate_history_by_truth.items():
        for (previous_time, previous), (current_time, current) in zip(
            estimate_history, estimate_history[1:]
        ):
            dt = current_time - previous_time
            previous_truth = truth_history_by_target[target_id][
                truth_index_by_timestamp[previous_time]
            ]
            current_truth = truth_history_by_target[target_id][
                truth_index_by_timestamp[current_time]
            ]
            estimated = (current[3:] - previous[3:]) / dt
            truth = (current_truth[3:] - previous_truth[3:]) / dt
            errors_by_timestamp.setdefault(current_time, []).append(
                float(np.linalg.norm(estimated - truth))
            )
    return {
        timestamp: np.asarray(errors, dtype=float)
        for timestamp, errors in errors_by_timestamp.items()
    }


def _phase_metrics(
    config,
    state_errors_by_timestamp,
    acceleration_errors_by_timestamp,
) -> dict[str, BilateralPhaseMetrics]:
    phase_bounds = _configured_phase_bounds(config)
    if phase_bounds is None:
        selectors = {"FULL_RUN": lambda _timestamp: True}
    else:
        start, end = phase_bounds
        selectors = {
            "PRE_DROPOUT": lambda timestamp: timestamp < start,
            "DROPOUT": lambda timestamp: start <= timestamp <= end,
            "RECOVERY": lambda timestamp: timestamp > end,
        }
    result = {}
    for name, selected in selectors.items():
        timestamps = tuple(
            timestamp for timestamp in state_errors_by_timestamp
            if selected(timestamp) and state_errors_by_timestamp[timestamp].size
        )
        state_values = [state_errors_by_timestamp[value] for value in timestamps]
        states = (
            np.concatenate(state_values, axis=0)
            if state_values else np.empty((0, 2), dtype=float)
        )
        acceleration_values = [
            errors for timestamp, errors in acceleration_errors_by_timestamp.items()
            if selected(timestamp) and errors.size
        ]
        accelerations = (
            np.concatenate(acceleration_values)
            if acceleration_values else np.empty(0, dtype=float)
        )
        result[name] = BilateralPhaseMetrics(
            epoch_count=len(timestamps),
            position_sample_count=len(states),
            acceleration_sample_count=len(accelerations),
            position_rmse_m=_rmse_or_nan(states[:, 0]),
            velocity_rmse_mps=_rmse_or_nan(states[:, 1]),
            acceleration_rmse_mps2=_rmse_or_nan(accelerations),
        )
    return result


def _rmse_or_nan(values) -> float:
    array = np.asarray(values, dtype=float)
    return (
        float(np.sqrt(np.mean(np.square(array))))
        if array.size else float("nan")
    )


def _strip_truth_identity(message, detection_group: str):
    metadata = {
        key: value
        for key, value in message.metadata.items()
        if "target" not in str(key).lower()
    }
    return UnlabeledObservationMessage(
        message_id=f"{detection_group}:{message.modality.lower()}",
        observer_id=message.observer_id,
        timestamp=message.timestamp,
        modality=message.modality,
        measurement=message.measurement,
        covariance=message.covariance,
        detection_group_id=detection_group,
        confidence=message.confidence,
        valid_flag=message.valid_flag,
        metadata=metadata,
    )


def _drop_message(config, *, timestamp, target_index, modality) -> bool:
    if config.dropout_modality is None:
        return False
    if not (
        config.dropout_start_seconds
        <= timestamp <= config.dropout_end_seconds
    ):
        return False
    if (
        config.dropout_target_indices is not None
        and target_index not in config.dropout_target_indices
    ):
        return False
    return config.dropout_modality in {"ALL", str(modality).upper()}


def _observer_failed(config, *, timestamp, observer_index) -> bool:
    if config.failed_observer_indices is None:
        return False
    return bool(
        config.observer_failure_start_seconds
        <= timestamp <= config.observer_failure_end_seconds
        and observer_index in config.failed_observer_indices
    )


def _configured_phase_bounds(config):
    if config.dropout_start_seconds is not None:
        return config.dropout_start_seconds, config.dropout_end_seconds
    if config.observer_failure_start_seconds is not None:
        return (
            config.observer_failure_start_seconds,
            config.observer_failure_end_seconds,
        )
    return None


def _active_observer_ids(frontend, *, timestamp):
    config = frontend.config
    failed = set()
    if (
        config.failed_observer_indices is not None
        and config.observer_failure_start_seconds
        <= timestamp <= config.observer_failure_end_seconds
    ):
        failed = {
            frontend.truth.manifest.friendly_observer_ids[index]
            for index in config.failed_observer_indices
        }
    return tuple(
        node_id
        for node_id in frontend.truth.manifest.friendly_observer_ids
        if node_id not in failed
    )


def _topology_with_inactive_nodes(base, *, inactive_node_ids):
    inactive = {str(value) for value in inactive_node_ids}
    unknown = inactive - set(base.node_ids)
    if unknown:
        raise ValueError(f"Inactive nodes are absent from topology: {sorted(unknown)}")
    return NetworkTopology({
        node_id: (
            ()
            if node_id in inactive
            else tuple(
                neighbor
                for neighbor in base.neighbors(node_id)
                if neighbor not in inactive
            )
        )
        for node_id in base.node_ids
    })


def _undirected_edge_count(topology):
    return sum(
        len(topology.neighbors(node_id)) for node_id in topology.node_ids
    ) // 2


def _rng_by_modality(seed, epoch_index, observer_index, target_index):
    root = np.random.SeedSequence(
        [int(seed), int(epoch_index), int(observer_index), int(target_index)]
    )
    return {
        modality: np.random.default_rng(child)
        for modality, child in zip(MODALITIES, root.spawn(len(MODALITIES)))
    }


def _sensor_config(image_size: int) -> MultimodalSensorSimulationConfig:
    common = {
        "width": int(image_size),
        "height": int(image_size),
        "read_noise_sigma": 1.0,
    }
    return MultimodalSensorSimulationConfig(
        optical=OpticalCameraConfig(**common),
        infrared=InfraredCameraConfig(**common),
        radar=RadarRangeDopplerConfig(**common),
    )


def _target_pointing_quaternion(relative_position) -> np.ndarray:
    forward = np.asarray(relative_position, dtype=float).reshape(3)
    norm = float(np.linalg.norm(forward))
    if norm <= 0.0:
        raise ValueError("Target-pointing attitude requires nonzero separation.")
    forward /= norm
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(forward @ reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    lateral = np.cross(reference, forward)
    lateral /= np.linalg.norm(lateral)
    vertical = np.cross(forward, lateral)
    return dcm_to_quat_wxyz(np.column_stack((forward, lateral, vertical)).T)
