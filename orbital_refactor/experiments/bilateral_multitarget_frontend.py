"""Raw three-modality frontend for the bilateral 15-friendly/15-hostile case.

The simulation/evaluation side knows which truth target generated a detection.
The algorithm-facing messages deliberately do not: identity is retained only
in a separate truth lookup used by tests and later scoring.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
import hashlib
import time
from types import MappingProxyType
from typing import Mapping

import numpy as np

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
from tracking.association.candidate_manager import AutonomousIODManager
from tracking.association.online_tracker import OnlineMultiTargetTracker
from tracking.association.pipeline import MultiTargetAssociationPipeline
from tracking.distributed_target_consensus import DistributedTargetConsensus
from tracking.initial_orbit.multi_target_initializer import MultiTargetIODManager

from experiments.multitarget_identity_evaluation import (
    IdentityAwareTrackEvaluator,
)


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
    backup_radial_range_sigma_m: float | None = None
    backup_radial_range_rate_sigma_mps: float | None = None
    backup_radial_observer_indices: tuple[int, ...] | None = None
    independent_radial_backup_channel: bool = False
    cann_process_radial_backup: bool = False
    prefer_complete_radial_lineage: bool = False
    consensus_exchange_hops: int = 1
    dropout_modality: str | None = None
    dropout_start_seconds: float | None = None
    dropout_end_seconds: float | None = None
    dropout_target_indices: tuple[int, ...] | None = None
    observer_failure_start_seconds: float | None = None
    observer_failure_end_seconds: float | None = None
    failed_observer_indices: tuple[int, ...] | None = None
    association_position_sigma_floor_m: float = 50.0
    association_velocity_sigma_floor_mps: float = 0.25
    association_gate_squared_mahalanobis: float = 13.815510557964274
    association_minimum_cost_margin: float = 2.0
    tracking_max_coast_epochs: int = 3
    tracking_max_lost_epochs: int = 30
    maximum_track_count: int | None = 15
    process_noise_acceleration: float = 1e-4
    preserve_angular_radial_nullspace: bool = True
    use_multi_observer_bearing_update: bool = True
    use_unified_multimodal_measurement_update: bool = False
    use_local_unified_multimodal_measurement_update: bool = False
    use_inverse_depth_los_update: bool = False
    unified_multimodal_window_epochs: int = 1
    multi_epoch_bearing_window_epochs: int = 1

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
        for field_name in (
            "backup_radial_range_sigma_m",
            "backup_radial_range_rate_sigma_mps",
        ):
            raw_value = getattr(self, field_name)
            if raw_value is None:
                continue
            value = float(raw_value)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be positive and finite.")
            object.__setattr__(self, field_name, value)
        if self.backup_radial_observer_indices is not None:
            indices = tuple(dict.fromkeys(
                int(value) for value in self.backup_radial_observer_indices
            ))
            if any(value < 0 for value in indices):
                raise ValueError(
                    "backup_radial_observer_indices must be nonnegative."
                )
            object.__setattr__(self, "backup_radial_observer_indices", indices)
        if self.independent_radial_backup_channel and (
            self.backup_radial_range_sigma_m is None
            or self.backup_radial_range_rate_sigma_mps is None
        ):
            raise ValueError(
                "The independent radial backup channel requires both range "
                "and range-rate uncertainty settings."
            )
        if int(self.consensus_exchange_hops) < 1:
            raise ValueError("consensus_exchange_hops must be positive.")
        object.__setattr__(
            self, "consensus_exchange_hops", int(self.consensus_exchange_hops)
        )
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
        if (
            not np.isfinite(self.association_position_sigma_floor_m)
            or self.association_position_sigma_floor_m < 0.0
        ):
            raise ValueError(
                "association_position_sigma_floor_m must be finite and nonnegative."
            )
        if (
            not np.isfinite(self.association_velocity_sigma_floor_mps)
            or self.association_velocity_sigma_floor_mps < 0.0
        ):
            raise ValueError(
                "association_velocity_sigma_floor_mps must be finite and nonnegative."
            )
        if (
            not np.isfinite(self.association_gate_squared_mahalanobis)
            or self.association_gate_squared_mahalanobis <= 0.0
        ):
            raise ValueError(
                "association_gate_squared_mahalanobis must be positive and finite."
            )
        if (
            not np.isfinite(self.association_minimum_cost_margin)
            or self.association_minimum_cost_margin < 0.0
        ):
            raise ValueError(
                "association_minimum_cost_margin must be finite and nonnegative."
            )
        if int(self.tracking_max_coast_epochs) < 0:
            raise ValueError("tracking_max_coast_epochs must be nonnegative.")
        if int(self.tracking_max_lost_epochs) < 1:
            raise ValueError("tracking_max_lost_epochs must be positive.")
        if self.maximum_track_count is not None and int(self.maximum_track_count) < 1:
            raise ValueError("maximum_track_count must be positive when supplied.")
        if (
            not np.isfinite(self.process_noise_acceleration)
            or self.process_noise_acceleration < 0.0
        ):
            raise ValueError(
                "process_noise_acceleration must be finite and nonnegative."
            )
        if int(self.multi_epoch_bearing_window_epochs) < 1:
            raise ValueError(
                "multi_epoch_bearing_window_epochs must be positive."
            )
        if int(self.unified_multimodal_window_epochs) < 1:
            raise ValueError(
                "unified_multimodal_window_epochs must be positive."
            )
        if (
            self.use_inverse_depth_los_update
            and not self.use_unified_multimodal_measurement_update
        ):
            raise ValueError(
                "use_inverse_depth_los_update requires "
                "use_unified_multimodal_measurement_update."
            )


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
    mean_local_modality_weight: Mapping[str, float]
    local_modality_weight_sample_count: Mapping[str, int]
    association_match_count: int
    association_unassigned_observation_count: int
    association_ambiguous_observation_count: int
    association_unobserved_track_count: int
    suppressed_birth_count: int
    joint_bearing_update_count: int
    joint_bearing_measurement_count: int
    unified_multimodal_update_count: int
    unified_multimodal_observation_count: int
    tracking_modality_restriction_start_seconds: float | None
    all_states_finite: bool
    phase_metrics: Mapping[str, "BilateralPhaseMetrics"]
    identity_switch_count: int
    nearest_truth_disagreement_count: int
    track_replacement_count: int
    missing_truth_sample_count: int
    worst_target_position_rmse_m: float
    position_rmse_m_by_target: Mapping[str, float]
    consensus_phase_metrics: Mapping[str, "BilateralPhaseMetrics"]
    consensus_identity_switch_count: int
    consensus_nearest_truth_disagreement_count: int
    consensus_track_replacement_count: int
    consensus_missing_truth_sample_count: int
    consensus_worst_target_position_rmse_m: float
    consensus_position_rmse_m_by_target: Mapping[str, float]
    representative_node_metrics: "BilateralNodeViewMetrics"
    independent_node_metrics: "BilateralNodeViewMetrics"
    consensus_received_message_count: int
    consensus_recovered_link_count: int
    active_communication_node_count_by_timestamp: Mapping[float, int]
    active_communication_edge_count_by_timestamp: Mapping[float, int]
    matched_local_consensus_metrics: "BilateralMatchedViewMetrics | None"
    matched_local_central_metrics: "BilateralMatchedViewMetrics"


@dataclass(frozen=True)
class BilateralPhaseMetrics:
    """Fleet-wide error summary for one temporal phase."""

    epoch_count: int
    position_sample_count: int
    acceleration_sample_count: int
    position_mean_error_m: float
    velocity_mean_error_mps: float
    acceleration_mean_error_mps2: float
    position_rmse_m: float
    velocity_rmse_mps: float
    acceleration_rmse_mps2: float


@dataclass(frozen=True)
class BilateralMatchedViewMetrics:
    """Local-versus-consensus errors on identical node/target/epoch samples."""

    local_sample_count: int
    consensus_sample_count: int
    common_sample_count: int
    local_only_sample_count: int
    consensus_only_sample_count: int
    local_position_mean_error_m: float
    consensus_position_mean_error_m: float
    position_mean_error_improvement_percent: float
    local_position_rmse_m: float
    consensus_position_rmse_m: float
    position_rmse_improvement_percent: float
    local_velocity_mean_error_mps: float
    consensus_velocity_mean_error_mps: float
    velocity_mean_error_improvement_percent: float
    local_velocity_rmse_mps: float
    consensus_velocity_rmse_mps: float
    velocity_rmse_improvement_percent: float
    local_acceleration_sample_count: int
    consensus_acceleration_sample_count: int
    common_acceleration_sample_count: int
    local_only_acceleration_sample_count: int
    consensus_only_acceleration_sample_count: int
    local_acceleration_rmse_mps2: float
    consensus_acceleration_rmse_mps2: float
    acceleration_rmse_improvement_percent: float


@dataclass(frozen=True)
class BilateralNodeViewMetrics:
    """Identity-aware accuracy for one node or unfused independent nodes."""

    node_count: int
    node_ids: tuple[str, ...]
    scored_sample_count: int
    scored_sample_key_sha256: str
    scored_truth_ids: tuple[str, ...]
    final_estimate_count: int
    phase_metrics: Mapping[str, BilateralPhaseMetrics]
    identity_switch_count: int
    nearest_truth_disagreement_count: int
    track_replacement_count: int
    missing_truth_sample_count: int
    worst_target_position_rmse_m: float
    position_rmse_m_by_target: Mapping[str, float]


@dataclass(frozen=True)
class BilateralOnlineComparisonResult:
    """Paired no-CANN/CANN arms evaluated on the same frontend frames."""

    without_cann: BilateralOnlineSmokeResult
    incremental_cann: BilateralOnlineSmokeResult
    message_count_by_timestamp: Mapping[float, int]


@dataclass(frozen=True)
class BilateralBaselineMatrixResult:
    """Comparable estimator arms sharing truth and rendered sensor frames."""

    arms: Mapping[str, BilateralOnlineSmokeResult]
    local_node_baselines: Mapping[str, BilateralNodeViewMetrics]
    matched_single_observer_cann_metrics: BilateralMatchedViewMetrics | None
    matched_single_observer_joint_cann_metrics: BilateralMatchedViewMetrics | None
    matched_single_observer_ablation_metrics: Mapping[
        str, BilateralMatchedViewMetrics
    ]
    matched_single_observer_cann_vs_modality: Mapping[
        str, BilateralMatchedViewMetrics
    ]
    matched_single_observer_joint_cann_vs_modality: Mapping[
        str, BilateralMatchedViewMetrics
    ]
    matched_independent_consensus_metrics_by_cann: Mapping[
        str, BilateralMatchedViewMetrics
    ]
    matched_joint_independent_consensus_metrics: (
        BilateralMatchedViewMetrics | None
    )
    tracking_modalities_by_arm: Mapping[str, tuple[str, ...]]
    uses_common_multimodal_initialization: bool
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
    if (
        selected.backup_radial_observer_indices
        and max(selected.backup_radial_observer_indices)
        >= len(scene_truth.manifest.friendly_observer_ids)
    ):
        raise ValueError(
            "backup_radial_observer_indices is outside the friendly roster."
        )
    sensor_config = selected.sensor_config or _sensor_config(selected.image_size)
    if (
        selected.backup_radial_range_sigma_m is not None
        or selected.backup_radial_range_rate_sigma_mps is not None
    ):
        sensor_config = replace(
            sensor_config,
            backup_radial_range_sigma_m=(
                selected.backup_radial_range_sigma_m
            ),
            backup_radial_range_rate_sigma_mps=(
                selected.backup_radial_range_rate_sigma_mps
            ),
        )
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
                for sensor_message in sensor_epoch.messages:
                    message = sensor_message
                    if (
                        selected.backup_radial_observer_indices is not None
                        and observer_index
                        not in selected.backup_radial_observer_indices
                    ):
                        message = _without_radial_backup_shadow(message)
                    separated = (
                        _separate_radial_backup_channel(message)
                        if selected.independent_radial_backup_channel
                        else (message,)
                    )
                    for separated_message in separated:
                        if _drop_message(
                            selected,
                            timestamp=timestamp,
                            target_index=target_index,
                            modality=_logical_modality(separated_message),
                        ):
                            continue
                        epoch_messages.append(
                            _strip_truth_identity(
                                separated_message, detection_group
                            )
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


def derive_bilateral_fault_frontend(
    nominal: BilateralFrontendResult,
    config: BilateralFrontendConfig,
) -> BilateralFrontendResult:
    """Apply a modality or observer fault to an already rendered stream.

    The caller must preserve the nominal orbit, sensor, noise, and radial
    backup configuration.  This operation changes only message availability,
    which guarantees that surviving observations remain byte-for-byte equal
    to the matched control.
    """

    if (
        config.duration_seconds != nominal.config.duration_seconds
        or config.sample_interval_seconds
        != nominal.config.sample_interval_seconds
        or config.random_seed != nominal.config.random_seed
    ):
        raise ValueError(
            "Derived fault frontend must share duration, sampling, and seed."
        )
    observer_ids = nominal.truth.manifest.friendly_observer_ids
    observer_index = {
        observer_id: index for index, observer_id in enumerate(observer_ids)
    }
    target_ids = nominal.truth.manifest.hostile_target_ids
    target_index = {
        target_id: index for index, target_id in enumerate(target_ids)
    }
    messages_by_timestamp = {}
    visible_by_target_epoch = {}
    for timestamp_value in nominal.truth.timestamps:
        timestamp = float(timestamp_value)
        kept = []
        for message in nominal.messages_by_timestamp[timestamp]:
            failed = _observer_failed(
                config,
                timestamp=timestamp,
                observer_index=observer_index[message.observer_id],
            )
            target_id = nominal.truth_target_by_detection_group[
                message.detection_group_id
            ]
            dropped = _drop_message(
                config,
                timestamp=timestamp,
                target_index=target_index[target_id],
                modality=_logical_modality(message),
            )
            if not failed and not dropped:
                kept.append(message)
        messages_by_timestamp[timestamp] = tuple(kept)
        for target_id in target_ids:
            visible = nominal.visible_observers_by_target_and_timestamp[
                (target_id, timestamp)
            ]
            visible_by_target_epoch[(target_id, timestamp)] = tuple(
                observer_id for observer_id in visible
                if not _observer_failed(
                    config,
                    timestamp=timestamp,
                    observer_index=observer_index[observer_id],
                )
            )
    return BilateralFrontendResult(
        config=config,
        truth=nominal.truth,
        messages_by_timestamp=MappingProxyType(messages_by_timestamp),
        observer_state_by_epoch=nominal.observer_state_by_epoch,
        truth_target_by_detection_group=(
            nominal.truth_target_by_detection_group
        ),
        visible_observers_by_target_and_timestamp=MappingProxyType(
            visible_by_target_epoch
        ),
        raw_frames_by_detection_group=nominal.raw_frames_by_detection_group,
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
    initialization_cache = {}
    return BilateralOnlineComparisonResult(
        without_cann=_run_bilateral_online_arm(
            frontend, enable_cann=False,
            iod_initialization_cache=initialization_cache,
        ),
        incremental_cann=_run_bilateral_online_arm(
            frontend, enable_cann=True,
            iod_initialization_cache=initialization_cache,
        ),
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
    initialization_cache = {}
    return BilateralOnlineComparisonResult(
        without_cann=_run_bilateral_online_arm(
            frontend, enable_cann=False, enable_consensus=True,
            iod_initialization_cache=initialization_cache,
        ),
        incremental_cann=_run_bilateral_online_arm(
            frontend, enable_cann=True, enable_consensus=True,
            iod_initialization_cache=initialization_cache,
        ),
        message_count_by_timestamp=MappingProxyType({
            timestamp: len(messages)
            for timestamp, messages in frontend.messages_by_timestamp.items()
        }),
    )


def run_bilateral_cann_acceptance_arm(
    config: BilateralFrontendConfig | None = None,
    *,
    enable_consensus: bool = False,
) -> BilateralOnlineSmokeResult:
    """Run only the CANN arm required by a frozen acceptance item."""

    selected = config or BilateralFrontendConfig(duration_seconds=4.0)
    frontend = build_bilateral_raw_frontend(selected)
    return run_bilateral_cann_acceptance_frontend(
        frontend, enable_consensus=enable_consensus,
    )


def run_bilateral_cann_acceptance_frontend(
    frontend: BilateralFrontendResult,
    *,
    enable_consensus: bool = False,
    topology_by_timestamp: Mapping[float, NetworkTopology] | None = None,
    iod_initialization_cache: dict | None = None,
    epoch_callback=None,
) -> BilateralOnlineSmokeResult:
    """Run the acceptance CANN arm on an already rendered frontend."""

    return _run_bilateral_online_arm(
        frontend,
        enable_cann=True,
        enable_consensus=enable_consensus,
        topology_by_timestamp=topology_by_timestamp,
        iod_initialization_cache=(
            {} if iod_initialization_cache is None
            else iod_initialization_cache
        ),
        epoch_callback=epoch_callback,
    )


def run_bilateral_baseline_matrix(
    config: BilateralFrontendConfig | None = None,
    *,
    primary_acceptance_only: bool = False,
    acceptance_scope: str | None = None,
    fixed_federated_weights: Mapping[str, float] | None = None,
    frontend: BilateralFrontendResult | None = None,
    topology_by_timestamp: Mapping[float, NetworkTopology] | None = None,
) -> BilateralBaselineMatrixResult:
    """Compare modality, CANN, and fixed-topology arms on identical frames.

    Autonomous cold-start IOD requires radar range plus a line of sight.  Every
    arm therefore uses the same multimodal initialization frames and applies
    its configured modality ablation only after all hostile tracks exist.
    """

    if frontend is None:
        selected = config or BilateralFrontendConfig(duration_seconds=4.0)
        frontend = build_bilateral_raw_frontend(selected)
    elif config is not None and config != frontend.config:
        raise ValueError("config must match the supplied frontend.config.")
    definitions = {
        "centralized_radar_only": (("RADAR",), False, False, False, None),
        "centralized_infrared_only": (("INFRARED",), False, False, False, None),
        "centralized_optical_only": (("OPTICAL",), False, False, False, None),
        "centralized_multimodal_no_cann": (
            MODALITIES, False, False, False, None,
        ),
        "centralized_multimodal_cann": (
            MODALITIES, True, False, False, None,
        ),
        "single_observer_joint_multimodal_no_cann": (
            MODALITIES, False, False, True, None,
        ),
        "single_observer_joint_multimodal_cann": (
            MODALITIES, True, False, True, None,
        ),
        "fixed_topology_multimodal_no_cann": (
            MODALITIES, False, True, False, None,
        ),
        "fixed_topology_multimodal_cann": (
            MODALITIES, True, True, False, None,
        ),
        "fixed_topology_joint_multimodal_cann": (
            MODALITIES, True, True, True, None,
        ),
    }
    acceptance_arm_names = {
        "p05-control": {
            "centralized_multimodal_no_cann",
        },
        "p05-candidate": {
            "centralized_multimodal_no_cann",
            "single_observer_joint_multimodal_cann",
        },
        "p05-optical-pilot": {
            "centralized_optical_only",
            "single_observer_joint_multimodal_cann",
        },
        "p05-fixed-weight-pilot": {
            "single_observer_fixed_weight_federated_no_cann",
            "single_observer_joint_multimodal_cann",
        },
        "p07-control": {
            "single_observer_joint_multimodal_cann",
        },
        "p07-candidate": {
            "single_observer_joint_multimodal_cann",
            "fixed_topology_joint_multimodal_cann",
        },
    }
    if acceptance_scope == "p05-fixed-weight-pilot":
        pilot_weights = fixed_federated_weights or {
            "rad": 0.4, "ir": 0.3, "opt": 0.3,
        }
        definitions["single_observer_fixed_weight_federated_no_cann"] = (
            MODALITIES, False, False, False,
            dict(pilot_weights),
        )
    if acceptance_scope is not None:
        try:
            required = acceptance_arm_names[acceptance_scope]
        except KeyError as exc:
            raise ValueError(
                f"Unknown bilateral acceptance scope: {acceptance_scope}"
            ) from exc
        definitions = {
            name: definition for name, definition in definitions.items()
            if name in required
        }
    elif primary_acceptance_only:
        required = {
            "centralized_multimodal_no_cann",
            "centralized_multimodal_cann",
            "single_observer_joint_multimodal_no_cann",
            "single_observer_joint_multimodal_cann",
            "fixed_topology_joint_multimodal_cann",
        }
        definitions = {
            name: definition for name, definition in definitions.items()
            if name in required
        }
    arms = {}
    sample_collectors = {}
    initialization_cache = {}
    for name, (
        modalities, enable_cann, enable_consensus, enable_local_unified,
        fixed_information_weights,
    ) in definitions.items():
        collector = {}
        arms[name] = _run_bilateral_online_arm(
            frontend,
            enable_cann=enable_cann,
            enable_consensus=enable_consensus,
            tracking_modalities=modalities,
            enable_local_unified=enable_local_unified,
            fixed_information_weights=fixed_information_weights,
            sample_collector=collector,
            iod_initialization_cache=initialization_cache,
            topology_by_timestamp=topology_by_timestamp,
        )
        sample_collectors[name] = collector
    representative_node_id = frontend.truth.manifest.friendly_observer_ids[0]
    def representative_samples(arm_name: str) -> dict:
        if arm_name not in sample_collectors:
            return {}
        return {
            key: sample
            for key, sample in sample_collectors[arm_name]["local"].items()
            if key[0] == representative_node_id
        }

    def representative_acceleration_samples(arm_name: str) -> dict:
        if arm_name not in sample_collectors:
            return {}
        return {
            key: sample
            for key, sample in sample_collectors[arm_name][
                "localAcceleration"
            ].items()
            if key[0] == representative_node_id
        }

    representative_no_cann = representative_samples(
        "centralized_multimodal_no_cann"
    )
    representative_cann = representative_samples(
        "centralized_multimodal_cann"
    )
    representative_joint_cann = representative_samples(
        "single_observer_joint_multimodal_cann"
    )
    representative_joint_no_cann = representative_samples(
        "single_observer_joint_multimodal_no_cann"
    )
    representative_fixed_weight = representative_samples(
        "single_observer_fixed_weight_federated_no_cann"
    )
    representative_no_cann_acceleration = (
        representative_acceleration_samples("centralized_multimodal_no_cann")
    )
    representative_cann_acceleration = representative_acceleration_samples(
        "centralized_multimodal_cann"
    )
    representative_joint_cann_acceleration = (
        representative_acceleration_samples(
            "single_observer_joint_multimodal_cann"
        )
    )
    representative_joint_no_cann_acceleration = (
        representative_acceleration_samples(
            "single_observer_joint_multimodal_no_cann"
        )
    )
    representative_fixed_weight_acceleration = (
        representative_acceleration_samples(
            "single_observer_fixed_weight_federated_no_cann"
        )
    )
    representative_single_modality = {
        modality: representative_samples(arm_name)
        for modality, arm_name in {
            "radar": "centralized_radar_only",
            "infrared": "centralized_infrared_only",
            "optical": "centralized_optical_only",
        }.items()
        if arm_name in sample_collectors
    }
    representative_single_modality_acceleration = {
        modality: representative_acceleration_samples(arm_name)
        for modality, arm_name in {
            "radar": "centralized_radar_only",
            "infrared": "centralized_infrared_only",
            "optical": "centralized_optical_only",
        }.items()
        if arm_name in sample_collectors
    }

    local_node_baselines = {}
    if "centralized_multimodal_no_cann" in arms:
        local_node_baselines.update({
            "single_node_multimodal_no_cann": arms[
                "centralized_multimodal_no_cann"
            ].representative_node_metrics,
            "independent_15_node_multimodal_no_cann": arms[
                "centralized_multimodal_no_cann"
            ].independent_node_metrics,
        })
    if "centralized_multimodal_cann" in arms:
        local_node_baselines.update({
            "single_node_multimodal_cann": arms[
                "centralized_multimodal_cann"
            ].representative_node_metrics,
            "independent_15_node_multimodal_cann": arms[
                "centralized_multimodal_cann"
            ].independent_node_metrics,
        })
    if "single_observer_joint_multimodal_cann" in arms:
        if acceptance_scope == "p05-candidate":
            # In the lean P05 acceptance pair this is the selected CANN
            # candidate.  Keep the established summary key so older evidence
            # aggregation remains compatible without running an extra arm.
            local_node_baselines[
                "single_node_multimodal_cann"
            ] = arms[
                "single_observer_joint_multimodal_cann"
            ].representative_node_metrics
        local_node_baselines[
            "independent_15_node_joint_multimodal_cann"
        ] = arms[
            "single_observer_joint_multimodal_cann"
        ].independent_node_metrics
    if "single_observer_fixed_weight_federated_no_cann" in arms:
        local_node_baselines[
            "single_node_fixed_weight_federated_no_cann"
        ] = arms[
            "single_observer_fixed_weight_federated_no_cann"
        ].representative_node_metrics

    def matched_or_none(
        left: dict, right: dict, left_acceleration: dict,
        right_acceleration: dict,
    ):
        return (
            _matched_view_metrics(
                left, right, left_acceleration, right_acceleration,
            )
            if left and right else None
        )

    matched_ablation_pairs = {
        "federatedCannIncrement": (
            representative_no_cann, representative_cann,
            representative_no_cann_acceleration,
            representative_cann_acceleration,
        ),
        "jointFusionWithoutCann": (
            representative_no_cann, representative_joint_no_cann,
            representative_no_cann_acceleration,
            representative_joint_no_cann_acceleration,
        ),
        "jointCannIncrement": (
            representative_joint_no_cann, representative_joint_cann,
            representative_joint_no_cann_acceleration,
            representative_joint_cann_acceleration,
        ),
        "fixedWeightFederatedToJointCann": (
            representative_fixed_weight, representative_joint_cann,
            representative_fixed_weight_acceleration,
            representative_joint_cann_acceleration,
        ),
        "jointFusionWithCannCombined": (
            representative_no_cann, representative_joint_cann,
            representative_no_cann_acceleration,
            representative_joint_cann_acceleration,
        ),
    }
    matched_ablation = {
        name: _matched_view_metrics(left, right, left_acc, right_acc)
        for name, (left, right, left_acc, right_acc)
        in matched_ablation_pairs.items()
        if left and right
    }
    matched_by_cann = {}
    for label, local_arm, consensus_arm in (
        (
            "withoutCann", "centralized_multimodal_no_cann",
            "fixed_topology_multimodal_no_cann",
        ),
        (
            "incrementalCann", "centralized_multimodal_cann",
            "fixed_topology_multimodal_cann",
        ),
    ):
        if local_arm in sample_collectors and consensus_arm in sample_collectors:
            matched_by_cann[label] = _matched_view_metrics(
                sample_collectors[local_arm]["local"],
                sample_collectors[consensus_arm]["consensus"],
                sample_collectors[local_arm]["localAcceleration"],
                sample_collectors[consensus_arm]["consensusAcceleration"],
            )
    matched_joint = None
    if {
        "single_observer_joint_multimodal_cann",
        "fixed_topology_joint_multimodal_cann",
    }.issubset(sample_collectors):
        matched_joint = _matched_view_metrics(
            sample_collectors[
                "single_observer_joint_multimodal_cann"
            ]["local"],
            sample_collectors[
                "fixed_topology_joint_multimodal_cann"
            ]["consensus"],
            sample_collectors[
                "single_observer_joint_multimodal_cann"
            ]["localAcceleration"],
            sample_collectors[
                "fixed_topology_joint_multimodal_cann"
            ]["consensusAcceleration"],
        )

    return BilateralBaselineMatrixResult(
        arms=MappingProxyType(arms),
        local_node_baselines=MappingProxyType(local_node_baselines),
        matched_single_observer_cann_metrics=matched_or_none(
            representative_no_cann, representative_cann,
            representative_no_cann_acceleration,
            representative_cann_acceleration,
        ),
        matched_single_observer_joint_cann_metrics=matched_or_none(
            representative_no_cann, representative_joint_cann,
            representative_no_cann_acceleration,
            representative_joint_cann_acceleration,
        ),
        matched_single_observer_ablation_metrics=MappingProxyType(
            matched_ablation
        ),
        matched_single_observer_cann_vs_modality=MappingProxyType({
            modality: _matched_view_metrics(
                samples,
                representative_cann,
                representative_acceleration_samples({
                    "radar": "centralized_radar_only",
                    "infrared": "centralized_infrared_only",
                    "optical": "centralized_optical_only",
                }[modality]),
                representative_cann_acceleration,
            )
            for modality, samples in representative_single_modality.items()
            if representative_cann
        }),
        matched_single_observer_joint_cann_vs_modality=MappingProxyType({
            modality: _matched_view_metrics(
                samples,
                representative_joint_cann,
                representative_single_modality_acceleration[modality],
                representative_joint_cann_acceleration,
            )
            for modality, samples in representative_single_modality.items()
            if representative_joint_cann
        }),
        matched_independent_consensus_metrics_by_cann=MappingProxyType(
            matched_by_cann
        ),
        matched_joint_independent_consensus_metrics=matched_joint,
        tracking_modalities_by_arm=MappingProxyType({
            name: tuple(modalities)
            for name, (
                modalities, _cann, _consensus, _local_unified, _fixed_weights,
            ) in definitions.items()
        }),
        uses_common_multimodal_initialization=True,
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
    enable_local_unified: bool = False,
    fixed_information_weights: Mapping[str, float] | None = None,
    tracking_modalities: tuple[str, ...] | None = None,
    tracking_observer_ids: tuple[str, ...] | None = None,
    tracking_target_ids: tuple[str, ...] | None = None,
    require_scored_truth: bool = True,
    sample_collector: dict | None = None,
    epoch_callback=None,
    iod_initialization_cache: dict | None = None,
    topology_by_timestamp: Mapping[float, NetworkTopology] | None = None,
) -> BilateralOnlineSmokeResult:
    allowed_tracking_modalities = _normalized_modalities(tracking_modalities)
    allowed_tracking_observers = (
        None if tracking_observer_ids is None
        else frozenset(str(value) for value in tracking_observer_ids)
    )
    allowed_tracking_targets = (
        None if tracking_target_ids is None
        else frozenset(str(value) for value in tracking_target_ids)
    )
    known_observers = frozenset(
        frontend.truth.manifest.friendly_observer_ids
    )
    if allowed_tracking_observers is not None:
        if not allowed_tracking_observers:
            raise ValueError("tracking_observer_ids cannot be empty.")
        unknown = allowed_tracking_observers - known_observers
        if unknown:
            raise ValueError(
                f"Unknown tracking observer ids: {sorted(unknown)}"
            )
    known_targets = frozenset(frontend.truth.manifest.hostile_target_ids)
    if allowed_tracking_targets is not None:
        if not allowed_tracking_targets:
            raise ValueError("tracking_target_ids cannot be empty.")
        unknown = allowed_tracking_targets - known_targets
        if unknown:
            raise ValueError(
                f"Unknown tracking target ids: {sorted(unknown)}"
            )
    tracker = OnlineMultiTargetTracker(
        scene_id=frontend.truth.manifest.scene_id,
        process_noise_acceleration=(
            frontend.config.process_noise_acceleration
        ),
        association_pipeline=MultiTargetAssociationPipeline(
            gate_squared_mahalanobis=(
                frontend.config.association_gate_squared_mahalanobis
            ),
            minimum_cost_margin=(
                frontend.config.association_minimum_cost_margin
            ),
            prediction_position_sigma_floor_m=(
                frontend.config.association_position_sigma_floor_m
            ),
            prediction_velocity_sigma_floor_mps=(
                frontend.config.association_velocity_sigma_floor_mps
            ),
            autonomous_iod_manager=AutonomousIODManager(
                iod_manager=MultiTargetIODManager(
                    initialization_cache=iod_initialization_cache,
                )
            ),
        ),
        max_coast_epochs=frontend.config.tracking_max_coast_epochs,
        max_lost_epochs=frontend.config.tracking_max_lost_epochs,
        maximum_track_count=frontend.config.maximum_track_count,
        preserve_angular_radial_nullspace=(
            frontend.config.preserve_angular_radial_nullspace
        ),
        enable_multi_observer_bearing_update=(
            frontend.config.use_multi_observer_bearing_update
        ),
        enable_unified_multimodal_update=(
            frontend.config.use_unified_multimodal_measurement_update
        ),
        enable_local_unified_multimodal_update=(
            enable_local_unified
            or frontend.config.use_local_unified_multimodal_measurement_update
        ),
        local_fusion_method=(
            "fixed_information"
            if fixed_information_weights is not None else "ci"
        ),
        local_information_allocation_weights=fixed_information_weights,
        enable_inverse_depth_los_update=(
            frontend.config.use_inverse_depth_los_update
        ),
        unified_multimodal_window_epochs=(
            frontend.config.unified_multimodal_window_epochs
        ),
        multi_epoch_bearing_window_epochs=(
            frontend.config.multi_epoch_bearing_window_epochs
        ),
        measurement_preprocessor_factory=(
            (
                lambda _observer, _target: IncrementalMultimodalCANNPreprocessor(
                    IncrementalMultimodalCANNConfig(
                        profile="realtime",
                        radial_backup=(
                            frontend.config.cann_process_radial_backup
                        ),
                    )
                )
            )
            if enable_cann else None
        ),
    )
    base_topology = frontend.truth.manifest.friendly_topology
    if topology_by_timestamp is not None:
        expected_nodes = set(base_topology.node_ids)
        missing_epochs = {
            float(value) for value in frontend.truth.timestamps
        } - {float(value) for value in topology_by_timestamp}
        if missing_epochs:
            raise ValueError(
                "topology_by_timestamp is missing frontend epochs: "
                f"{sorted(missing_epochs)[:3]}"
            )
        for timestamp, topology in topology_by_timestamp.items():
            if set(topology.node_ids) != expected_nodes:
                raise ValueError(
                    "Dynamic topology must contain exactly the friendly nodes "
                    f"at timestamp {timestamp}."
                )
        initial_topology = topology_by_timestamp[
            float(frontend.truth.timestamps[0])
        ]
    else:
        initial_topology = base_topology
    consensus = (
        DistributedTargetConsensus(
            scene_id=frontend.truth.manifest.scene_id,
            topology=initial_topology,
            prefer_complete_radial_lineage=(
                frontend.config.prefer_complete_radial_lineage
            ),
            exchange_hops=frontend.config.consensus_exchange_hops,
        )
        if enable_consensus else None
    )
    initialized_ids: set[str] = set()
    retired_ids: set[str] = set()
    final_update = None
    processing_seconds: list[float] = []
    cann_observations = 0
    cann_actions: Counter[str] = Counter()
    local_weight_sums: Counter[str] = Counter()
    local_weight_counts: Counter[str] = Counter()
    association_matches = 0
    association_unassigned = 0
    association_ambiguous = 0
    association_unobserved_tracks = 0
    suppressed_births = 0
    joint_bearing_updates = 0
    joint_bearing_measurements = 0
    unified_multimodal_updates = 0
    unified_multimodal_observations = 0
    consensus_received = 0
    consensus_recovered = 0
    active_node_counts: dict[float, int] = {}
    active_edge_counts: dict[float, int] = {}
    truth_ids = (
        frontend.truth.manifest.hostile_target_ids
        if allowed_tracking_targets is None
        else tuple(
            target_id for target_id in frontend.truth.manifest.hostile_target_ids
            if target_id in allowed_tracking_targets
        )
    )
    central_evaluator = IdentityAwareTrackEvaluator(truth_ids)
    consensus_evaluators = {
        node_id: IdentityAwareTrackEvaluator(truth_ids)
        for node_id in frontend.truth.manifest.friendly_observer_ids
    }
    local_evaluators = {
        node_id: IdentityAwareTrackEvaluator(truth_ids)
        for node_id in frontend.truth.manifest.friendly_observer_ids
    }
    representative_node_id = frontend.truth.manifest.friendly_observer_ids[0]
    state_errors_by_timestamp: dict[float, np.ndarray] = {}
    consensus_state_errors_by_timestamp: dict[float, np.ndarray] = {}
    local_state_errors_by_timestamp: dict[float, np.ndarray] = {}
    representative_state_errors_by_timestamp: dict[float, np.ndarray] = {}
    local_samples_by_key = {}
    consensus_samples_by_key = {}
    central_samples_by_key = {}
    final_state_errors = np.empty((0, 2), dtype=float)
    final_local_estimate_count = 0
    final_representative_estimate_count = 0
    restriction_start = None
    for epoch_index, timestamp_value in enumerate(frontend.truth.timestamps):
        timestamp = float(timestamp_value)
        observer_states = {
            observer_id: frontend.observer_state_by_epoch[(observer_id, timestamp)]
            for observer_id in frontend.truth.manifest.friendly_observer_ids
        }
        epoch_messages = frontend.messages_by_timestamp[timestamp]
        if allowed_tracking_targets is not None:
            epoch_messages = tuple(
                message for message in epoch_messages
                if frontend.truth_target_by_detection_group[
                    message.detection_group_id
                ] in allowed_tracking_targets
            )
        if (
            allowed_tracking_observers is not None
            and len(initialized_ids) == len(truth_ids)
        ):
            epoch_messages = tuple(
                message for message in epoch_messages
                if message.observer_id in allowed_tracking_observers
            )
        if (
            allowed_tracking_modalities is not None
            and len(initialized_ids) == len(truth_ids)
        ):
            if (
                restriction_start is None
                and allowed_tracking_modalities != frozenset(MODALITIES)
            ):
                restriction_start = timestamp
            epoch_messages = tuple(
                message for message in epoch_messages
                if message.modality in allowed_tracking_modalities
            )
        observations = unlabeled_iod_observations_from_messages(
            epoch_messages,
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
        truth_state_by_target = {
            target_id: history[epoch_index]
            for target_id, history in (
                frontend.truth.hostile_truth_state_history_by_target.items()
            )
            if target_id in truth_ids
        }
        reports_by_node = {
            node_id: {
                report.track_id: report.state_eci
                for report in final_update.local_reports
                if report.observer_id == node_id and report.valid_flag
            }
            for node_id in local_evaluators
        }
        local_samples = []
        for node_id, evaluator in local_evaluators.items():
            node_epoch = evaluator.update(
                timestamp=timestamp,
                state_by_track_id=reports_by_node[node_id],
                truth_state_by_id=truth_state_by_target,
            )
            local_samples.extend(node_epoch.samples_by_truth.values())
            for truth_id, sample in node_epoch.samples_by_truth.items():
                local_samples_by_key[(node_id, truth_id, timestamp)] = sample
            if node_id == representative_node_id:
                representative_state_errors_by_timestamp[timestamp] = (
                    _sample_errors(node_epoch.samples_by_truth.values())
                )
        local_state_errors_by_timestamp[timestamp] = _sample_errors(local_samples)
        final_local_estimate_count = sum(map(len, reports_by_node.values()))
        final_representative_estimate_count = len(
            reports_by_node[representative_node_id]
        )
        if consensus is not None:
            epoch_base_topology = (
                topology_by_timestamp[timestamp]
                if topology_by_timestamp is not None else base_topology
            )
            active_observers = _active_observer_ids(
                frontend, timestamp=timestamp,
            )
            active_topology = _topology_with_inactive_nodes(
                epoch_base_topology,
                inactive_node_ids=(
                    set(epoch_base_topology.node_ids) - set(active_observers)
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
            consensus_samples = []
            for node_id in active_observers:
                epoch_result = consensus_evaluators[node_id].update(
                    timestamp=timestamp,
                    state_by_track_id={
                        estimate.track_id: estimate.state_eci
                        for estimate in (
                            consensus_step.estimates_by_node[node_id].values()
                        )
                    },
                    truth_state_by_id=truth_state_by_target,
                )
                consensus_samples.extend(epoch_result.samples_by_truth.values())
                for truth_id, sample in epoch_result.samples_by_truth.items():
                    consensus_samples_by_key[(
                        node_id, truth_id, timestamp,
                    )] = sample
            consensus_state_errors_by_timestamp[timestamp] = _sample_errors(
                consensus_samples
            )
        processing_seconds.append(time.perf_counter() - started)
        association_matches += len(final_update.association.matches)
        association_unassigned += len(
            final_update.association.unassigned_observation_indices
        )
        association_ambiguous += len(
            final_update.association.ambiguous_observation_indices
        )
        association_unobserved_tracks += len(
            final_update.association.unobserved_target_ids
        )
        suppressed_births += len(final_update.suppressed_birth_ids)
        joint_bearing_updates += final_update.joint_bearing_update_count
        joint_bearing_measurements += (
            final_update.joint_bearing_measurement_count
        )
        unified_multimodal_updates += (
            final_update.unified_multimodal_update_count
        )
        unified_multimodal_observations += (
            final_update.unified_multimodal_observation_count
        )
        for report in final_update.local_reports:
            for modality, weight in report.modality_weights.items():
                local_weight_sums[str(modality)] += float(weight)
                local_weight_counts[str(modality)] += 1
        cann_observations += final_update.cann_preprocessed_observation_count
        cann_actions.update(final_update.cann_action_counts)
        initialized_ids.update(final_update.initialized_states_by_target)
        retired_ids.update(final_update.retired_target_ids)
        central_epoch = central_evaluator.update(
            timestamp=timestamp,
            state_by_track_id={
                track.estimate.track_id: track.estimate.state_eci
                for track in final_update.tracks_by_target.values()
                if track.lifecycle.value != "TERMINATED"
            },
            truth_state_by_id=truth_state_by_target,
        )
        for node_id in local_evaluators:
            for truth_id, sample in central_epoch.samples_by_truth.items():
                central_samples_by_key[(node_id, truth_id, timestamp)] = sample
        final_state_errors = _sample_errors(
            central_epoch.samples_by_truth.values()
        )
        state_errors_by_timestamp[timestamp] = final_state_errors
        if epoch_callback is not None:
            epoch_callback({
                "epoch_index": epoch_index,
                "timestamp": timestamp,
                "observer_states": observer_states,
                "epoch_messages": epoch_messages,
                "tracking_observations": observations,
                "tracker_update": final_update,
                "truth_state_by_target": truth_state_by_target,
                "identity_epoch": central_epoch,
            })

    final_active_tracks = tuple(
        track for track in final_update.tracks_by_target.values()
        if track.lifecycle.value != "TERMINATED"
    )
    estimates = tuple(track.estimate.state_eci for track in final_active_tracks)
    if not final_state_errors.size and require_scored_truth:
        raise ValueError("Smoke-test scoring requires at least one tracked truth.")
    acceleration_errors = (
        _final_acceleration_errors(
            central_evaluator.state_history_by_truth,
            frontend.truth.hostile_truth_state_history_by_target,
            frontend.truth.timestamps,
        )
        if any(
            len(history) >= 2
            for history in central_evaluator.state_history_by_truth.values()
        ) else np.empty(0, dtype=float)
    )
    acceleration_errors_by_timestamp = _acceleration_errors_by_timestamp(
        central_evaluator.state_history_by_truth,
        frontend.truth.hostile_truth_state_history_by_target,
        frontend.truth.timestamps,
    )
    position_rmse_by_target = central_evaluator.position_rmse_by_truth()
    consensus_acceleration_errors_by_timestamp = (
        _combined_acceleration_errors_by_timestamp(
            consensus_evaluators,
            frontend.truth.hostile_truth_state_history_by_target,
            frontend.truth.timestamps,
        )
        if consensus is not None else {}
    )
    consensus_position_rmse_by_target = (
        _combined_position_rmse_by_truth(consensus_evaluators, truth_ids)
        if consensus is not None else {}
    )
    independent_node_metrics = _node_view_metrics(
        frontend=frontend,
        evaluators=local_evaluators,
        state_errors_by_timestamp=local_state_errors_by_timestamp,
        final_estimate_count=final_local_estimate_count,
    )
    representative_node_metrics = _node_view_metrics(
        frontend=frontend,
        evaluators={
            representative_node_id: local_evaluators[representative_node_id]
        },
        state_errors_by_timestamp=representative_state_errors_by_timestamp,
        final_estimate_count=final_representative_estimate_count,
    )
    local_acceleration_samples = _acceleration_error_samples_by_key(
        local_evaluators,
        frontend.truth.hostile_truth_state_history_by_target,
        frontend.truth.timestamps,
        expected_interval_seconds=frontend.config.sample_interval_seconds,
    )
    consensus_acceleration_samples = (
        _acceleration_error_samples_by_key(
            consensus_evaluators,
            frontend.truth.hostile_truth_state_history_by_target,
            frontend.truth.timestamps,
            expected_interval_seconds=frontend.config.sample_interval_seconds,
        )
        if consensus is not None else {}
    )
    central_acceleration = _acceleration_error_samples_by_key(
        {"central": central_evaluator},
        frontend.truth.hostile_truth_state_history_by_target,
        frontend.truth.timestamps,
        expected_interval_seconds=frontend.config.sample_interval_seconds,
    )
    central_acceleration_samples = {
        (node_id, truth_id, previous_time, current_time): error
        for (_central, truth_id, previous_time, current_time), error
        in central_acceleration.items()
        for node_id in local_evaluators
    }
    if sample_collector is not None:
        sample_collector.update({
            "local": dict(local_samples_by_key),
            "consensus": dict(consensus_samples_by_key),
            "central": dict(central_samples_by_key),
            "localAcceleration": local_acceleration_samples,
            "consensusAcceleration": consensus_acceleration_samples,
            "centralAcceleration": central_acceleration_samples,
        })
    return BilateralOnlineSmokeResult(
        frame_count=len(frontend.truth.timestamps),
        initialized_track_count=len(initialized_ids),
        final_active_track_count=len(estimates),
        retired_track_count=len(retired_ids),
        final_position_rmse_m=_rmse_or_nan(final_state_errors[:, 0]),
        final_maximum_position_error_m=(
            float(np.max(final_state_errors[:, 0]))
            if final_state_errors.size else float("nan")
        ),
        final_velocity_rmse_mps=_rmse_or_nan(final_state_errors[:, 1]),
        final_acceleration_rmse_mps2=_rmse_or_nan(acceleration_errors),
        processing_mean_ms=float(np.mean(processing_seconds) * 1_000.0),
        processing_p95_ms=float(
            np.percentile(processing_seconds, 95) * 1_000.0
        ),
        cann_preprocessed_observation_count=cann_observations,
        cann_action_counts=MappingProxyType(dict(sorted(cann_actions.items()))),
        mean_local_modality_weight=MappingProxyType({
            modality: local_weight_sums[modality] / count
            for modality, count in sorted(local_weight_counts.items())
        }),
        local_modality_weight_sample_count=MappingProxyType(
            dict(sorted(local_weight_counts.items()))
        ),
        association_match_count=association_matches,
        association_unassigned_observation_count=association_unassigned,
        association_ambiguous_observation_count=association_ambiguous,
        association_unobserved_track_count=association_unobserved_tracks,
        suppressed_birth_count=suppressed_births,
        joint_bearing_update_count=joint_bearing_updates,
        joint_bearing_measurement_count=joint_bearing_measurements,
        unified_multimodal_update_count=unified_multimodal_updates,
        unified_multimodal_observation_count=(
            unified_multimodal_observations
        ),
        tracking_modality_restriction_start_seconds=restriction_start,
        all_states_finite=bool(
            estimates and all(np.all(np.isfinite(state)) for state in estimates)
        ),
        phase_metrics=MappingProxyType(_phase_metrics(
            frontend.config,
            state_errors_by_timestamp,
            acceleration_errors_by_timestamp,
        )),
        identity_switch_count=central_evaluator.identity_switch_count,
        nearest_truth_disagreement_count=(
            central_evaluator.nearest_truth_disagreement_count
        ),
        track_replacement_count=central_evaluator.track_replacement_count,
        missing_truth_sample_count=central_evaluator.missing_truth_sample_count,
        worst_target_position_rmse_m=_worst_finite_value(
            position_rmse_by_target.values()
        ),
        position_rmse_m_by_target=MappingProxyType(position_rmse_by_target),
        consensus_phase_metrics=MappingProxyType(
            _phase_metrics(
                frontend.config,
                consensus_state_errors_by_timestamp,
                consensus_acceleration_errors_by_timestamp,
            ) if consensus is not None else {}
        ),
        consensus_identity_switch_count=sum(
            evaluator.identity_switch_count
            for evaluator in consensus_evaluators.values()
        ) if consensus is not None else 0,
        consensus_nearest_truth_disagreement_count=sum(
            evaluator.nearest_truth_disagreement_count
            for evaluator in consensus_evaluators.values()
        ) if consensus is not None else 0,
        consensus_track_replacement_count=sum(
            evaluator.track_replacement_count
            for evaluator in consensus_evaluators.values()
        ) if consensus is not None else 0,
        consensus_missing_truth_sample_count=sum(
            evaluator.missing_truth_sample_count
            for evaluator in consensus_evaluators.values()
        ) if consensus is not None else 0,
        consensus_worst_target_position_rmse_m=_worst_finite_value(
            consensus_position_rmse_by_target.values()
        ),
        consensus_position_rmse_m_by_target=MappingProxyType(
            consensus_position_rmse_by_target
        ),
        representative_node_metrics=representative_node_metrics,
        independent_node_metrics=independent_node_metrics,
        consensus_received_message_count=consensus_received,
        consensus_recovered_link_count=consensus_recovered,
        active_communication_node_count_by_timestamp=MappingProxyType(
            active_node_counts
        ),
        active_communication_edge_count_by_timestamp=MappingProxyType(
            active_edge_counts
        ),
        matched_local_consensus_metrics=(
            _matched_view_metrics(
                local_samples_by_key,
                consensus_samples_by_key,
                local_acceleration_samples,
                consensus_acceleration_samples,
            )
            if consensus is not None else None
        ),
        matched_local_central_metrics=_matched_view_metrics(
            local_samples_by_key,
            central_samples_by_key,
            local_acceleration_samples,
            central_acceleration_samples,
        ),
    )


def _matched_view_metrics(
    local_samples,
    consensus_samples,
    local_acceleration_samples=None,
    consensus_acceleration_samples=None,
):
    local_keys = set(local_samples)
    consensus_keys = set(consensus_samples)
    common_keys = tuple(sorted(local_keys & consensus_keys))
    local_position = np.asarray([
        local_samples[key].position_error_m for key in common_keys
    ], dtype=float)
    consensus_position = np.asarray([
        consensus_samples[key].position_error_m for key in common_keys
    ], dtype=float)
    local_velocity = np.asarray([
        local_samples[key].velocity_error_mps for key in common_keys
    ], dtype=float)
    consensus_velocity = np.asarray([
        consensus_samples[key].velocity_error_mps for key in common_keys
    ], dtype=float)
    local_acceleration_samples = local_acceleration_samples or {}
    consensus_acceleration_samples = consensus_acceleration_samples or {}
    local_acceleration_keys = set(local_acceleration_samples)
    consensus_acceleration_keys = set(consensus_acceleration_samples)
    common_acceleration_keys = tuple(sorted(
        local_acceleration_keys & consensus_acceleration_keys
    ))
    local_acceleration = np.asarray([
        local_acceleration_samples[key] for key in common_acceleration_keys
    ], dtype=float)
    consensus_acceleration = np.asarray([
        consensus_acceleration_samples[key] for key in common_acceleration_keys
    ], dtype=float)
    return BilateralMatchedViewMetrics(
        local_sample_count=len(local_keys),
        consensus_sample_count=len(consensus_keys),
        common_sample_count=len(common_keys),
        local_only_sample_count=len(local_keys - consensus_keys),
        consensus_only_sample_count=len(consensus_keys - local_keys),
        local_position_mean_error_m=_mean_or_nan(local_position),
        consensus_position_mean_error_m=_mean_or_nan(consensus_position),
        position_mean_error_improvement_percent=_improvement_percent(
            _mean_or_nan(local_position),
            _mean_or_nan(consensus_position),
        ),
        local_position_rmse_m=_rmse_or_nan(local_position),
        consensus_position_rmse_m=_rmse_or_nan(consensus_position),
        position_rmse_improvement_percent=_improvement_percent(
            _rmse_or_nan(local_position),
            _rmse_or_nan(consensus_position),
        ),
        local_velocity_mean_error_mps=_mean_or_nan(local_velocity),
        consensus_velocity_mean_error_mps=_mean_or_nan(consensus_velocity),
        velocity_mean_error_improvement_percent=_improvement_percent(
            _mean_or_nan(local_velocity),
            _mean_or_nan(consensus_velocity),
        ),
        local_velocity_rmse_mps=_rmse_or_nan(local_velocity),
        consensus_velocity_rmse_mps=_rmse_or_nan(consensus_velocity),
        velocity_rmse_improvement_percent=_improvement_percent(
            _rmse_or_nan(local_velocity),
            _rmse_or_nan(consensus_velocity),
        ),
        local_acceleration_sample_count=len(local_acceleration_keys),
        consensus_acceleration_sample_count=len(consensus_acceleration_keys),
        common_acceleration_sample_count=len(common_acceleration_keys),
        local_only_acceleration_sample_count=len(
            local_acceleration_keys - consensus_acceleration_keys
        ),
        consensus_only_acceleration_sample_count=len(
            consensus_acceleration_keys - local_acceleration_keys
        ),
        local_acceleration_rmse_mps2=_rmse_or_nan(local_acceleration),
        consensus_acceleration_rmse_mps2=_rmse_or_nan(
            consensus_acceleration
        ),
        acceleration_rmse_improvement_percent=_improvement_percent(
            _rmse_or_nan(local_acceleration),
            _rmse_or_nan(consensus_acceleration),
        ),
    )


def _improvement_percent(baseline, candidate):
    baseline = float(baseline)
    candidate = float(candidate)
    if not np.isfinite(baseline) or not np.isfinite(candidate) or baseline <= 0.0:
        return float("nan")
    return float((baseline - candidate) / baseline * 100.0)


def _sample_errors(samples) -> np.ndarray:
    values = tuple(samples)
    if not values:
        return np.empty((0, 2), dtype=float)
    return np.asarray([
        (sample.position_error_m, sample.velocity_error_mps)
        for sample in values
    ], dtype=float)


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


def _acceleration_error_samples_by_key(
    evaluators,
    truth_history_by_target,
    timestamps,
    *,
    expected_interval_seconds,
) -> dict[tuple[str, str, float, float], float]:
    """Return ECI acceleration errors on identity-continuous adjacent epochs."""

    truth_index_by_timestamp = {
        float(timestamp): index for index, timestamp in enumerate(timestamps)
    }
    result = {}
    for node_id, evaluator in evaluators.items():
        for target_id, estimate_history in (
            evaluator.state_history_by_truth.items()
        ):
            for (previous_time, previous), (current_time, current) in zip(
                estimate_history, estimate_history[1:]
            ):
                previous_time = float(previous_time)
                current_time = float(current_time)
                dt = current_time - previous_time
                if not np.isclose(dt, expected_interval_seconds):
                    continue
                previous_truth = truth_history_by_target[target_id][
                    truth_index_by_timestamp[previous_time]
                ]
                current_truth = truth_history_by_target[target_id][
                    truth_index_by_timestamp[current_time]
                ]
                estimated = (current[3:] - previous[3:]) / dt
                truth = (current_truth[3:] - previous_truth[3:]) / dt
                result[(
                    str(node_id), str(target_id), previous_time, current_time,
                )] = float(np.linalg.norm(estimated - truth))
    return result


def _combined_acceleration_errors_by_timestamp(
    evaluators,
    truth_history_by_target,
    timestamps,
) -> dict[float, np.ndarray]:
    combined: dict[float, list[np.ndarray]] = {}
    for evaluator in evaluators.values():
        node_errors = _acceleration_errors_by_timestamp(
            evaluator.state_history_by_truth,
            truth_history_by_target,
            timestamps,
        )
        for timestamp, errors in node_errors.items():
            combined.setdefault(timestamp, []).append(errors)
    return {
        timestamp: np.concatenate(values)
        for timestamp, values in combined.items()
        if values
    }


def _combined_position_rmse_by_truth(evaluators, truth_ids):
    result = {}
    for truth_id in truth_ids:
        errors = [
            sample.position_error_m
            for evaluator in evaluators.values()
            for sample in evaluator.error_history_by_truth.get(truth_id, ())
        ]
        result[truth_id] = _rmse_or_nan(errors)
    return result


def _node_view_metrics(
    *,
    frontend,
    evaluators,
    state_errors_by_timestamp,
    final_estimate_count,
) -> BilateralNodeViewMetrics:
    acceleration_errors = _combined_acceleration_errors_by_timestamp(
        evaluators,
        frontend.truth.hostile_truth_state_history_by_target,
        frontend.truth.timestamps,
    )
    position_rmse = _combined_position_rmse_by_truth(
        evaluators, frontend.truth.manifest.hostile_target_ids,
    )
    sample_keys = tuple(sorted(
        (node_id, sample.truth_id, float(sample.timestamp))
        for node_id, evaluator in evaluators.items()
        for samples in evaluator.error_history_by_truth.values()
        for sample in samples
    ))
    return BilateralNodeViewMetrics(
        node_count=len(evaluators),
        node_ids=tuple(evaluators),
        scored_sample_count=len(sample_keys),
        scored_sample_key_sha256=_sample_key_sha256(sample_keys),
        scored_truth_ids=tuple(sorted({key[1] for key in sample_keys})),
        final_estimate_count=int(final_estimate_count),
        phase_metrics=MappingProxyType(_phase_metrics(
            frontend.config,
            state_errors_by_timestamp,
            acceleration_errors,
        )),
        identity_switch_count=sum(
            evaluator.identity_switch_count for evaluator in evaluators.values()
        ),
        nearest_truth_disagreement_count=sum(
            evaluator.nearest_truth_disagreement_count
            for evaluator in evaluators.values()
        ),
        track_replacement_count=sum(
            evaluator.track_replacement_count
            for evaluator in evaluators.values()
        ),
        missing_truth_sample_count=sum(
            evaluator.missing_truth_sample_count
            for evaluator in evaluators.values()
        ),
        worst_target_position_rmse_m=_worst_finite_value(
            position_rmse.values()
        ),
        position_rmse_m_by_target=MappingProxyType(position_rmse),
    )


def _sample_key_sha256(sample_keys) -> str:
    serialized = "\n".join(
        f"{node_id}\x1f{truth_id}\x1f{timestamp:.17g}"
        for node_id, truth_id, timestamp in sample_keys
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _worst_finite_value(values) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return max(finite) if finite else float("nan")


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
            position_mean_error_m=_mean_or_nan(states[:, 0]),
            velocity_mean_error_mps=_mean_or_nan(states[:, 1]),
            acceleration_mean_error_mps2=_mean_or_nan(accelerations),
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


def _mean_or_nan(values) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)) if array.size else float("nan")


def _strip_truth_identity(message, detection_group: str):
    metadata = {
        key: value
        for key, value in message.metadata.items()
        if "target" not in str(key).lower()
    }
    return UnlabeledObservationMessage(
        message_id=(
            f"{detection_group}:radial_backup"
            if _logical_modality(message) == "RADIAL_BACKUP"
            else f"{detection_group}:{message.modality.lower()}"
        ),
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


def _logical_modality(message) -> str:
    return str(
        message.metadata.get("sourceModality", message.modality)
    ).upper()


def _separate_radial_backup_channel(message):
    metadata = dict(message.metadata)
    if not (
        metadata.get("rangeProxySource")
        == "INDEPENDENT_RADIAL_BACKUP_SHADOW"
        and metadata.get("rangeRateProxySource")
        == "INDEPENDENT_RADIAL_BACKUP_SHADOW"
    ):
        return (message,)
    measurement = np.array([
        float(metadata["rangeProxyM"]),
        float(metadata["rangeRateProxyMps"]),
    ])
    covariance = np.diag([
        float(metadata["rangeProxyVarianceM2"]),
        float(metadata["rangeRateProxyVarianceM2ps2"]),
    ])
    clean = _without_radial_backup_shadow(message)
    backup = replace(
        message,
        message_id=f"{message.message_id}:radial_backup",
        physical_observation_id=f"{message.information_id}:radial_backup",
        modality="RADAR",
        frame="ECI",
        measurement=measurement,
        covariance=covariance,
        confidence=1.0,
        valid_flag=True,
        metadata={
            "sensor_modality": "RADIAL_BACKUP",
            "sourceModality": "RADIAL_BACKUP",
            "measurement_type": "RANGE_RANGE_RATE",
            "measurement_component": "INDEPENDENT_RADIAL_BACKUP",
            "raw_source_type": "NUMERICAL_RADIAL_BACKUP_SHADOW",
            "rangeProxySource": "INDEPENDENT_RADIAL_BACKUP_SHADOW",
            "rangeRateProxySource": "INDEPENDENT_RADIAL_BACKUP_SHADOW",
            "rangeProxyIndependentNoiseAssumed": True,
            "rangeRateProxyIndependentNoiseAssumed": True,
            "radialBackupPhysicalValidation": False,
            "radialBackupCarrierModality": "INDEPENDENT_CHANNEL",
            "independentRadialBackupChannel": True,
        },
    )
    return clean, backup


def _without_radial_backup_shadow(message):
    metadata = dict(message.metadata)
    if metadata.get("rangeProxySource") == (
        "INDEPENDENT_RADIAL_BACKUP_SHADOW"
    ):
        for key in (
            "rangeProxyM", "rangeProxyVarianceM2", "rangeProxySource",
            "rangeProxyIndependentNoiseAssumed",
        ):
            metadata.pop(key, None)
    if metadata.get("rangeRateProxySource") == (
        "INDEPENDENT_RADIAL_BACKUP_SHADOW"
    ):
        for key in (
            "rangeRateProxyMps", "rangeRateProxyVarianceM2ps2",
            "rangeRateProxySource", "rangeRateProxyIndependentNoiseAssumed",
        ):
            metadata.pop(key, None)
    if not any(
        metadata.get(key) == "INDEPENDENT_RADIAL_BACKUP_SHADOW"
        for key in ("rangeProxySource", "rangeRateProxySource")
    ):
        metadata.pop("radialBackupPhysicalValidation", None)
        metadata.pop("radialBackupCarrierModality", None)
    return replace(message, metadata=metadata)


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


def _normalized_modalities(values):
    if values is None:
        return None
    normalized = tuple(dict.fromkeys(str(value).upper() for value in values))
    if not normalized:
        raise ValueError("tracking_modalities cannot be empty.")
    unsupported = set(normalized) - set(MODALITIES)
    if unsupported:
        raise ValueError(
            f"Unsupported tracking modalities: {sorted(unsupported)}"
        )
    return frozenset(normalized)


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
    stream_names = (*MODALITIES, "RADIAL_BACKUP")
    return {
        modality: np.random.default_rng(child)
        for modality, child in zip(stream_names, root.spawn(len(stream_names)))
    }


def _sensor_config(image_size: int) -> MultimodalSensorSimulationConfig:
    common = {
        "width": int(image_size),
        "height": int(image_size),
        "read_noise_sigma": 1.0,
    }
    return MultimodalSensorSimulationConfig(
        # The renderer must realize the same one-sigma uncertainty that the
        # frontend reports to the filters.  Read noise alone produces centroid
        # errors thousands of times smaller than those covariances.
        optical=OpticalCameraConfig(
            **common, pointing_jitter_sigma_pixels=0.2,
        ),
        infrared=InfraredCameraConfig(
            **common, pointing_jitter_sigma_pixels=0.25,
        ),
        radar=RadarRangeDopplerConfig(
            **common,
            range_jitter_sigma_m=2.5,
            range_rate_jitter_sigma_mps=0.0125,
        ),
    )


def electro_optical_radial_failover_sensor_config(
    image_size: int = 17,
    *,
    range_sigma_m: float = 2.5,
    range_rate_sigma_mps: float = 0.010,
) -> MultimodalSensorSimulationConfig:
    """Return the frozen R06 radar-primary electro-optical failover sensors.

    Optical and infrared observations both carry idealized radial proxy
    metadata.  The measurement filter suppresses those components while radar
    is present and admits one electro-optical proxy only when radar is absent.
    """

    return replace(
        _sensor_config(image_size),
        optical_range_proxy_sigma_m=float(range_sigma_m),
        optical_range_rate_proxy_sigma_mps=float(range_rate_sigma_mps),
        infrared_range_proxy_sigma_m=float(range_sigma_m),
        infrared_range_rate_proxy_sigma_mps=float(range_rate_sigma_mps),
        electro_optical_radial_proxy_policy="radar_primary_failover",
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
