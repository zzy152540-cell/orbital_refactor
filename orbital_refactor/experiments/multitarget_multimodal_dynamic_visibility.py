"""End-to-end multi-target, multi-modal, visibility-aware acceptance scenario."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from adapters.multimodal_sensor_simulator import simulate_multimodal_sensor_epoch
from cooperative.message_transport import MessageChannel
from cooperative.topology import (
    chain_topology,
    fully_connected_topology,
    ring_topology,
)
from orbital_core.constants import R_EARTH
from orbital_core.coordinates import build_rtn_quaternion, dcm_to_quat_wxyz
from orbital_core.dynamics import propagate_absolute_orbit
from orbital_core.orbit_elements import keplerian_to_eci
from scenarios.measurement_visibility import (
    VisibilityConfig,
    evaluate_inter_satellite_visibility,
)
from tracking import (
    DistributedTargetConsensus,
    DistributedTargetNodeNetwork,
    OnlineMultiTargetTracker,
    TargetInitialState,
    UnlabeledObservationMessage,
    unlabeled_iod_observations_from_messages,
)


@dataclass(frozen=True)
class MultitargetMultimodalVisibilityResult:
    timestamps: tuple[float, ...]
    observer_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    opportunity_count_by_modality: dict[str, int]
    visible_count_by_modality: dict[str, int]
    visibility_switch_count_by_modality: dict[str, int]
    valid_measurement_count_by_modality: dict[str, int]
    consensus_contributor_count_by_epoch_node_target: dict[
        tuple[float, str, str], int
    ]
    final_position_error_m_by_node_target: dict[tuple[str, str], float]
    final_velocity_error_mps_by_node_target: dict[tuple[str, str], float]
    feedback_application_count: int
    covariance_psd_failure_count: int
    elapsed_seconds: float
    received_message_count: int
    rejected_message_count: int
    recovered_source_event_count: int
    missing_target_estimate_count: int
    outage_epoch_count: int
    pending_message_count_final: int
    maximum_position_error_m: float
    maximum_velocity_error_mps: float
    topology_switch_count: int
    final_target_retention_fraction: float
    final_consensus_availability_fraction: float
    maximum_consecutive_missing_epochs: int
    outage_recovery_delay_seconds: float | None

    @property
    def maximum_final_position_error_m(self) -> float:
        return max(self.final_position_error_m_by_node_target.values(), default=0.0)

    @property
    def maximum_final_velocity_error_mps(self) -> float:
        return max(self.final_velocity_error_mps_by_node_target.values(), default=0.0)


def run_multitarget_multimodal_dynamic_visibility(
    *,
    duration: float = 600.0,
    dt: float = 10.0,
    random_seed: int = 0,
    observer_count: int = 3,
    communication_delay_seconds: float = 0.0,
    packet_loss_rate: float = 0.0,
    outage_node_id: str | None = None,
    outage_start: float | None = None,
    outage_end: float | None = None,
    topology: str = "fully_connected",
    topology_switch_time: float | None = None,
    topology_after_switch: str | None = None,
    visibility_by_modality: dict[str, VisibilityConfig] | None = None,
) -> MultitargetMultimodalVisibilityResult:
    """Run raw three-modal measurements through independent node trackers.

    The scenario deliberately uses different maximum ranges per modality. The
    geometry evolves under the common orbital propagator, so measurements enter
    and leave the local filters through the physical visibility layer rather
    than through a scripted dropout mask. Ten seconds is the validated default
    filter step; larger steps remain useful for visibility-only scans but can
    exceed the present nonlinear association/filter stability envelope.
    """

    if duration <= 0.0 or dt <= 0.0:
        raise ValueError("duration and dt must be positive.")
    if not 2 <= int(observer_count) <= 10:
        raise ValueError("observer_count must be between 2 and 10.")
    if communication_delay_seconds < 0.0:
        raise ValueError("communication_delay_seconds cannot be negative.")
    if not 0.0 <= packet_loss_rate <= 1.0:
        raise ValueError("packet_loss_rate must lie in [0, 1].")
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt, dtype=float)
    if timestamps.size < 2:
        raise ValueError("The scenario requires at least two epochs.")

    observer_initials = _observer_initial_states(int(observer_count))
    target_initials = {
        "target-01": _orbit_state(0.5, 55.0, 700e3),
        "target-02": _orbit_state(3.5, 55.3, 705e3),
    }
    observer_histories = {
        key: propagate_absolute_orbit(value, timestamps)
        for key, value in observer_initials.items()
    }
    target_histories = {
        key: propagate_absolute_orbit(value, timestamps)
        for key, value in target_initials.items()
    }
    limits = visibility_by_modality or {
        "RADAR": VisibilityConfig(maximum_range=675e3),
        "INFRARED": VisibilityConfig(
            maximum_range=492e3,
            field_of_view_half_angle=np.deg2rad(5.0),
        ),
        "OPTICAL": VisibilityConfig(
            maximum_range=307e3,
            field_of_view_half_angle=np.deg2rad(3.0),
        ),
    }
    expected_modalities = {"RADAR", "INFRARED", "OPTICAL"}
    if set(limits) != expected_modalities:
        raise ValueError(
            "visibility_by_modality must define RADAR, INFRARED, and OPTICAL."
        )

    observer_ids = tuple(observer_initials)
    target_ids = tuple(target_initials)
    if outage_node_id is not None and outage_node_id not in observer_ids:
        raise ValueError("outage_node_id must identify a configured observer.")
    if (outage_start is None) != (outage_end is None):
        raise ValueError("outage_start and outage_end must be supplied together.")
    if outage_start is not None and float(outage_end) < float(outage_start):
        raise ValueError("outage_end cannot precede outage_start.")
    if (topology_switch_time is None) != (topology_after_switch is None):
        raise ValueError(
            "topology_switch_time and topology_after_switch must be supplied together."
        )
    initial_covariance = np.diag([100.0**2] * 3 + [0.5**2] * 3)
    trackers = {
        observer_id: OnlineMultiTargetTracker(
            scene_id="multitarget-multimodal-dynamic-visibility",
            initial_states={
                target_id: TargetInitialState(
                    target_id=target_id,
                    track_id=f"track-{target_id}",
                    timestamp=0.0,
                    state_eci=target_initials[target_id],
                    covariance_eci=initial_covariance,
                )
                for target_id in target_ids
            },
            max_coast_epochs=max(3, timestamps.size),
            max_lost_epochs=max(6, 2 * timestamps.size),
        )
        for observer_id in observer_ids
    }
    consensus = DistributedTargetConsensus(
        scene_id="multitarget-multimodal-dynamic-visibility",
        topology=_build_topology(topology, observer_ids),
        message_channel=MessageChannel(
            packet_loss_rate={
                observer_id: float(packet_loss_rate)
                for observer_id in observer_ids
            },
            delay_by_source={
                observer_id: float(communication_delay_seconds)
                for observer_id in observer_ids
            },
            random_seed=int(random_seed) + 91_003,
        ),
    )
    network = DistributedTargetNodeNetwork(
        trackers_by_node=trackers,
        consensus=consensus,
    )

    opportunity_counts = defaultdict(int)
    visible_counts = defaultdict(int)
    valid_measurement_counts = defaultdict(int)
    visibility_history = defaultdict(list)
    contributor_counts = {}
    feedback_count = 0
    covariance_psd_failures = 0
    received_message_count = 0
    rejected_message_count = 0
    recovered_source_event_count = 0
    missing_target_estimate_count = 0
    outage_epoch_count = 0
    maximum_position_error = 0.0
    maximum_velocity_error = 0.0
    topology_switch_count = 0
    topology_switched = False
    missing_streak_by_node_target = defaultdict(int)
    maximum_consecutive_missing_epochs = 0
    outage_recovery_delay_seconds = None
    seed_sequence = np.random.SeedSequence(int(random_seed))
    pair_count = len(observer_ids) * len(target_ids)
    epoch_sequences = seed_sequence.spawn(timestamps.size * pair_count)

    last_update = None
    started_at = perf_counter()
    for epoch_index, timestamp in enumerate(timestamps):
        if (
            not topology_switched
            and topology_switch_time is not None
            and float(timestamp) >= float(topology_switch_time)
        ):
            consensus.update_topology(
                _build_topology(str(topology_after_switch), observer_ids)
            )
            topology_switch_count += 1
            topology_switched = True
        inactive_node = (
            outage_node_id
            if outage_node_id is not None
            and outage_start is not None
            and float(outage_start) <= float(timestamp) <= float(outage_end)
            else None
        )
        if inactive_node is not None:
            outage_epoch_count += 1
        active_observer_ids = tuple(
            value for value in observer_ids if value != inactive_node
        )
        observer_states = {
            key: observer_histories[key][epoch_index]
            for key in active_observer_ids
        }
        observations_by_node = {key: [] for key in active_observer_ids}
        for observer_index, observer_id in enumerate(observer_ids):
            if observer_id not in observer_states:
                continue
            observer_state = observer_states[observer_id]
            for target_index, target_id in enumerate(target_ids):
                target_state = target_histories[target_id][epoch_index]
                camera_quaternion = _target_pointing_quaternion(
                    target_state[:3] - observer_state[:3]
                )
                valid_by_modality = {}
                for modality in sorted(expected_modalities):
                    visibility = evaluate_inter_satellite_visibility(
                        observer_state,
                        target_state,
                        limits[modality],
                        quaternion_i2b_wxyz=(
                            camera_quaternion
                            if modality in {"INFRARED", "OPTICAL"}
                            else None
                        ),
                    )
                    opportunity_counts[modality] += 1
                    visible_counts[modality] += int(visibility.visible)
                    visibility_history[(observer_id, target_id, modality)].append(
                        visibility.visible
                    )
                    valid_by_modality[modality] = visibility.visible

                modality_sequences = epoch_sequences[
                    epoch_index * pair_count
                    + observer_index * len(target_ids)
                    + target_index
                ].spawn(3)
                sensor_epoch = simulate_multimodal_sensor_epoch(
                    timestamp=timestamp,
                    observer_id=observer_id,
                    target_id=target_id,
                    observer_state=observer_state,
                    target_state=target_state,
                    quaternion_i2b_wxyz=camera_quaternion,
                    rng_by_modality={
                        modality: np.random.default_rng(sequence)
                        for modality, sequence in zip(
                            ("RADAR", "INFRARED", "OPTICAL"),
                            modality_sequences,
                            strict=True,
                        )
                    },
                    valid_by_modality=valid_by_modality,
                )
                group_id = f"{observer_id}:{target_id}:{timestamp:.9f}"
                raw_messages = tuple(
                    UnlabeledObservationMessage(
                        message_id=message.information_id,
                        observer_id=message.observer_id,
                        timestamp=message.timestamp,
                        modality=message.modality,
                        measurement=message.measurement,
                        covariance=message.covariance,
                        detection_group_id=group_id,
                        confidence=message.confidence,
                        valid_flag=message.valid_flag,
                        metadata=message.metadata,
                    )
                    for message in sensor_epoch.messages
                )
                for message in raw_messages:
                    valid_measurement_counts[message.modality] += int(
                        message.valid_flag
                    )
                observations_by_node[observer_id].extend(
                    unlabeled_iod_observations_from_messages(
                        raw_messages,
                        observer_state_by_epoch={
                            (observer_id, float(timestamp)): observer_state,
                        },
                    )
                )

        last_update = network.step(
            timestamp=timestamp,
            observations_by_node={
                key: tuple(value) for key, value in observations_by_node.items()
            },
            observer_states_eci=observer_states,
            q_eci2pri_by_observer={
                key: build_rtn_quaternion(value)
                for key, value in observer_states.items()
            },
        )
        for node_id, estimates in last_update.consensus.estimates_by_node.items():
            if node_id in active_observer_ids:
                missing_target_estimate_count += len(target_ids) - len(estimates)
                for target_id in target_ids:
                    key = (node_id, target_id)
                    if target_id in estimates:
                        missing_streak_by_node_target[key] = 0
                    else:
                        missing_streak_by_node_target[key] += 1
                        maximum_consecutive_missing_epochs = max(
                            maximum_consecutive_missing_epochs,
                            missing_streak_by_node_target[key],
                        )
            for target_id, estimate in estimates.items():
                contributor_counts[(float(timestamp), node_id, target_id)] = len(
                    estimate.contributing_observer_ids
                )
                covariance_psd_failures += int(
                    np.linalg.eigvalsh(estimate.covariance_eci).min() <= 0.0
                )
                truth = target_histories[target_id][epoch_index]
                maximum_position_error = max(
                    maximum_position_error,
                    float(np.linalg.norm(estimate.state_eci[:3] - truth[:3])),
                )
                maximum_velocity_error = max(
                    maximum_velocity_error,
                    float(np.linalg.norm(estimate.state_eci[3:] - truth[3:])),
                )
        received_message_count += sum(
            len(values)
            for values in last_update.consensus.received_message_ids_by_node.values()
        )
        rejected_message_count += sum(
            len(values)
            for values in last_update.consensus.rejected_message_ids_by_node.values()
        )
        recovered_source_event_count += sum(
            len(values)
            for values in last_update.consensus.recovered_source_ids_by_node.values()
        )
        feedback_count += sum(
            len(value.applied_observer_target_keys)
            for value in last_update.feedback_by_node.values()
        )
        if (
            outage_node_id is not None
            and outage_end is not None
            and outage_recovery_delay_seconds is None
            and float(timestamp) > float(outage_end)
            and outage_node_id in last_update.feedback_by_node
        ):
            applied = {
                target_id
                for observer_id, target_id in (
                    last_update.feedback_by_node[outage_node_id]
                    .applied_observer_target_keys
                )
                if observer_id == outage_node_id
            }
            if applied == set(target_ids):
                outage_recovery_delay_seconds = (
                    float(timestamp) - float(outage_end)
                )

    assert last_update is not None
    elapsed_seconds = perf_counter() - started_at
    final_position_errors = {}
    final_velocity_errors = {}
    for node_id, estimates in last_update.consensus.estimates_by_node.items():
        for target_id, estimate in estimates.items():
            truth = target_histories[target_id][-1]
            final_position_errors[(node_id, target_id)] = float(
                np.linalg.norm(estimate.state_eci[:3] - truth[:3])
            )
            final_velocity_errors[(node_id, target_id)] = float(
                np.linalg.norm(estimate.state_eci[3:] - truth[3:])
            )
    final_consensus_availability_fraction = (
        len(final_position_errors) / (len(observer_ids) * len(target_ids))
    )
    final_target_retention_fraction = sum(
        target_id in last_update.tracking_by_node[node_id].tracks_by_target
        for node_id in observer_ids
        for target_id in target_ids
    ) / (len(observer_ids) * len(target_ids))

    switch_counts = defaultdict(int)
    for (_observer_id, _target_id, modality), values in visibility_history.items():
        switch_counts[modality] += sum(
            left != right for left, right in zip(values, values[1:])
        )
    return MultitargetMultimodalVisibilityResult(
        timestamps=tuple(map(float, timestamps)),
        observer_ids=observer_ids,
        target_ids=target_ids,
        opportunity_count_by_modality=dict(sorted(opportunity_counts.items())),
        visible_count_by_modality=dict(sorted(visible_counts.items())),
        visibility_switch_count_by_modality=dict(sorted(switch_counts.items())),
        valid_measurement_count_by_modality=dict(
            sorted(valid_measurement_counts.items())
        ),
        consensus_contributor_count_by_epoch_node_target=contributor_counts,
        final_position_error_m_by_node_target=final_position_errors,
        final_velocity_error_mps_by_node_target=final_velocity_errors,
        feedback_application_count=feedback_count,
        covariance_psd_failure_count=covariance_psd_failures,
        elapsed_seconds=elapsed_seconds,
        received_message_count=received_message_count,
        rejected_message_count=rejected_message_count,
        recovered_source_event_count=recovered_source_event_count,
        missing_target_estimate_count=missing_target_estimate_count,
        outage_epoch_count=outage_epoch_count,
        pending_message_count_final=last_update.consensus.pending_message_count,
        maximum_position_error_m=maximum_position_error,
        maximum_velocity_error_mps=maximum_velocity_error,
        topology_switch_count=topology_switch_count,
        final_target_retention_fraction=final_target_retention_fraction,
        final_consensus_availability_fraction=(
            final_consensus_availability_fraction
        ),
        maximum_consecutive_missing_epochs=maximum_consecutive_missing_epochs,
        outage_recovery_delay_seconds=outage_recovery_delay_seconds,
    )


def _orbit_state(anomaly_deg: float, inclination_deg: float, altitude_m: float):
    return keplerian_to_eci(
        R_EARTH + altitude_m,
        0.001,
        np.deg2rad(inclination_deg),
        np.deg2rad(15.0),
        0.0,
        np.deg2rad(anomaly_deg),
    )


def _observer_initial_states(count: int) -> dict[str, np.ndarray]:
    definitions = (
        (-2.0, 54.7, 698e3),
        (-0.5, 54.9, 702e3),
        (-3.0, 55.1, 696e3),
        (-1.2, 55.2, 704e3),
        (-2.6, 54.8, 700e3),
        (-0.8, 55.4, 699e3),
        (-3.4, 54.6, 703e3),
        (-2.1, 54.75, 699e3),
        (-2.4, 54.85, 701e3),
        (-3.1, 55.0, 698e3),
    )
    return {
        f"observer-{index + 1}": _orbit_state(*definitions[index])
        for index in range(count)
    }


def _build_topology(name: str, node_ids: tuple[str, ...]):
    builders = {
        "fully_connected": fully_connected_topology,
        "chain": chain_topology,
        "ring": ring_topology,
    }
    normalized = str(name).lower()
    try:
        builder = builders[normalized]
    except KeyError as exc:
        raise ValueError(
            "topology must be 'fully_connected', 'chain', or 'ring'."
        ) from exc
    return builder(node_ids)


def _target_pointing_quaternion(relative_position: np.ndarray) -> np.ndarray:
    forward = np.asarray(relative_position, dtype=float)
    forward /= np.linalg.norm(forward)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(forward @ reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    lateral = np.cross(reference, forward)
    lateral /= np.linalg.norm(lateral)
    vertical = np.cross(forward, lateral)
    return dcm_to_quat_wxyz(np.column_stack((forward, lateral, vertical)).T)
