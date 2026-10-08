"""End-to-end multi-target, multi-modal, visibility-aware acceptance scenario."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from adapters.multimodal_sensor_simulator import simulate_multimodal_sensor_epoch
from cooperative.topology import fully_connected_topology
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
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt, dtype=float)
    if timestamps.size < 2:
        raise ValueError("The scenario requires at least two epochs.")

    observer_initials = {
        "observer-1": _orbit_state(-2.0, 54.7, 698e3),
        "observer-2": _orbit_state(-0.5, 54.9, 702e3),
        "observer-3": _orbit_state(-3.0, 55.1, 696e3),
    }
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
        topology=fully_connected_topology(observer_ids),
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
    seed_sequence = np.random.SeedSequence(int(random_seed))
    pair_count = len(observer_ids) * len(target_ids)
    epoch_sequences = seed_sequence.spawn(timestamps.size * pair_count)

    last_update = None
    for epoch_index, timestamp in enumerate(timestamps):
        observer_states = {
            key: value[epoch_index] for key, value in observer_histories.items()
        }
        observations_by_node = {key: [] for key in observer_ids}
        pair_index = 0
        for observer_id in observer_ids:
            observer_state = observer_states[observer_id]
            for target_id in target_ids:
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
                    epoch_index * pair_count + pair_index
                ].spawn(3)
                pair_index += 1
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
            for target_id, estimate in estimates.items():
                contributor_counts[(float(timestamp), node_id, target_id)] = len(
                    estimate.contributing_observer_ids
                )
                covariance_psd_failures += int(
                    np.linalg.eigvalsh(estimate.covariance_eci).min() <= 0.0
                )
        feedback_count += sum(
            len(value.applied_observer_target_keys)
            for value in last_update.feedback_by_node.values()
        )

    assert last_update is not None
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
