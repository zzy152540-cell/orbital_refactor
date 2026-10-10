"""Raw unlabeled multimodal IOD-to-reacquisition acceptance scenario."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from adapters.infrared_image_adapter import InfraredCameraConfig
from adapters.multimodal_sensor_simulator import (
    MultimodalSensorSimulationConfig,
    simulate_multimodal_sensor_epoch,
)
from adapters.optical_image_adapter import OpticalCameraConfig
from adapters.radar_range_doppler_adapter import RadarRangeDopplerConfig
from cooperative.topology import fully_connected_topology
from orbital_core.constants import R_EARTH
from orbital_core.coordinates import build_rtn_quaternion, dcm_to_quat_wxyz
from orbital_core.dynamics import propagate_absolute_orbit
from orbital_core.orbit_elements import keplerian_to_eci
from tracking import (
    AutonomousIODManager,
    CooperativeTrackIdentityResolver,
    DistributedTargetConsensus,
    DistributedTargetNodeNetwork,
    IODBufferConfig,
    MultiTargetAssociationPipeline,
    MultiTargetIODManager,
    OnlineMultiTargetTracker,
    UnlabeledObservationMessage,
    unlabeled_iod_observations_from_messages,
)


@dataclass(frozen=True)
class AutonomousIdentityAcceptance:
    seed: int
    observer_count: int
    epoch_count: int
    minimum_target_separation_m: float
    initial_birth_count: int
    post_outage_birth_count: int
    identity_switch_count: int
    false_cooperative_birth_count: int
    reacquired_cooperative_target_count: int
    final_tracking_target_count: int
    final_consensus_target_count: int
    maximum_final_position_error_m: float
    passed: bool


def run_multitarget_autonomous_identity_acceptance(
    *,
    seed: int = 0,
    observer_count: int = 3,
    duration: float = 300.0,
    dt: float = 10.0,
    outage_start: float = 170.0,
    outage_end: float = 230.0,
    image_size: int = 17,
) -> AutonomousIdentityAcceptance:
    """Exercise raw frames, autonomous IOD, tracking, identity and CI together."""

    if observer_count < 2:
        raise ValueError("observer_count must be at least two.")
    if duration <= 0.0 or dt <= 0.0:
        raise ValueError("duration and dt must be positive.")
    if not 20.0 < outage_start <= outage_end < duration - 30.0:
        raise ValueError("The outage must leave IOD arcs before and after it.")
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt)
    observer_initials = {
        f"observer-{index}": _orbit_state(
            anomaly_deg=-1.5 + 0.5 * (index - 1),
            inclination_deg=54.7 + 0.15 * (index - 1),
            altitude_m=698e3 + 1_500.0 * (index - 1),
        )
        for index in range(1, observer_count + 1)
    }
    target_initials = {
        "physical-alpha": np.array([
            7.0e6, -1_000.0, 0.0, 0.0, 7_505.0, 0.0,
        ]),
        "physical-beta": np.array([
            7.0e6, 1_000.0, 0.0, 0.0, 7_495.0, 0.0,
        ]),
    }
    observer_histories = {
        key: propagate_absolute_orbit(state, timestamps)
        for key, state in observer_initials.items()
    }
    target_histories = {
        key: propagate_absolute_orbit(state, timestamps)
        for key, state in target_initials.items()
    }
    resolver = CooperativeTrackIdentityResolver(
        position_gate_m=20_000.0,
        velocity_gate_mps=50.0,
        minimum_match_margin=0.02,
        max_reacquisition_gap_seconds=duration,
    )
    scene_id = "multitarget-autonomous-identity-acceptance"
    trackers = {
        observer_id: OnlineMultiTargetTracker(
            scene_id=scene_id,
            association_pipeline=MultiTargetAssociationPipeline(
                autonomous_iod_manager=AutonomousIODManager(
                    spatial_gate_m=100_000.0,
                    bundle_clustering_gate_m=100.0,
                    candidate_timeout_seconds=40.0,
                    iod_manager=MultiTargetIODManager(
                        buffer_config=IODBufferConfig(
                            minimum_epochs=3,
                            minimum_observers=1,
                            minimum_time_span_seconds=20.0,
                            maximum_time_span_seconds=40.0,
                        ),
                    ),
                ),
            ),
            max_coast_epochs=2,
            max_lost_epochs=2,
        )
        for observer_id in observer_initials
    }
    consensus = DistributedTargetConsensus(
        scene_id=scene_id,
        topology=fully_connected_topology(tuple(observer_initials)),
    )
    network = DistributedTargetNodeNetwork(
        trackers_by_node=trackers,
        consensus=consensus,
        identity_resolver=resolver,
    )
    sensors = _sensor_config(image_size)
    expected_identity_by_physical_target: dict[str, str] = {}
    cooperative_ids_created = set()
    cooperative_ids_reacquired = set()
    initial_birth_count = 0
    post_outage_birth_count = 0
    identity_switch_count = 0
    minimum_separation = np.inf
    last_update = None

    seed_sequence = np.random.SeedSequence(int(seed))
    random_streams = seed_sequence.spawn(
        len(timestamps) * observer_count * len(target_initials) * 3
    )
    stream_index = 0
    for epoch_index, timestamp in enumerate(timestamps):
        observer_states = {
            observer_id: history[epoch_index]
            for observer_id, history in observer_histories.items()
        }
        target_states = {
            target_id: history[epoch_index]
            for target_id, history in target_histories.items()
        }
        minimum_separation = min(
            minimum_separation,
            float(np.linalg.norm(
                target_states["physical-alpha"][:3]
                - target_states["physical-beta"][:3]
            )),
        )
        observations_by_node = {observer_id: [] for observer_id in observer_states}
        in_outage = outage_start <= timestamp <= outage_end
        for node_index, (observer_id, observer_state) in enumerate(
            observer_states.items()
        ):
            for target_index, (_physical_target_id, target_state) in enumerate(
                target_states.items()
            ):
                streams = random_streams[stream_index:stream_index + 3]
                stream_index += 3
                if in_outage:
                    continue
                attitude = _target_pointing_quaternion(
                    target_state[:3] - observer_state[:3]
                )
                sensor_epoch = simulate_multimodal_sensor_epoch(
                    timestamp=timestamp,
                    observer_id=observer_id,
                    target_id="opaque-detection-source",
                    observer_state=observer_state,
                    target_state=target_state,
                    quaternion_i2b_wxyz=attitude,
                    config=sensors,
                    rng_by_modality={
                        modality: np.random.default_rng(stream)
                        for modality, stream in zip(
                            ("RADAR", "INFRARED", "OPTICAL"),
                            streams,
                            strict=True,
                        )
                    },
                )
                group_id = (
                    f"detection-{node_index}-{epoch_index}-{target_index}"
                )
                unlabeled_messages = tuple(
                    UnlabeledObservationMessage(
                        message_id=(
                            f"raw-{node_index}-{epoch_index}-{target_index}-"
                            f"{message.modality.lower()}"
                        ),
                        observer_id=observer_id,
                        timestamp=timestamp,
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
                observations_by_node[observer_id].extend(
                    unlabeled_iod_observations_from_messages(
                        unlabeled_messages,
                        observer_state_by_epoch={
                            (observer_id, float(timestamp)): observer_state,
                        },
                    )
                )

        last_update = network.step(
            timestamp=timestamp,
            observations_by_node={
                key: tuple(values) for key, values in observations_by_node.items()
            },
            observer_states_eci=observer_states,
            q_eci2pri_by_observer={
                key: build_rtn_quaternion(value)
                for key, value in observer_states.items()
            },
        )
        for tracking in last_update.tracking_by_node.values():
            births = len(tracking.initialized_states_by_target)
            if timestamp < outage_start:
                initial_birth_count += births
            elif timestamp > outage_end:
                post_outage_birth_count += births
        resolution = last_update.identity_resolution
        if resolution is None:
            continue
        cooperative_ids_created.update(resolution.created_cooperative_target_ids)
        cooperative_ids_reacquired.update(
            resolution.reacquired_cooperative_target_ids
        )
        assignment_by_node_local = {
            (item.observer_id, item.local_target_id): item.cooperative_target_id
            for item in resolution.assignments
        }
        for observer_id, tracking in last_update.tracking_by_node.items():
            current = {
                target_id: track.estimate.state_eci
                for target_id, track in tracking.tracks_by_target.items()
                if (observer_id, target_id) in assignment_by_node_local
            }
            for local_target_id, physical_target_id in _match_tracks_to_truth(
                current, target_states,
            ).items():
                cooperative_target_id = assignment_by_node_local[
                    (observer_id, local_target_id)
                ]
                expected = expected_identity_by_physical_target.setdefault(
                    physical_target_id, cooperative_target_id,
                )
                if cooperative_target_id != expected:
                    identity_switch_count += 1

    if last_update is None:
        raise RuntimeError("The scenario produced no tracking update.")
    final_tracking_count = min(
        len(tracker.tracks) for tracker in trackers.values()
    )
    final_consensus = last_update.consensus.estimates_by_node
    final_consensus_count = min(
        len(values) for values in final_consensus.values()
    )
    final_errors = []
    final_truth = {
        target_id: history[-1] for target_id, history in target_histories.items()
    }
    for estimates in final_consensus.values():
        if len(estimates) != len(final_truth):
            continue
        estimate_states = {
            target_id: estimate.state_eci for target_id, estimate in estimates.items()
        }
        matches = _match_tracks_to_truth(estimate_states, final_truth)
        final_errors.extend(
            np.linalg.norm(
                estimate_states[estimate_id][:3]
                - final_truth[physical_id][:3]
            )
            for estimate_id, physical_id in matches.items()
        )
    maximum_final_error = max(final_errors, default=np.inf)
    false_births = max(0, len(cooperative_ids_created) - len(target_initials))
    expected_local_births = observer_count * len(target_initials)
    passed = bool(
        initial_birth_count == expected_local_births
        and post_outage_birth_count == expected_local_births
        and identity_switch_count == 0
        and false_births == 0
        and len(cooperative_ids_reacquired) == len(target_initials)
        and final_tracking_count == len(target_initials)
        and final_consensus_count == len(target_initials)
        and maximum_final_error < 2_000.0
    )
    return AutonomousIdentityAcceptance(
        seed=int(seed),
        observer_count=int(observer_count),
        epoch_count=len(timestamps),
        minimum_target_separation_m=float(minimum_separation),
        initial_birth_count=initial_birth_count,
        post_outage_birth_count=post_outage_birth_count,
        identity_switch_count=identity_switch_count,
        false_cooperative_birth_count=false_births,
        reacquired_cooperative_target_count=len(cooperative_ids_reacquired),
        final_tracking_target_count=final_tracking_count,
        final_consensus_target_count=final_consensus_count,
        maximum_final_position_error_m=float(maximum_final_error),
        passed=passed,
    )


def _match_tracks_to_truth(track_states, truth_states):
    track_items = tuple(sorted(track_states.items()))
    truth_items = tuple(sorted(truth_states.items()))
    if not track_items or not truth_items:
        return {}
    costs = np.asarray([
        [
            np.linalg.norm(track_state[:3] - truth_state[:3])
            for _, truth_state in truth_items
        ]
        for _, track_state in track_items
    ])
    rows, columns = linear_sum_assignment(costs)
    return {
        track_items[row][0]: truth_items[column][0]
        for row, column in zip(rows, columns)
    }


def _orbit_state(anomaly_deg, inclination_deg, altitude_m):
    return keplerian_to_eci(
        R_EARTH + altitude_m,
        0.001,
        np.deg2rad(inclination_deg),
        np.deg2rad(15.0),
        0.0,
        np.deg2rad(anomaly_deg),
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
        "width": int(image_size),
        "height": int(image_size),
        "read_noise_sigma": 1.0,
    }
    return MultimodalSensorSimulationConfig(
        optical=OpticalCameraConfig(**common),
        infrared=InfraredCameraConfig(**common),
        radar=RadarRangeDopplerConfig(**common),
    )
