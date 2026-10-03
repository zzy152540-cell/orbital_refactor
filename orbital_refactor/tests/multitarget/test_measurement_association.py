import numpy as np

from orbital_core.constants import R_EARTH
from orbital_core.dynamics import propagate_absolute_orbit, rk4_step_absolute
from orbital_core.measurements import measure_relative_range, measure_relative_range_rate
from orbital_core.orbit_elements import keplerian_to_eci
from orbital_core.coordinates import build_rtn_quaternion, dcm_to_quat_wxyz
from tracking import (
    AutonomousIODManager,
    MultiTargetAssociationPipeline,
    OnlineMultiTargetTracker,
    TargetInitialState,
    UnlabeledIODObservation,
    UnlabeledObservationMessage,
    associate_to_tracks,
    label_matches,
    unlabeled_iod_observations_from_messages,
    run_unlabeled_tracking_sequence,
)
from tracking.track_lifecycle import TrackLifecycle


def _state(anomaly_deg, inclination_deg=55.0, altitude_m=700e3):
    return keplerian_to_eci(
        R_EARTH + altitude_m, 0.001, np.deg2rad(inclination_deg),
        np.deg2rad(15.0), 0.0, np.deg2rad(anomaly_deg),
    )


def _unlabeled(observer_id, timestamp, observer, target, *, group):
    relative = target[:3] - observer[:3]
    return (
        UnlabeledIODObservation(
            timestamp=timestamp, observer_id=observer_id, modality="LOS",
            observer_state_eci=observer,
            measurement=relative / np.linalg.norm(relative),
            covariance=np.diag([2e-5, 2e-5]) ** 2,
            detection_group_id=group,
        ),
        UnlabeledIODObservation(
            timestamp=timestamp, observer_id=observer_id, modality="RADAR",
            observer_state_eci=observer,
            measurement=np.array([
                measure_relative_range(observer, target),
                measure_relative_range_rate(observer, target),
            ]),
            covariance=np.diag([10.0, 0.05]) ** 2,
            detection_group_id=group,
        ),
    )


def _camera_quaternion(relative_position):
    forward = np.asarray(relative_position, dtype=float)
    forward /= np.linalg.norm(forward)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(forward @ reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    lateral = np.cross(reference, forward)
    lateral /= np.linalg.norm(lateral)
    vertical = np.cross(forward, lateral)
    return dcm_to_quat_wxyz(np.column_stack((forward, lateral, vertical)).T)


def test_gnn_labels_unordered_measurements_without_using_target_ids():
    observer = _state(-2.0)
    first = _state(0.5)
    second = _state(3.5)
    tracks = {
        "target-a": TargetInitialState(
            "target-a", "track-a", 0.0, first, np.diag([100.0] * 3 + [1.0] * 3),
        ),
        "target-b": TargetInitialState(
            "target-b", "track-b", 0.0, second, np.diag([100.0] * 3 + [1.0] * 3),
        ),
    }
    observations = (
        _unlabeled("observer-1", 0.0, observer, second, group="blob-2")[0],
        _unlabeled("observer-1", 0.0, observer, first, group="blob-1")[0],
    )

    result = associate_to_tracks(observations, tracks)
    labeled = label_matches(observations, result)

    assert [item.target_id for item in labeled] == ["target-b", "target-a"]
    assert result.unassigned_observation_indices == ()
    assert result.ambiguous_observation_indices == ()
    assert result.unobserved_target_ids == ()
    assert all(
        item.metadata["associationSquaredMahalanobis"] < 1e-8
        for item in labeled
    )


def test_gnn_leaves_out_of_gate_false_alarm_unassigned():
    observer = _state(-2.0)
    target = _state(0.5)
    track = TargetInitialState(
        "target-a", "track-a", 0.0, target,
        np.diag([100.0] * 3 + [1.0] * 3),
    )
    false_line = np.array([0.0, 0.0, 1.0])
    observation = UnlabeledIODObservation(
        timestamp=0.0, observer_id="observer-1", modality="LOS",
        observer_state_eci=observer, measurement=false_line,
        covariance=np.diag([2e-5, 2e-5]) ** 2,
    )

    result = associate_to_tracks((observation,), {"target-a": track})

    assert result.matches == ()
    assert result.unassigned_observation_indices == (0,)
    assert result.ambiguous_observation_indices == ()
    assert result.unobserved_target_ids == ("target-a",)


def test_unassigned_multimodal_detections_create_two_independent_iod_tracks():
    times = np.array([0.0, 10.0, 20.0])
    target_initials = (_state(0.5), _state(3.5))
    observer_initials = {
        "observer-1": _state(-1.0, 54.7, 698e3),
        "observer-2": _state(-0.5, 54.9, 702e3),
    }
    target_histories = [propagate_absolute_orbit(value, times) for value in target_initials]
    observer_histories = {
        key: propagate_absolute_orbit(value, times)
        for key, value in observer_initials.items()
    }
    manager = AutonomousIODManager(spatial_gate_m=100_000.0)

    update = None
    for epoch_index, timestamp in enumerate(times):
        observations = []
        for observer_id, observer_history in observer_histories.items():
            for target_index, target_history in enumerate(target_histories):
                observations.extend(_unlabeled(
                    observer_id, timestamp, observer_history[epoch_index],
                    target_history[epoch_index],
                    group=f"{observer_id}-epoch-{epoch_index}-blob-{target_index}",
                ))
        update = manager.ingest(observations)

    assert update is not None
    assert set(update.initialized_states_by_target) == {
        "target-auto-0001", "target-auto-0002",
    }
    estimates = sorted(
        update.initialized_states_by_target.values(),
        key=lambda item: item.state_eci[1],
    )
    truths = sorted(target_initials, key=lambda item: item[1])
    assert all(
        np.linalg.norm(estimate.state_eci[:3] - truth[:3]) < 2.0
        for estimate, truth in zip(estimates, truths)
    )


def test_pipeline_routes_known_tracks_and_keeps_false_alarm_for_initiation():
    observer = _state(-2.0)
    known = _state(0.5)
    false_target = _state(4.0)
    initial = TargetInitialState(
        "target-known", "track-known", 0.0, known,
        np.diag([100.0] * 3 + [1.0] * 3),
    )
    known_observation = _unlabeled(
        "observer-1", 0.0, observer, known, group="known-blob",
    )[0]
    unmatched = _unlabeled(
        "observer-1", 0.0, observer, false_target, group="new-blob",
    )[0]
    pipeline = MultiTargetAssociationPipeline()

    update = pipeline.step(
        (unmatched, known_observation), {"target-known": initial},
    )

    assert tuple(update.observations_by_target) == ("target-known",)
    assert update.observations_by_target["target-known"][0].measurement is not None
    assert update.association.unassigned_observation_indices == (0,)
    assert update.autonomous_iod.initialized_states_by_target == {}


def test_target_free_raw_messages_feed_the_association_pipeline():
    observer = _state(-2.0)
    target = _state(0.5)
    relative = target[:3] - observer[:3]
    group = "observer-1-epoch-0-blob-7"
    messages = (
        UnlabeledObservationMessage(
            message_id="radar-7", observer_id="observer-1", timestamp=0.0,
            modality="RADAR",
            measurement=np.array([
                measure_relative_range(observer, target),
                measure_relative_range_rate(observer, target),
            ]),
            covariance=np.diag([10.0, 0.05]) ** 2,
            detection_group_id=group,
        ),
        UnlabeledObservationMessage(
            message_id="optical-7", observer_id="observer-1", timestamp=0.0,
            modality="OPTICAL", measurement=np.zeros(2),
            covariance=np.diag([2e-5, 2e-5]) ** 2,
            detection_group_id=group,
            metadata={"quaternion_i2b_wxyz": _camera_quaternion(relative)},
        ),
    )

    observations = unlabeled_iod_observations_from_messages(
        messages,
        observer_state_by_epoch={("observer-1", 0.0): observer},
    )

    assert not hasattr(messages[0], "target_id")
    assert [item.modality for item in observations] == ["RADAR", "LOS"]
    assert all(item.detection_group_id == group for item in observations)
    assert np.dot(observations[1].measurement, relative / np.linalg.norm(relative)) > 0.999999


def test_missing_target_measurements_are_reported_without_identity_switch():
    observer = _state(-2.0)
    first = _state(0.5)
    second = _state(1.0)
    tracks = {
        target_id: TargetInitialState(
            target_id, f"track-{target_id}", 0.0, state,
            np.diag([100.0] * 3 + [1.0] * 3),
        )
        for target_id, state in (("a", first), ("b", second))
    }
    only_second = _unlabeled(
        "observer-1", 0.0, observer, second, group="visible-b",
    )[0]

    result = associate_to_tracks((only_second,), tracks)

    assert result.matches[0].target_id == "b"
    assert result.unobserved_target_ids == ("a",)


def test_joint_radar_los_groups_keep_one_identity_for_close_targets():
    observer = _state(-2.0)
    first = _state(0.50)
    second = _state(0.56)
    tracks = {
        target_id: TargetInitialState(
            target_id, f"track-{target_id}", 0.0, state,
            np.diag([100.0] * 3 + [1.0] * 3),
        )
        for target_id, state in (("a", first), ("b", second))
    }
    first_los, first_radar = _unlabeled(
        "observer-1", 0.0, observer, first, group="blob-first",
    )
    second_los, second_radar = _unlabeled(
        "observer-1", 0.0, observer, second, group="blob-second",
    )
    observations = (second_radar, first_los, first_radar, second_los)

    result = associate_to_tracks(observations, tracks)

    assigned = {
        observations[match.observation_index].detection_group_id: match.target_id
        for match in result.matches
    }
    assert assigned == {"blob-first": "a", "blob-second": "b"}
    assert len(result.matches) == 4
    assert result.ambiguous_observation_indices == ()


def test_ambiguous_joint_detection_coasts_instead_of_starting_duplicate_track():
    observer = _state(-2.0)
    middle = _state(0.50)
    first = middle.copy()
    second = middle.copy()
    first[1] -= 1.0
    second[1] += 1.0
    covariance = np.diag([1.0e8] * 3 + [1.0e4] * 3)
    tracks = {
        "a": TargetInitialState("a", "track-a", 0.0, first, covariance),
        "b": TargetInitialState("b", "track-b", 0.0, second, covariance),
    }
    observations = _unlabeled(
        "observer-1", 0.0, observer, middle, group="ambiguous-blob",
    )
    pipeline = MultiTargetAssociationPipeline(minimum_cost_margin=2.0)

    update = pipeline.step(observations, tracks)

    assert update.association.matches == ()
    assert update.association.ambiguous_observation_indices == (0, 1)
    assert update.autonomous_iod.waiting_target_ids == ()
    assert pipeline.autonomous_iod_manager.candidate_ids == ()


def test_prediction_preserves_track_identity_after_two_targets_cross():
    observer = _state(-2.0)
    first = np.array([7.0e6, -1000.0, 0.0, 0.0, 7900.0, 0.0])
    second = np.array([7.0e6, 1000.0, 0.0, 0.0, 7500.0, 0.0])
    first_after = rk4_step_absolute(first, 10.0)
    second_after = rk4_step_absolute(second, 10.0)
    assert first_after[1] > second_after[1]
    tracks = {
        "a": TargetInitialState(
            "a", "track-a", 0.0, first,
            np.diag([10.0] * 3 + [0.1] * 3),
        ),
        "b": TargetInitialState(
            "b", "track-b", 0.0, second,
            np.diag([10.0] * 3 + [0.1] * 3),
        ),
    }
    first_values = _unlabeled(
        "observer-1", 10.0, observer, first_after, group="post-cross-a",
    )
    second_values = _unlabeled(
        "observer-1", 10.0, observer, second_after, group="post-cross-b",
    )

    result = associate_to_tracks(
        (second_values[0], first_values[1], first_values[0], second_values[1]),
        tracks,
    )

    values = (second_values[0], first_values[1], first_values[0], second_values[1])
    assigned = {
        values[match.observation_index].detection_group_id: match.target_id
        for match in result.matches
    }
    assert assigned == {"post-cross-a": "a", "post-cross-b": "b"}


def test_target_free_sequence_runs_iod_filter_fusion_coast_and_reacquisition():
    times = np.arange(0.0, 51.0, 10.0)
    target_initials = (_state(0.5), _state(3.5))
    observer_initials = {
        "observer-1": _state(-1.0, 54.7, 698e3),
        "observer-2": _state(-0.5, 54.9, 702e3),
    }
    target_histories = [propagate_absolute_orbit(value, times) for value in target_initials]
    observer_histories = {
        key: propagate_absolute_orbit(value, times)
        for key, value in observer_initials.items()
    }
    observer_states = {}
    quaternions = {}
    observations = []
    for epoch_index, timestamp in enumerate(times):
        for observer_id, history in observer_histories.items():
            observer = history[epoch_index]
            observer_states[(observer_id, timestamp)] = observer
            quaternions[(observer_id, timestamp)] = build_rtn_quaternion(observer)
            for target_index, target_history in enumerate(target_histories):
                if timestamp == 40.0 and target_index == 0:
                    continue
                epoch_values = _unlabeled(
                    observer_id, timestamp, observer, target_history[epoch_index],
                    group=f"{observer_id}-{epoch_index}-{target_index}",
                )
                observations.extend((
                    epoch_values[1],
                    UnlabeledIODObservation(
                        timestamp=epoch_values[0].timestamp,
                        observer_id=epoch_values[0].observer_id,
                        modality=epoch_values[0].modality,
                        observer_state_eci=epoch_values[0].observer_state_eci,
                        measurement=epoch_values[0].measurement,
                        covariance=epoch_values[0].covariance,
                        detection_group_id=epoch_values[0].detection_group_id,
                        metadata={"sourceModality": "INFRARED"},
                    ),
                ))

    result = run_unlabeled_tracking_sequence(
        scene_id="autonomous-two-targets",
        observations=observations,
        observer_state_by_epoch=observer_states,
        q_eci2pri_by_epoch=quaternions,
        max_coast_epochs=2,
    )

    assert set(result.initialized_states_by_target) == {
        "target-auto-0001", "target-auto-0002",
    }
    assert np.array_equal(result.tracking_history.timestamps, [30.0, 40.0, 50.0])
    lifecycle_values = {
        target_id: [track.lifecycle for track in history]
        for target_id, history in result.tracking_history.track_history_by_target.items()
    }
    assert any(values[1] is TrackLifecycle.COASTING for values in lifecycle_values.values())
    assert all(values[-1] is TrackLifecycle.TRACKING for values in lifecycle_values.values())
    assert set(result.tracking_history.output_by_epoch[-1].estimates_by_target) == set(
        result.initialized_states_by_target
    )


def test_online_tracker_persists_state_one_epoch_at_a_time():
    times = np.arange(0.0, 51.0, 10.0)
    target_initials = (_state(0.5), _state(3.5))
    observer_initials = {
        "observer-1": _state(-1.0, 54.7, 698e3),
        "observer-2": _state(-0.5, 54.9, 702e3),
    }
    target_histories = [propagate_absolute_orbit(value, times) for value in target_initials]
    observer_histories = {
        key: propagate_absolute_orbit(value, times)
        for key, value in observer_initials.items()
    }
    tracker = OnlineMultiTargetTracker(
        scene_id="online-autonomous-two-targets", max_coast_epochs=2,
    )
    updates = []
    for epoch_index, timestamp in enumerate(times):
        observer_states = {
            observer_id: history[epoch_index]
            for observer_id, history in observer_histories.items()
        }
        quaternions = {
            observer_id: build_rtn_quaternion(state)
            for observer_id, state in observer_states.items()
        }
        observations = []
        for observer_id, observer in observer_states.items():
            for target_index, target_history in enumerate(target_histories):
                if timestamp == 40.0 and target_index == 0:
                    continue
                los, radar = _unlabeled(
                    observer_id, timestamp, observer, target_history[epoch_index],
                    group=f"{observer_id}-{epoch_index}-{target_index}",
                )
                observations.extend((
                    radar,
                    UnlabeledIODObservation(
                        timestamp=los.timestamp, observer_id=los.observer_id,
                        modality=los.modality,
                        observer_state_eci=los.observer_state_eci,
                        measurement=los.measurement, covariance=los.covariance,
                        detection_group_id=los.detection_group_id,
                        metadata={"sourceModality": "INFRARED"},
                    ),
                ))
        updates.append(tracker.step(
            timestamp=timestamp,
            observations=observations,
            observer_states_eci=observer_states,
            q_eci2pri_by_observer=quaternions,
        ))

    assert set(updates[2].initialized_states_by_target) == {
        "target-auto-0001", "target-auto-0002",
    }
    assert updates[3].output is not None
    assert len(updates[3].output.estimates_by_target) == 2
    assert any(
        track.lifecycle is TrackLifecycle.COASTING
        for track in updates[4].tracks_by_target.values()
    )
    assert all(
        track.lifecycle is TrackLifecycle.TRACKING
        for track in updates[5].tracks_by_target.values()
    )


def test_retired_autonomous_target_can_reappear_with_a_new_identity():
    times = np.array([0.0, 10.0, 20.0, 200.0, 210.0, 220.0])
    target_history = propagate_absolute_orbit(_state(0.5), times)
    observer_initials = {
        "observer-1": _state(-1.0, 54.7, 698e3),
        "observer-2": _state(-0.5, 54.9, 702e3),
    }
    observer_histories = {
        key: propagate_absolute_orbit(value, times)
        for key, value in observer_initials.items()
    }
    manager = AutonomousIODManager(
        spatial_gate_m=100_000.0, candidate_timeout_seconds=60.0,
    )
    first_update = None
    for epoch_index in range(3):
        observations = []
        for observer_id, history in observer_histories.items():
            observations.extend(_unlabeled(
                observer_id, times[epoch_index], history[epoch_index],
                target_history[epoch_index],
                group=f"first-{observer_id}-{epoch_index}",
            ))
        first_update = manager.ingest(observations)
    assert first_update is not None
    assert tuple(first_update.initialized_states_by_target) == ("target-auto-0001",)

    manager.retire_target("target-auto-0001")
    second_update = None
    for epoch_index in range(3, 6):
        observations = []
        for observer_id, history in observer_histories.items():
            observations.extend(_unlabeled(
                observer_id, times[epoch_index], history[epoch_index],
                target_history[epoch_index],
                group=f"second-{observer_id}-{epoch_index}",
            ))
        second_update = manager.ingest(observations)

    assert second_update is not None
    assert tuple(second_update.initialized_states_by_target) == ("target-auto-0002",)

