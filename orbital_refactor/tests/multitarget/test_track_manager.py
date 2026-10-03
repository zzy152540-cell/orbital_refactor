import numpy as np
import pytest

from tracking import (
    GlobalTargetEstimate,
    MultiTargetOutput,
    TargetInitialState,
    TrackLifecycle,
    TrackManager,
)


def _initial(target_id, *, track_id=None, offset=0.0):
    return TargetInitialState(
        target_id=target_id,
        track_id=track_id or f"track-{target_id}",
        timestamp=0.0,
        state_eci=np.array([7.0e6 + offset, 0.0, 0.0, 0.0, 7500.0, 0.0]),
        covariance_eci=np.eye(6),
    )


def _estimate(initial, timestamp, observer="observer_a"):
    return GlobalTargetEstimate(
        target_id=initial.target_id,
        track_id=initial.track_id,
        timestamp=timestamp,
        state_eci=initial.state_eci,
        covariance_eci=initial.covariance_eci,
        contributing_observer_ids=(observer,),
        node_weights={observer: 1.0},
    )


def _output(timestamp, estimates):
    return MultiTargetOutput(
        scene_id="scene",
        timestamp=timestamp,
        estimates_by_target={value.target_id: value for value in estimates},
    )


def test_tracks_transition_from_initializing_to_tracking_independently():
    first = _initial("target_01")
    second = _initial("target_02", offset=1000.0)
    manager = TrackManager(
        scene_id="scene",
        initial_states={first.target_id: first, second.target_id: second},
    )

    tracks = manager.step(_output(0.0, [_estimate(first, 0.0)]))

    assert tracks["target_01"].lifecycle == TrackLifecycle.TRACKING
    assert tracks["target_02"].lifecycle == TrackLifecycle.COASTING
    assert tracks["target_02"].estimate.contributing_observer_ids == ()


def test_missing_target_coasts_then_becomes_lost_and_can_be_reacquired():
    initial = _initial("target_01")
    manager = TrackManager(
        scene_id="scene",
        initial_states={initial.target_id: initial},
        max_coast_epochs=2,
    )
    manager.step(_output(0.0, [_estimate(initial, 0.0)]))

    first = manager.step(_output(1.0, []))["target_01"]
    second = manager.step(_output(2.0, []))["target_01"]
    lost = manager.step(_output(3.0, []))["target_01"]

    assert first.lifecycle == TrackLifecycle.COASTING
    assert second.lifecycle == TrackLifecycle.COASTING
    assert lost.lifecycle == TrackLifecycle.LOST
    assert lost.estimate.timestamp == 3.0
    assert np.trace(lost.estimate.covariance_eci) > np.trace(initial.covariance_eci)

    reacquired = manager.step(
        _output(4.0, [_estimate(initial, 4.0, observer="observer_b")])
    )["target_01"]
    assert reacquired.lifecycle == TrackLifecycle.TRACKING
    assert reacquired.estimate.contributing_observer_ids == ("observer_b",)


def test_manager_rejects_unknown_target_track_change_and_old_epoch():
    initial = _initial("target_01")
    manager = TrackManager(
        scene_id="scene", initial_states={initial.target_id: initial},
    )
    unknown = _initial("target_02")
    with pytest.raises(ValueError, match="unregistered"):
        manager.step(_output(0.0, [_estimate(unknown, 0.0)]))

    changed = _initial("target_01", track_id="different-track")
    with pytest.raises(ValueError, match="changed track_id"):
        manager.step(_output(0.0, [_estimate(changed, 0.0)]))

    manager.step(_output(0.0, [_estimate(initial, 0.0)]))
    with pytest.raises(ValueError, match="strictly increasing"):
        manager.step(_output(0.0, [_estimate(initial, 0.0)]))


def test_terminated_track_ignores_later_missing_epochs():
    initial = _initial("target_01")
    manager = TrackManager(
        scene_id="scene", initial_states={initial.target_id: initial},
    )
    terminated = manager.terminate("target_01")
    tracks = manager.step(_output(1.0, []))

    assert terminated.lifecycle == TrackLifecycle.TERMINATED
    assert tracks["target_01"].lifecycle == TrackLifecycle.TERMINATED
    assert tracks["target_01"].estimate.timestamp == 0.0


def test_two_targets_coast_and_reacquire_on_independent_timelines():
    first = _initial("target_01")
    second = _initial("target_02", offset=2000.0)
    manager = TrackManager(
        scene_id="scene",
        initial_states={"target_01": first, "target_02": second},
        max_coast_epochs=1,
    )
    manager.step(_output(0.0, [
        _estimate(first, 0.0, observer="observer_a"),
        _estimate(second, 0.0, observer="observer_b"),
    ]))

    epoch_1 = manager.step(_output(1.0, [
        _estimate(first, 1.0, observer="observer_c"),
    ]))
    assert epoch_1["target_01"].lifecycle == TrackLifecycle.TRACKING
    assert epoch_1["target_02"].lifecycle == TrackLifecycle.COASTING
    epoch_2 = manager.step(_output(2.0, [
        _estimate(first, 2.0, observer="observer_c"),
    ]))
    assert epoch_2["target_01"].lifecycle == TrackLifecycle.TRACKING
    assert epoch_2["target_02"].lifecycle == TrackLifecycle.LOST
    np.testing.assert_allclose(epoch_2["target_01"].estimate.state_eci, first.state_eci)

    epoch_3 = manager.step(_output(3.0, [
        _estimate(second, 3.0, observer="observer_a"),
    ]))
    assert epoch_3["target_01"].lifecycle == TrackLifecycle.COASTING
    assert epoch_3["target_02"].lifecycle == TrackLifecycle.TRACKING
    assert epoch_3["target_02"].estimate.contributing_observer_ids == (
        "observer_a",
    )
    assert not np.shares_memory(
        epoch_3["target_01"].estimate.state_eci,
        epoch_3["target_02"].estimate.state_eci,
    )


def test_prediction_covariance_is_target_local_during_partial_outage():
    first = _initial("target_01")
    second = _initial("target_02", offset=1000.0)
    manager = TrackManager(
        scene_id="scene",
        initial_states={"target_01": first, "target_02": second},
        process_noise_acceleration=1e-3,
    )
    manager.step(_output(0.0, [
        _estimate(first, 0.0), _estimate(second, 0.0),
    ]))

    tracks = manager.step(_output(10.0, [_estimate(first, 10.0)]))

    np.testing.assert_allclose(
        tracks["target_01"].estimate.covariance_eci,
        first.covariance_eci,
    )
    assert np.trace(tracks["target_02"].estimate.covariance_eci) > np.trace(
        second.covariance_eci
    )


def test_lost_track_terminates_and_can_be_removed_from_active_set():
    initial = _initial("target_01")
    manager = TrackManager(
        scene_id="scene",
        initial_states={initial.target_id: initial},
        max_coast_epochs=1,
        max_lost_epochs=2,
    )
    manager.step(_output(0.0, [_estimate(initial, 0.0)]))

    assert manager.step(_output(1.0, []))["target_01"].lifecycle is TrackLifecycle.COASTING
    assert manager.step(_output(2.0, []))["target_01"].lifecycle is TrackLifecycle.LOST
    assert manager.step(_output(3.0, []))["target_01"].lifecycle is TrackLifecycle.LOST
    assert manager.step(_output(4.0, []))["target_01"].lifecycle is TrackLifecycle.TERMINATED

    removed = manager.remove_terminated()

    assert tuple(removed) == ("target_01",)
    assert manager.tracks == {}
