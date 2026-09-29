import numpy as np
from dataclasses import replace

from orbital_core.constants import R_EARTH
from orbital_core.dynamics import propagate_absolute_orbit
from orbital_core.coordinates import build_rtn_quaternion_history, state_history_eci_to_spri
from orbital_core.measurements import measure_relative_range, measure_relative_range_rate
from orbital_core.orbit_elements import keplerian_to_eci
from adapters.synthetic_measurement_adapter import (
    create_infrared_observations,
    create_radar_observations,
)
from tracking import (
    IODObservation,
    IODStatus,
    IODBufferConfig,
    MultiTargetIODManager,
    initialize_target,
    target_initial_state_from_iod,
    build_tracking_handoff,
    run_known_target_sequence,
)


def _state(anomaly_deg, inclination_deg, altitude_m):
    return keplerian_to_eci(
        R_EARTH + altitude_m,
        0.001,
        np.deg2rad(inclination_deg),
        np.deg2rad(15.0),
        0.0,
        np.deg2rad(anomaly_deg),
    )


def _observations():
    times = np.array([0.0, 10.0, 20.0])
    target = _state(0.6, 55.2, 703e3)
    observers = {
        "observer-1": _state(0.0, 54.8, 699e3),
        "observer-2": _state(0.2, 54.9, 701e3),
        "observer-3": _state(-0.2, 54.7, 698e3),
    }
    target_history = propagate_absolute_orbit(target, times)
    observer_histories = {
        key: propagate_absolute_orbit(value, times)
        for key, value in observers.items()
    }
    result = []
    for index, timestamp in enumerate(times):
        for observer_id, history in observer_histories.items():
            observer = history[index]
            target_epoch = target_history[index]
            relative = target_epoch[:3] - observer[:3]
            line = relative / np.linalg.norm(relative)
            result.extend((
                IODObservation(
                    timestamp=timestamp,
                    observer_id=observer_id,
                    target_id="target-1",
                    modality="LOS",
                    observer_state_eci=observer,
                    measurement=line,
                    covariance=np.diag([2e-5, 2e-5]) ** 2,
                ),
                IODObservation(
                    timestamp=timestamp,
                    observer_id=observer_id,
                    target_id="target-1",
                    modality="RADAR",
                    observer_state_eci=observer,
                    measurement=np.array([
                        measure_relative_range(observer, target_epoch),
                        measure_relative_range_rate(observer, target_epoch),
                    ]),
                    covariance=np.diag([10.0, 0.05]) ** 2,
                ),
            ))
    return target, tuple(result)


def test_multimodal_short_arc_iod_recovers_j2000_state_and_covariance():
    truth, observations = _observations()

    estimate = initialize_target(
        observations,
        target_id="target-1",
        track_id="enemy-track-1",
    )

    assert estimate.status is IODStatus.SUCCESS
    assert estimate.converged
    assert np.linalg.norm(estimate.state_eci[:3] - truth[:3]) < 1.0
    assert np.linalg.norm(estimate.state_eci[3:] - truth[3:]) < 0.01
    assert estimate.diagnostics["jacobianRank"] == 6
    assert np.linalg.eigvalsh(estimate.covariance_eci).min() >= -1e-10
    initial = target_initial_state_from_iod(estimate)
    assert initial.track_id == "enemy-track-1"


def test_iod_rejects_an_unobservable_single_epoch():
    _truth, observations = _observations()
    one_epoch = tuple(item for item in observations if item.timestamp == 0.0)

    estimate = initialize_target(one_epoch, target_id="target-1")

    assert estimate.status is IODStatus.INSUFFICIENT_OBSERVATIONS
    assert not estimate.converged


def test_multi_target_manager_waits_for_short_arc_then_initializes_each_target():
    _truth, target_1 = _observations()
    target_2 = tuple(replace(item, target_id="target-2") for item in target_1)
    manager = MultiTargetIODManager(
        track_id_by_target={
            "target-1": "enemy-track-1",
            "target-2": "enemy-track-2",
        },
        buffer_config=IODBufferConfig(
            minimum_epochs=3,
            minimum_observers=2,
            minimum_time_span_seconds=20.0,
            maximum_time_span_seconds=30.0,
        ),
    )

    for timestamp in (0.0, 10.0):
        update = manager.ingest([
            item for item in target_1 + target_2 if item.timestamp == timestamp
        ])
        assert update.initialized_states_by_target == {}
        assert set(update.waiting_target_ids) == {"target-1", "target-2"}

    update = manager.ingest([
        item for item in target_1 + target_2 if item.timestamp == 20.0
    ])

    assert set(update.initialized_states_by_target) == {"target-1", "target-2"}
    assert update.waiting_target_ids == ()
    assert manager.initialized_states["target-1"].track_id == "enemy-track-1"
    assert manager.initialized_states["target-2"].track_id == "enemy-track-2"

    repeated = manager.ingest(target_1)
    assert repeated.estimates_by_target == {}
    assert repeated.initialized_states_by_target == {}


def test_successful_iod_hands_off_to_continuous_tracking_without_reinitializing():
    truth, iod_observations = _observations()
    estimate = initialize_target(iod_observations, target_id="target-1")
    initial = target_initial_state_from_iod(estimate)
    tracking_times = np.array([30.0, 40.0, 50.0])
    propagation_times = np.concatenate(([0.0], tracking_times))
    target_history = propagate_absolute_orbit(truth, propagation_times)[1:]
    observer_initials = {
        "observer-1": _state(0.0, 54.8, 699e3),
        "observer-2": _state(0.2, 54.9, 701e3),
        "observer-3": _state(-0.2, 54.7, 698e3),
    }
    observer_histories = {
        key: propagate_absolute_orbit(value, propagation_times)[1:]
        for key, value in observer_initials.items()
    }
    quaternions = {
        key: build_rtn_quaternion_history(history)
        for key, history in observer_histories.items()
    }
    observations_by_link = {}
    for index, (observer_id, observer_history) in enumerate(observer_histories.items()):
        relative = target_history - observer_history
        spri = state_history_eci_to_spri(relative, quaternions[observer_id])
        rng = np.random.default_rng(100 + index)
        observations_by_link[(observer_id, "target-1")] = (
            create_infrared_observations(
                timestamps=tracking_times,
                relative_position_spri=spri[:, :3],
                covariance=np.diag(np.deg2rad([0.02, 0.02])) ** 2,
                observer_id=observer_id,
                target_id="target-1",
                rng=rng,
            )
            + create_radar_observations(
                timestamps=tracking_times,
                relative_position_spri=spri[:, :3],
                relative_velocity_spri=spri[:, 3:],
                covariance=np.diag([10.0, 0.05]) ** 2,
                observer_id=observer_id,
                target_id="target-1",
                rng=rng,
            )
        )
    handoff = build_tracking_handoff(
        initialized_states={"target-1": initial},
        timestamps=tracking_times,
        observer_state_history_by_id=observer_histories,
        q_eci2pri_history_by_id=quaternions,
        observations_by_link=observations_by_link,
    )

    history = run_known_target_sequence(
        scene_id="iod-handoff",
        module_inputs=handoff.module_inputs,
        initial_target_states=handoff.target_initial_states,
    )

    assert handoff.start_timestamp == 30.0
    assert handoff.target_initial_states["target-1"].timestamp == 30.0
    assert len(handoff.module_inputs) == 3
    assert all(
        track.lifecycle.value == "TRACKING"
        for track in history.track_history_by_target["target-1"]
    )
