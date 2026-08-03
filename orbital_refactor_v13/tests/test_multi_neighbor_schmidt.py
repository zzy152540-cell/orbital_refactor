import numpy as np

from cooperative.multi_neighbor_schmidt import (
    add_consider_neighbor,
    initialize_multi_neighbor_schmidt,
    multi_neighbor_schmidt_update,
    remove_consider_neighbor,
    run_multi_neighbor_schmidt_history,
)
from cooperative.schmidt_consider import SchmidtState, schmidt_update
from interfaces.data_objects import ObservationMessage
from orbital_core.measurements import (
    measure_relative_az_el,
    measure_relative_range,
    measure_relative_range_rate,
)


def _states():
    active = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    left = active + np.array([-1000.0, 100.0, 20.0, 0.0, -0.02, 0.0])
    right = active + np.array([1200.0, -80.0, 30.0, 0.0, 0.03, 0.0])
    return active, left, right


def _message(observer, target, modality, measurement, covariance, suffix):
    return ObservationMessage(
        message_id=suffix,
        physical_observation_id=suffix,
        observer_id=observer,
        target_id=target,
        timestamp=0.0,
        modality=modality,
        measurement=np.asarray(measurement, dtype=float).reshape(-1),
        covariance=np.asarray(covariance, dtype=float),
        frame="RTN" if modality == "AZ_EL" else "ECI",
    )


def test_one_neighbor_multi_schmidt_matches_two_node_implementation():
    active, _, right = _states()
    multi = initialize_multi_neighbor_schmidt(
        timestamp=0.0,
        active_node_id="sat_02",
        active_state=active,
        active_covariance=np.eye(6),
        neighbor_state_by_id={"sat_03": right},
        neighbor_covariance_by_id={"sat_03": 2.0 * np.eye(6)},
    )
    pair = SchmidtState(
        timestamp=0.0,
        active_node_id="sat_02",
        consider_node_id="sat_03",
        active_state=active,
        consider_state=right,
        active_covariance=np.eye(6),
        consider_covariance=2.0 * np.eye(6),
        cross_covariance=np.zeros((6, 6)),
    )
    observations = [
        _message(
            "sat_02", "sat_03", "RANGE",
            [measure_relative_range(active, right) + 1.0], [[4.0]], "range",
        ),
        _message(
            "sat_02", "sat_03", "RANGE_RATE",
            [measure_relative_range_rate(active, right) + 0.01], [[0.01]], "rate",
        ),
        _message(
            "sat_02", "sat_03", "AZ_EL",
            measure_relative_az_el(active, right, frame="RTN") + np.array([1e-4, -2e-4]),
            np.eye(2) * 1e-6, "angles",
        ),
    ]
    for observation in observations:
        multi_result = multi_neighbor_schmidt_update(multi, observation)
        pair_result = schmidt_update(pair, observation)
        multi, pair = multi_result.state, pair_result.state

    np.testing.assert_allclose(multi.active_state, pair.active_state)
    np.testing.assert_allclose(multi.active_covariance, pair.active_covariance)
    np.testing.assert_allclose(
        multi.active_cross_covariance("sat_03"), pair.cross_covariance
    )


def test_body_angle_update_reads_epoch_attitude_from_observation_metadata():
    active, _, right = _states()
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    state = initialize_multi_neighbor_schmidt(
        timestamp=0.0, active_node_id="sat_02", active_state=active,
        active_covariance=np.eye(6),
        neighbor_state_by_id={"sat_03": right},
        neighbor_covariance_by_id={"sat_03": 2.0 * np.eye(6)},
    )
    observation = ObservationMessage(
        message_id="body-angles", observer_id="sat_02", target_id="sat_03",
        timestamp=0.0, modality="AZ_EL", frame="BODY",
        measurement=measure_relative_az_el(
            active, right, frame="BODY", quaternion_i2b_wxyz=quaternion,
        ),
        covariance=np.eye(2) * 1e-6,
        metadata={"quaternion_i2b_wxyz": quaternion},
    )

    result = multi_neighbor_schmidt_update(state, observation)

    assert result.nis == 0.0
    assert np.linalg.eigvalsh(result.state.joint_covariance).min() >= -1e-10


def test_two_neighbor_updates_preserve_joint_psd_and_create_cross_terms():
    active, left, right = _states()
    state = initialize_multi_neighbor_schmidt(
        timestamp=0.0,
        active_node_id="sat_02",
        active_state=active,
        active_covariance=np.eye(6),
        neighbor_state_by_id={"sat_01": left, "sat_03": right},
        neighbor_covariance_by_id={
            "sat_01": 2.0 * np.eye(6),
            "sat_03": 3.0 * np.eye(6),
        },
    )
    for neighbor_id, neighbor_state in (("sat_01", left), ("sat_03", right)):
        observation = _message(
            "sat_02", neighbor_id, "RANGE",
            [measure_relative_range(active, neighbor_state) + 0.5],
            [[1.0]], f"range-{neighbor_id}",
        )
        state = multi_neighbor_schmidt_update(state, observation).state

    assert state.dimension == 18
    assert np.linalg.norm(state.active_cross_covariance("sat_01")) > 0.0
    assert np.linalg.norm(state.active_cross_covariance("sat_03")) > 0.0
    assert np.min(np.linalg.eigvalsh(state.joint_covariance)) >= -1e-9


def test_consider_neighbor_can_enter_and_leave_local_topology():
    active, left, right = _states()
    state = initialize_multi_neighbor_schmidt(
        timestamp=0.0,
        active_node_id="sat_02",
        active_state=active,
        active_covariance=np.eye(6),
        neighbor_state_by_id={"sat_01": left},
        neighbor_covariance_by_id={"sat_01": np.eye(6)},
    )
    expanded = add_consider_neighbor(
        state,
        neighbor_id="sat_03",
        neighbor_state=right,
        neighbor_covariance=2.0 * np.eye(6),
    )
    reduced = remove_consider_neighbor(expanded, "sat_01")

    assert expanded.neighbor_ids == ("sat_01", "sat_03")
    assert expanded.joint_covariance.shape == (18, 18)
    assert reduced.neighbor_ids == ("sat_03",)
    assert reduced.joint_covariance.shape == (12, 12)
    np.testing.assert_allclose(reduced.neighbor_state_by_id["sat_03"], right)


def test_three_satellite_active_node_history_accepts_multiple_modalities():
    active, left, right = _states()
    initial = initialize_multi_neighbor_schmidt(
        timestamp=0.0,
        active_node_id="sat_02",
        active_state=active,
        active_covariance=np.eye(6),
        neighbor_state_by_id={"sat_01": left, "sat_03": right},
        neighbor_covariance_by_id={
            "sat_01": np.eye(6), "sat_03": np.eye(6)
        },
    )
    observations = [
        _message(
            "sat_02", "sat_01", "RANGE",
            [measure_relative_range(active, left)], [[1.0]], "left-range",
        ),
        _message(
            "sat_03", "sat_02", "AZ_EL",
            measure_relative_az_el(right, active, frame="RTN"),
            np.eye(2) * 1e-4, "right-angle",
        ),
    ]
    history = run_multi_neighbor_schmidt_history(
        timestamps=np.array([0.0, 1.0]),
        initial_state=initial,
        observations=observations,
        process_noise_acceleration=0.0,
    )

    assert history.active_state_history.shape == (2, 6)
    assert history.joint_covariance_history.shape == (2, 18, 18)
    assert set(history.neighbor_state_history_by_id) == {"sat_01", "sat_03"}
    assert set(history.nis_history[0]) == {"left-range", "right-angle"}
