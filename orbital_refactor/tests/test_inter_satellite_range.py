import numpy as np
import pytest

from cooperative.inter_satellite_range import (
    update_with_inter_satellite_observation_block,
    update_with_relative_range,
    update_with_relative_range_rate,
)
from cooperative.satellite_node import SatelliteNode
from orbital_core.inter_satellite_model import (
    body_angle_attitude_jacobian,
    body_angle_effective_covariance,
    inter_satellite_jacobians,
    predict_inter_satellite_measurement,
)
from orbital_core.measurements import measure_relative_az_el, measure_relative_range_rate


def _report(node_id, state):
    return SatelliteNode(
        node_id=node_id,
        state=np.asarray(state, dtype=float),
        covariance=np.eye(6),
    ).estimate(0.0).to_report()


def test_range_update_reduces_position_error_along_line_of_sight():
    state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    covariance = np.eye(6)
    neighbor = _report("sat_02", [10.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    result = update_with_relative_range(
        state=state,
        covariance=covariance,
        neighbor_report=neighbor,
        measured_range=10.0,
        range_variance=0.01,
    )

    assert result.state[0] < state[0]
    assert result.covariance[0, 0] < covariance[0, 0]
    assert result.nis > 0.0


def test_range_update_accounts_for_neighbor_covariance():
    state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    local_covariance = np.eye(6)
    certain_neighbor = SatelliteNode(
        node_id="sat_02",
        state=np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        covariance=np.eye(6) * 1e-6,
    ).estimate(0.0).to_report()
    uncertain_neighbor = SatelliteNode(
        node_id="sat_02",
        state=np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        covariance=np.eye(6) * 100.0,
    ).estimate(0.0).to_report()

    certain = update_with_relative_range(
        state=state,
        covariance=local_covariance,
        neighbor_report=certain_neighbor,
        measured_range=10.0,
        range_variance=0.01,
    )
    uncertain = update_with_relative_range(
        state=state,
        covariance=local_covariance,
        neighbor_report=uncertain_neighbor,
        measured_range=10.0,
        range_variance=0.01,
    )

    assert abs(uncertain.state[0] - state[0]) < abs(certain.state[0] - state[0])
    assert uncertain.covariance[0, 0] > certain.covariance[0, 0]


def test_range_update_rejects_zero_predicted_range():
    state = np.zeros(6)
    neighbor = _report("sat_02", np.zeros(6))
    try:
        update_with_relative_range(
            state=state,
            covariance=np.eye(6),
            neighbor_report=neighbor,
            measured_range=1.0,
            range_variance=1.0,
        )
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("Expected zero range to be rejected.")


def test_measure_relative_range_rate_uses_relative_velocity_projection():
    state_i = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    state_j = np.array([10.0, 0.0, 0.0, 3.0, 4.0, 0.0])
    assert measure_relative_range_rate(state_i, state_j) == 2.0


def test_range_rate_update_reduces_velocity_covariance_along_line_of_sight():
    state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    covariance = np.eye(6)
    neighbor = _report("sat_02", [10.0, 0.0, 0.0, 3.0, 0.0, 0.0])

    result = update_with_relative_range_rate(
        state=state,
        covariance=covariance,
        neighbor_report=neighbor,
        measured_range_rate=3.0,
        range_rate_variance=0.01,
    )

    assert result.state[3] < state[3]
    assert result.covariance[3, 3] < covariance[3, 3]
    assert result.nis > 0.0


def test_inter_satellite_block_update_combines_range_and_range_rate():
    state = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    covariance = np.eye(6)
    neighbor = _report("sat_02", [10.0, 0.0, 0.0, 3.0, 0.0, 0.0])

    result = update_with_inter_satellite_observation_block(
        state=state,
        covariance=covariance,
        neighbor_report=neighbor,
        measurements_by_modality={"RANGE": 10.0, "RANGE_RATE": 3.0},
        variance_by_modality={"RANGE": 0.01, "RANGE_RATE": 0.01},
    )

    assert result.innovation.shape == (2,)
    assert result.modalities == ("RANGE", "RANGE_RATE")
    assert result.state[0] < state[0]
    assert result.state[3] < state[3]
    assert result.nis > 0.0


def test_measure_relative_az_el_returns_angles():
    state_i = np.zeros(6)
    state_j = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    measurement = measure_relative_az_el(state_i, state_j)
    np.testing.assert_allclose(
        measurement,
        np.array([np.pi / 4.0, np.arctan2(1.0, np.sqrt(2.0))]),
    )


def test_measure_relative_az_el_supports_rtn_frame():
    state_i = np.array([0.0, 7.0e6, 0.0, -7500.0, 0.0, 0.0])
    state_j = state_i + np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0])

    eci_measurement = measure_relative_az_el(state_i, state_j, frame="ECI")
    rtn_measurement = measure_relative_az_el(state_i, state_j, frame="RTN")

    np.testing.assert_allclose(eci_measurement, np.array([np.pi / 2.0, 0.0]))
    np.testing.assert_allclose(rtn_measurement, np.array([0.0, 0.0]), atol=1e-12)


def test_measure_relative_az_el_supports_body_frame_with_i2b_quaternion():
    state_i = np.zeros(6)
    state_j = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    half_angle = np.pi / 4.0
    quaternion_i2b = np.array([
        np.cos(half_angle),
        0.0,
        0.0,
        np.sin(half_angle),
    ])

    measurement = measure_relative_az_el(
        state_i,
        state_j,
        frame="BODY",
        quaternion_i2b_wxyz=quaternion_i2b,
    )

    np.testing.assert_allclose(measurement, np.array([np.pi / 2.0, 0.0]))


def test_measure_relative_az_el_requires_attitude_for_body_frame():
    with pytest.raises(ValueError, match="quaternion_i2b_wxyz"):
        measure_relative_az_el(
            np.zeros(6),
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            frame="BODY",
        )


def test_body_angle_model_identity_attitude_matches_eci_prediction_and_jacobians():
    state_i = np.array([7.0e6, 2.0, -1.0, 0.0, 7500.0, 0.0])
    state_j = state_i + np.array([100.0, 20.0, 10.0, 0.0, 0.0, 0.0])
    identity_i2b = np.array([1.0, 0.0, 0.0, 0.0])

    predicted_eci = predict_inter_satellite_measurement(
        state_i,
        state_j,
        modality="AZ_EL",
        frame="ECI",
    )
    predicted_body = predict_inter_satellite_measurement(
        state_i,
        state_j,
        modality="AZ_EL",
        frame="BODY",
        quaternion_i2b_wxyz=identity_i2b,
    )
    jacobians_eci = inter_satellite_jacobians(
        state_i,
        state_j,
        modality="AZ_EL",
        frame="ECI",
    )
    jacobians_body = inter_satellite_jacobians(
        state_i,
        state_j,
        modality="AZ_EL",
        frame="BODY",
        quaternion_i2b_wxyz=identity_i2b,
    )

    np.testing.assert_allclose(predicted_body, predicted_eci)
    np.testing.assert_allclose(jacobians_body[0], jacobians_eci[0])
    np.testing.assert_allclose(jacobians_body[1], jacobians_eci[1])


def test_body_angle_attitude_jacobian_matches_axis_rotation_geometry():
    state_i = np.zeros(6)
    state_j = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    jacobian = body_angle_attitude_jacobian(
        state_i,
        state_j,
        quaternion_i2b_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(
        jacobian,
        np.array([
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ]),
        atol=1e-9,
    )


def test_body_angle_effective_covariance_adds_attitude_uncertainty():
    state_i = np.zeros(6)
    state_j = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    sensor_covariance = np.diag([1.0e-4, 2.0e-4])
    attitude_covariance = np.diag([3.0e-4, 4.0e-4, 5.0e-4])

    effective = body_angle_effective_covariance(
        state_i,
        state_j,
        quaternion_i2b_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        sensor_covariance=sensor_covariance,
        attitude_covariance=attitude_covariance,
    )

    np.testing.assert_allclose(
        effective,
        sensor_covariance + np.diag([5.0e-4, 4.0e-4]),
    )
    assert np.all(np.linalg.eigvalsh(effective) >= 0.0)


def test_body_angle_effective_covariance_preserves_sensor_noise_at_zero_attitude_uncertainty():
    sensor_covariance = np.array([[2.0e-4, 1.0e-5], [1.0e-5, 3.0e-4]])

    effective = body_angle_effective_covariance(
        np.zeros(6),
        np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        quaternion_i2b_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        sensor_covariance=sensor_covariance,
        attitude_covariance=np.zeros((3, 3)),
    )

    np.testing.assert_allclose(effective, sensor_covariance)


@pytest.mark.parametrize(
    ("sensor_covariance", "attitude_covariance", "message"),
    [
        (np.eye(3), np.eye(3), "sensor_covariance must have shape"),
        (np.eye(2), np.diag([1.0, 1.0, -1.0]), "positive semidefinite"),
        (
            np.array([[1.0, 0.1], [0.0, 1.0]]),
            np.eye(3),
            "sensor_covariance must be symmetric",
        ),
    ],
)
def test_body_angle_effective_covariance_validates_inputs(
    sensor_covariance,
    attitude_covariance,
    message,
):
    with pytest.raises(ValueError, match=message):
        body_angle_effective_covariance(
            np.zeros(6),
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            quaternion_i2b_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            sensor_covariance=sensor_covariance,
            attitude_covariance=attitude_covariance,
        )


def test_inter_satellite_block_update_accepts_az_el():
    state = np.array([0.1, 0.0, 0.0, 1.0, 0.0, 0.0])
    covariance = np.eye(6)
    neighbor = _report("sat_02", [10.0, 1.0, 1.0, 3.0, 0.0, 0.0])
    truth_local = np.zeros(6)
    measurement = measure_relative_az_el(truth_local, neighbor.state_estimate)

    result = update_with_inter_satellite_observation_block(
        state=state,
        covariance=covariance,
        neighbor_report=neighbor,
        measurements_by_modality={"AZ_EL": measurement},
        covariance_by_modality={"AZ_EL": np.diag([1e-4, 1e-4])},
    )

    assert result.innovation.shape == (2,)
    assert result.modalities == ("AZ_EL",)
    assert result.covariance[0, 0] < covariance[0, 0]


def test_inter_satellite_block_update_accepts_rtn_az_el():
    state = np.array([0.0, 7.0e6, 0.0, -7500.0, 0.0, 0.0])
    covariance = np.eye(6)
    neighbor = _report(
        "sat_02",
        state + np.array([0.0, 10.0, 1.0, 0.0, 0.0, 0.0]),
    )
    measurement = measure_relative_az_el(state, neighbor.state_estimate, frame="RTN")

    result = update_with_inter_satellite_observation_block(
        state=state,
        covariance=covariance,
        neighbor_report=neighbor,
        measurements_by_modality={"AZ_EL": measurement},
        covariance_by_modality={"AZ_EL": np.diag([1e-4, 1e-4])},
        frame_by_modality={"AZ_EL": "RTN"},
    )

    np.testing.assert_allclose(result.innovation, np.zeros(2), atol=1e-12)
    assert result.nis == 0.0


def test_inter_satellite_block_update_soft_gate_downweights_outlier():
    state = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    covariance = np.eye(6)
    neighbor = _report("sat_02", [10.0, 0.0, 0.0, 3.0, 0.0, 0.0])

    ungated = update_with_inter_satellite_observation_block(
        state=state,
        covariance=covariance,
        neighbor_report=neighbor,
        measurements_by_modality={"RANGE": 100.0},
        variance_by_modality={"RANGE": 0.01},
    )
    soft = update_with_inter_satellite_observation_block(
        state=state,
        covariance=covariance,
        neighbor_report=neighbor,
        measurements_by_modality={"RANGE": 100.0},
        variance_by_modality={"RANGE": 0.01},
        gate_enable=True,
        gate_threshold=1.0,
        gate_mode="soft",
        soft_scale=100.0,
    )

    assert soft.gated
    assert not soft.skipped
    assert abs(soft.state[0] - state[0]) < abs(ungated.state[0] - state[0])


def test_inter_satellite_block_update_hard_gate_skips_outlier():
    state = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    covariance = np.eye(6)
    neighbor = _report("sat_02", [10.0, 0.0, 0.0, 3.0, 0.0, 0.0])

    result = update_with_inter_satellite_observation_block(
        state=state,
        covariance=covariance,
        neighbor_report=neighbor,
        measurements_by_modality={"RANGE": 100.0},
        variance_by_modality={"RANGE": 0.01},
        gate_enable=True,
        gate_threshold=1.0,
        gate_mode="hard",
    )

    assert result.gated
    assert result.skipped
    np.testing.assert_allclose(result.state, state)
    np.testing.assert_allclose(result.covariance, covariance)
