import numpy as np

from experiments.v14_three_satellite_local_observation import (
    run_v14_three_satellite_local_observation_experiment,
)
from experiments.v14_exact_transport_scale_scan import _build_case


def test_three_satellite_local_observation_experiment_uses_all_local_edges():
    result = run_v14_three_satellite_local_observation_experiment(
        seeds=1, duration=8.0, dt=2.0,
    )

    assert len(result.summary_by_case_and_mode) == 4
    assert 0.0 < result.visibility_summary.overall.visibility_rate < 1.0
    exact = result.summary_by_case_and_mode[
        ("visibility_limited", "exact_transport_event_replay")
    ]
    assert set(exact.mean_nis_by_modality) == {
        "RADAR", "INFRARED", "OPTICAL",
    }
    assert exact.configured_sensor_modalities == ("OPTICAL", "INFRARED", "RADAR")
    assert exact.transported_measurement_components == (
        "RADAR", "INFRARED", "OPTICAL",
    )
    assert exact.full_three_sensor_suite is True
    assert set(exact.mean_position_rmse_by_node) == {"sat_a", "sat_b", "sat_c"}
    assert set(exact.mean_nees_by_node) == {"sat_a", "sat_b", "sat_c"}
    assert set(exact.observation_count_by_directed_edge) == {
        ("sat_a", "sat_b"), ("sat_b", "sat_a"),
    }
    assert exact.message_acceptance_rate == 1.0
    assert exact.psd_failure_count == 0


def test_three_satellite_wire_components_record_physical_sensor_semantics():
    case = _build_case(
        seed=0, duration=2.0, dt=2.0,
        range_sigma=2.0, range_rate_sigma=0.05,
        az_el_sigma=np.deg2rad(0.05), absolute_sigma=3.0,
        process_noise_acceleration=1e-8,
        packet_loss=0.0, delay=0.0, acknowledge_messages=True,
        node_count=3, topology_type="ring",
        relative_modalities=("RANGE", "RANGE_RATE", "AZ_EL", "OPTICAL"),
    )

    semantics = {
        observation.modality: (
            observation.metadata["sensor_modality"],
            observation.metadata["measurement_type"],
        )
        for observation in case["observations"]
    }
    assert semantics == {
        "RANGE": ("RADAR", "RANGE"),
        "RANGE_RATE": ("RADAR", "RANGE_RATE"),
        "AZ_EL": ("INFRARED", "AZIMUTH_ELEVATION"),
        "OPTICAL": ("OPTICAL", "NORMALIZED_IMAGE_COORDINATES"),
    }


def test_three_satellite_joint_radar_message_preserves_correlated_covariance():
    case = _build_case(
        seed=0, duration=0.0, dt=2.0,
        range_sigma=2.0, range_rate_sigma=0.05, radar_correlation=0.4,
        az_el_sigma=np.deg2rad(0.05), optical_sigma=1e-3,
        absolute_sigma=3.0, process_noise_acceleration=1e-8,
        packet_loss=0.0, delay=0.0, acknowledge_messages=True,
        node_count=3, topology_type="ring",
        relative_modalities=("RADAR", "INFRARED", "OPTICAL"),
    )
    radar = next(
        observation for observation in case["observations"]
        if observation.modality == "RADAR"
    )

    assert radar.measurement.shape == (2,)
    np.testing.assert_allclose(
        radar.covariance,
        np.array([[4.0, 0.04], [0.04, 0.0025]]),
    )
    assert radar.metadata["measurement_type"] == "RANGE_RANGE_RATE"
