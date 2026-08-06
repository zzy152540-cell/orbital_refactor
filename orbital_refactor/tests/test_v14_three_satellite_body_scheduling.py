import numpy as np

from experiments.v14_exact_transport_scale_scan import _build_case
from experiments.v14_three_satellite_local_observation import (
    _longest_unobserved_first_attitudes,
    _three_satellite_scenario,
    run_v14_three_satellite_body_scheduling_experiment,
)
from scenarios.measurement_visibility import VisibilityConfig


def test_three_satellite_body_scheduler_limits_optical_target_per_observer():
    result = run_v14_three_satellite_body_scheduling_experiment(
        seeds=1, duration=8.0, dt=2.0,
    )

    assert result.eci_upper_bound.psd_failure_count == 0
    assert result.body_scheduled.psd_failure_count == 0
    assert set(result.body_scheduled.mean_nis_by_modality) == {
        "RADAR", "INFRARED", "OPTICAL",
    }
    assert result.body_scheduled.full_three_sensor_suite is True
    selected = result.scheduling.selected_count_by_directed_edge
    assert selected[("sat_a", "sat_b")] == 5
    assert selected[("sat_a", "sat_c")] == 0
    assert result.scheduling.maximum_unobserved_visible_epochs_by_directed_edge[
        ("sat_a", "sat_b")
    ] == 0


def test_physical_angular_modalities_share_one_scheduled_target_and_valid_image():
    timestamps = np.arange(0.0, 8.1, 2.0)
    scenario = _three_satellite_scenario(timestamps)
    truth = scenario.truth_state_history_by_node
    attitudes, _ = _longest_unobserved_first_attitudes(
        timestamps=timestamps, truth=truth, maximum_range=5000.0,
    )
    fov = np.deg2rad(2.0)
    case = _build_case(
        seed=0, duration=8.0, dt=2.0,
        range_sigma=2.0, range_rate_sigma=0.05,
        az_el_sigma=np.deg2rad(0.05), optical_sigma=1e-3,
        absolute_sigma=3.0, process_noise_acceleration=1e-8,
        packet_loss=0.0, delay=0.0, acknowledge_messages=True,
        node_count=3, topology_type="ring",
        truth_initial_state_by_node={node: values[0] for node, values in truth.items()},
        relative_modalities=("RADAR", "INFRARED", "OPTICAL"),
        visibility_by_modality={
            "RADAR": VisibilityConfig(maximum_range=5000.0),
            "INFRARED": VisibilityConfig(
                maximum_range=5000.0, field_of_view_half_angle=fov,
            ),
            "OPTICAL": VisibilityConfig(
                maximum_range=5000.0, field_of_view_half_angle=fov,
            ),
        },
        az_el_frame="BODY", attitude_history_by_node=attitudes,
    )

    angular_targets = {}
    for observation in case["observations"]:
        if observation.modality not in {"INFRARED", "OPTICAL"}:
            continue
        key = (observation.timestamp, observation.observer_id)
        angular_targets.setdefault(key, set()).add(observation.target_id)
        if observation.modality == "OPTICAL":
            assert observation.frame == "BODY"
            assert np.all(np.abs(observation.measurement) < np.tan(fov) + 0.01)

    assert angular_targets
    assert all(len(targets) <= 1 for targets in angular_targets.values())


def test_optical_and_infrared_fov_can_be_configured_independently():
    result = run_v14_three_satellite_body_scheduling_experiment(
        seeds=1, duration=4.0, dt=2.0,
        infrared_fov_half_angle=np.deg2rad(3.0),
        optical_fov_half_angle=np.deg2rad(1.0),
    )

    assert result.body_scheduled.psd_failure_count == 0
    assert set(result.body_scheduled.mean_nis_by_modality) == {
        "RADAR", "INFRARED", "OPTICAL",
    }
