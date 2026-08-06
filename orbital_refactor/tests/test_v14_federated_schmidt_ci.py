import numpy as np
from orbital_core.measurement_integrity import (
    INTEGRITY_MODE_PROPORTIONAL_WITH_HARD_GATE,
    MeasurementIntegrityPolicy,
)

from experiments.v14_federated_schmidt_ci import (
    run_v14_three_satellite_federated_schmidt_ci_experiment,
)


def test_output_only_federated_schmidt_ci_reports_all_architectures():
    result = run_v14_three_satellite_federated_schmidt_ci_experiment(
        seeds=1, duration=4.0, dt=2.0, ci_grid_points=7,
    )

    assert result.sequential_schmidt.run_count == 1
    assert result.federated_ci.run_count == 1
    assert set(result.local_by_modality) == {"RADAR", "INFRARED", "OPTICAL"}
    assert set(result.mean_ci_weight_by_modality) == {
        "RADAR", "INFRARED", "OPTICAL",
    }
    assert np.isclose(sum(result.mean_ci_weight_by_modality.values()), 1.0)
    assert set(result.mean_ci_weight_by_node_and_modality) == {
        "sat_a", "sat_b", "sat_c",
    }
    assert set(result.ci_prediction_only_exclusion_count_by_modality) == {
        "RADAR", "INFRARED", "OPTICAL",
    }
    assert result.ci_all_modalities_unavailable_count >= 0
    assert set(result.representative_module_output_by_node) == {
        "sat_a", "sat_b", "sat_c",
    }
    for output in result.representative_module_output_by_node.values():
        module = output.module_output
        assert module.state_output.position_estimate.shape == (3,)
        assert module.state_output.velocity_estimate.shape == (3,)
        assert module.state_output.acceleration_estimate.shape == (3,)
        assert set(module.fusion_status.modality_valid_flags) == {
            "RADAR", "INFRARED", "OPTICAL",
        }
        weight_sum = sum(module.fusion_status.modality_weights.values())
        assert np.isclose(weight_sum, 0.0) or np.isclose(weight_sum, 1.0)
        assert output.network_diagnostics.neighbor_count >= 1
    for weights in result.mean_ci_weight_by_node_and_modality.values():
        assert np.isclose(sum(weights.values()), 1.0) or np.isclose(
            sum(weights.values()), 0.0
        )
    for summary in (
        result.sequential_schmidt,
        result.federated_ci,
        *result.local_by_modality.values(),
    ):
        assert np.isfinite(summary.mean_position_rmse)
        assert np.isfinite(summary.mean_nees)
        assert 0.0 <= summary.mean_nees_95_coverage <= 1.0
        assert summary.mean_position_covariance_trace > 0.0
        assert set(summary.mean_position_rmse_by_node) == {
            "sat_a", "sat_b", "sat_c",
        }


def test_federated_ci_is_output_only_and_does_not_replace_local_histories():
    result = run_v14_three_satellite_federated_schmidt_ci_experiment(
        seeds=1, duration=8.0, dt=2.0, ci_grid_points=7,
    )

    local_position_rmse = {
        summary.mean_position_rmse
        for summary in result.local_by_modality.values()
    }
    assert len(local_position_rmse) == 3
    assert result.federated_ci.mean_position_covariance_trace != (
        result.local_by_modality["RADAR"].mean_position_covariance_trace
    )


def test_federated_ci_robustness_fault_inputs_are_supported():
    result = run_v14_three_satellite_federated_schmidt_ci_experiment(
        seeds=1, duration=8.0, dt=2.0, ci_grid_points=7,
        radar_actual_noise_scale=2.0,
        optical_outlier_bias=(0.02, -0.02),
        optical_outlier_window=(2.0, 4.0),
        infrared_outlier_bias=(0.01, -0.01),
        infrared_outlier_window=(2.0, 4.0),
        dropout_windows_by_modality={"INFRARED": ((6.0, 8.0),)},
    )

    assert np.isfinite(result.sequential_schmidt.mean_nees)
    assert np.isfinite(result.federated_ci.mean_nees)


def test_federated_ci_fault_windows_are_validated():
    try:
        run_v14_three_satellite_federated_schmidt_ci_experiment(
            seeds=1, duration=4.0, dt=2.0, ci_grid_points=7,
            optical_outlier_bias=(0.02, -0.02),
        )
    except ValueError as error:
        assert "requires optical_outlier_window" in str(error)
    else:
        raise AssertionError("Expected missing optical fault window rejection.")


def test_local_nis_gate_excludes_faulted_modality_from_epoch_ci():
    result = run_v14_three_satellite_federated_schmidt_ci_experiment(
        seeds=1, duration=8.0, dt=2.0, ci_grid_points=7,
        infrared_outlier_bias=(0.02, -0.02),
        infrared_outlier_window=(2.0, 4.0),
        nis_gate_threshold_by_modality={
            "RADAR": 9.21, "INFRARED": 9.21, "OPTICAL": 9.21,
        },
    )

    assert result.gated_observation_count_by_modality["INFRARED"] > 0
    assert result.ci_exclusion_count_by_modality["INFRARED"] > 0
    assert np.isfinite(result.federated_ci.mean_nees)


def test_adaptive_covariance_inflation_supports_extreme_outlier_gate():
    result = run_v14_three_satellite_federated_schmidt_ci_experiment(
        seeds=1, duration=8.0, dt=2.0, ci_grid_points=7,
        radar_actual_noise_scale=3.0,
        nis_inflation_threshold_by_modality={"RADAR": 5.99},
        maximum_measurement_covariance_scale_by_modality={"RADAR": 25.0},
        nis_gate_threshold_by_modality={"RADAR": 25.0},
    )

    assert np.isfinite(result.sequential_schmidt.mean_nees)
    assert np.isfinite(result.federated_ci.mean_nees)


def test_prediction_only_modality_is_excluded_from_ci():
    result = run_v14_three_satellite_federated_schmidt_ci_experiment(
        seeds=1, duration=8.0, dt=2.0, ci_grid_points=7,
        dropout_windows_by_modality={"OPTICAL": ((2.0, 8.0),)},
    )

    assert result.ci_prediction_only_exclusion_count_by_modality["OPTICAL"] > 0
    assert np.isfinite(result.federated_ci.mean_position_rmse)


def test_absolute_navigation_dropout_window_is_supported():
    result = run_v14_three_satellite_federated_schmidt_ci_experiment(
        seeds=1, duration=8.0, dt=2.0, ci_grid_points=7,
        absolute_navigation_dropout_windows=((2.0, 6.0),),
    )

    assert np.isfinite(result.sequential_schmidt.mean_position_rmse)
    assert np.isfinite(result.federated_ci.mean_position_rmse)
    assert set(result.phase_summary_by_architecture) == {
        "sequential_schmidt", "federated_ci",
    }
    for phases in result.phase_summary_by_architecture.values():
        assert set(phases) == {
            "pre_dropout", "dropout", "post_recovery",
        }
        assert all(np.isfinite(item.mean_nees) for item in phases.values())


def test_multimodal_experiment_supports_directed_state_link_outage():
    result = run_v14_three_satellite_federated_schmidt_ci_experiment(
        seeds=1, duration=8.0, dt=2.0, ci_grid_points=7,
        topology_type="ring",
        communication_outage_windows_by_directed_link={
            ("sat_b", "sat_a"): ((2.0, 4.0),),
        },
    )

    diagnostics = result.representative_module_output_by_node[
        "sat_b"
    ].network_diagnostics
    assert diagnostics.last_receive_timestamp_by_neighbor["sat_a"] == 8.0
    assert diagnostics.losses_before_last_delivery_by_neighbor["sat_a"] == 0
    assert diagnostics.maximum_consecutive_losses_by_neighbor["sat_a"] == 2
    assert diagnostics.recovery_delivery_count_by_neighbor["sat_a"] == 1
    assert diagnostics.current_topology_version == 0
    assert set(result.phase_summary_by_architecture[
        "sequential_schmidt"
    ]) == {"pre_dropout", "dropout", "post_recovery"}


def test_absolute_navigation_dropout_can_target_one_node():
    result = run_v14_three_satellite_federated_schmidt_ci_experiment(
        seeds=1, duration=8.0, dt=2.0, ci_grid_points=7,
        absolute_navigation_dropout_windows_by_node={
            "sat_b": ((2.0, 6.0),),
        },
    )

    assert set(result.phase_summary_by_architecture[
        "sequential_schmidt"
    ]) == {"pre_dropout", "dropout", "post_recovery"}


def test_multimodal_experiment_accepts_fixed_chain_topology():
    result = run_v14_three_satellite_federated_schmidt_ci_experiment(
        seeds=1, duration=4.0, dt=2.0, ci_grid_points=7,
        topology_type="chain",
    )

    assert result.representative_module_output_by_node[
        "sat_a"
    ].network_diagnostics.neighbor_count == 1


def test_runtime_topology_edge_schedule_suspends_observations_and_resumes():
    result = run_v14_three_satellite_federated_schmidt_ci_experiment(
        seeds=1, duration=8.0, dt=2.0, ci_grid_points=7,
        topology_type="chain",
        topology_inactive_windows_by_undirected_edge={
            ("sat_a", "sat_b"): ((2.0, 4.0),),
        },
    )

    assert result.topology_version_by_timestamp == {
        0.0: 0, 2.0: 1, 4.0: 1, 6.0: 2, 8.0: 2,
    }
    assert result.active_neighbors_by_timestamp[2.0]["sat_a"] == ()
    assert result.active_neighbors_by_timestamp[2.0]["sat_b"] == (
        "sat_c",
    )
    assert result.active_neighbors_by_timestamp[6.0]["sat_a"] == (
        "sat_b",
    )
    diagnostics = result.representative_module_output_by_node[
        "sat_b"
    ].network_diagnostics
    assert diagnostics.maximum_consecutive_losses_by_neighbor["sat_a"] == 2
    assert diagnostics.recovery_delivery_count_by_neighbor["sat_a"] == 1
    assert diagnostics.current_topology_version == 2
    assert diagnostics.topology_transition_count == 2
    assert diagnostics.active_neighbors == ("sat_a", "sat_c")
    assert diagnostics.inactive_configured_neighbors == ()


def test_long_topology_separation_requires_explicit_resynchronization():
    result = run_v14_three_satellite_federated_schmidt_ci_experiment(
        seeds=1, duration=10.0, dt=2.0, ci_grid_points=7,
        topology_type="chain", max_pinned_age=2.0,
        topology_inactive_windows_by_undirected_edge={
            ("sat_a", "sat_b"): ((2.0, 8.0),),
        },
    )

    diagnostics = result.representative_module_output_by_node[
        "sat_b"
    ].network_diagnostics
    assert diagnostics.maximum_resync_required_count >= 1
    assert diagnostics.resynchronization_required_by_neighbor["sat_a"]
    assert not diagnostics.resynchronization_required_by_neighbor["sat_c"]
    assert diagnostics.message_rejection_counts_by_reason[
        "resync_required"
    ] >= 1


def test_main_experiment_can_run_optional_online_resynchronization_backend():
    result = run_v14_three_satellite_federated_schmidt_ci_experiment(
        seeds=1, duration=12.0, dt=2.0, ci_grid_points=7,
        topology_type="ring", max_pinned_age=2.0,
        topology_inactive_windows_by_undirected_edge={
            ("sat_a", "sat_b"): ((2.0, 8.0),),
        },
        online_resynchronization_backend=True,
    )

    online = result.online_resynchronization_summary
    assert online is not None
    assert online.resynchronization_count == 2
    assert online.rejected_message_count == 0


def test_experiment_accepts_shared_integrity_policy_by_modality():
    policy = MeasurementIntegrityPolicy(
        mode=INTEGRITY_MODE_PROPORTIONAL_WITH_HARD_GATE,
        inflation_threshold=5.99,
        maximum_covariance_scale=9.0,
        hard_gate_threshold=9.21,
    )
    result = run_v14_three_satellite_federated_schmidt_ci_experiment(
        seeds=1, duration=8.0, dt=2.0, ci_grid_points=7,
        infrared_outlier_bias=(0.02, -0.02),
        infrared_outlier_window=(2.0, 4.0),
        integrity_policy_by_modality={"INFRARED": policy},
    )

    assert result.gated_observation_count_by_modality["INFRARED"] > 0
    assert result.ci_exclusion_count_by_modality["INFRARED"] > 0
