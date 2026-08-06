from experiments.v14_walker_filter_baseline import run_v14_walker_filter_smoke


def test_walker_20_satellite_static_filter_smoke_is_consistent():
    result = run_v14_walker_filter_smoke(seeds=1, duration=4.0, dt=2.0)

    assert result.walker_definition == (20, 10, 1)
    assert result.node_count == 20
    assert result.persistent_undirected_edge_count == 20
    assert result.minimum_node_degree == result.maximum_node_degree == 2
    assert set(result.mean_nis_by_modality) == {
        "RADAR", "INFRARED", "OPTICAL",
    }
    assert all(value > 0 for value in result.observation_count_by_modality_per_run.values())
    assert result.message_acceptance_rate == 1.0
    assert result.message_rejection_count == 0
    assert result.psd_failure_count == 0
    assert result.minimum_joint_eigenvalue >= -1e-8
    assert result.mean_run_seconds > 0.0
    assert result.replay_count > 0
    assert len(result.temporal_diagnostics) == 1
    diagnostic = result.temporal_diagnostics[0]
    assert diagnostic.start_timestamp == 0.0
    assert diagnostic.end_timestamp == 4.0
    assert diagnostic.sample_count == 20 * 3
    assert diagnostic.mean_position_standard_deviation > 0.0
    assert diagnostic.mean_velocity_standard_deviation > 0.0
