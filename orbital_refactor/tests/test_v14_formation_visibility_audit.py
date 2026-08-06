import numpy as np
import pytest

from experiments.v14_formation_visibility_audit import (
    run_v14_formation_visibility_audit,
    run_v14_physical_fleet_filter_baseline,
    run_v14_physical_visibility_transition_experiment,
    run_v14_ten_satellite_staggered_visibility_scan,
    run_v14_ten_satellite_topology_comparison,
)
from scenarios.fleet_scenario import centered_along_track_offsets


def test_centered_along_track_offsets_are_symmetric_and_named_stably():
    offsets = centered_along_track_offsets(
        node_count=5, orbital_radius=7.0e6, spacing=700.0,
    )

    assert tuple(offsets) == tuple(f"sat_{index:02d}" for index in range(1, 6))
    anomalies = np.array([offset.true_anomaly for offset in offsets.values()])
    assert np.isclose(anomalies.mean(), 0.0)
    assert np.allclose(np.diff(anomalies), 1.0e-4)
    assert all(offset.semi_major_axis == 0.0 for offset in offsets.values())


@pytest.mark.parametrize("node_count", [5, 10])
def test_reference_formation_has_continuously_visible_chain(node_count):
    result = run_v14_formation_visibility_audit(
        node_count=node_count, duration=60.0, dt=10.0,
    )

    assert result.scenario.node_ids == tuple(
        f"sat_{index:02d}" for index in range(1, node_count + 1)
    )
    assert result.connected_at_every_epoch
    assert result.chain_summary.overall.visibility_rate == 1.0
    assert set(result.chain_summary.by_modality) == {
        "RADAR", "INFRARED", "OPTICAL",
    }
    if node_count == 10:
        assert result.all_pairs_summary.overall.visibility_rate < 1.0


def test_reference_formation_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="node_count"):
        run_v14_formation_visibility_audit(node_count=1)
    with pytest.raises(ValueError, match="spacing"):
        centered_along_track_offsets(
            node_count=5, orbital_radius=7.0e6, spacing=0.0,
        )


def test_five_satellite_physical_filter_baseline_reports_all_modalities():
    result = run_v14_physical_fleet_filter_baseline(
        node_count=5, seeds=1, duration=4.0, dt=2.0,
    )

    assert result.run_count == 1
    assert set(result.mean_nis_by_modality) == {
        "RADAR", "INFRARED", "OPTICAL",
    }
    assert set(result.mean_nis_95_coverage_by_modality) == {
        "RADAR", "INFRARED", "OPTICAL",
    }
    assert result.observation_count_by_modality_per_run == {
        "RADAR": 24, "INFRARED": 24, "OPTICAL": 24,
    }
    assert result.message_acceptance_rate == 1.0
    assert result.message_rejection_count == 0
    assert result.psd_failure_count == 0
    assert result.minimum_joint_eigenvalue >= -1e-8
    assert result.mean_run_seconds > 0.0
    assert result.replay_count > 0
    assert result.maximum_remote_event_count > 0
    assert result.maximum_observation_count > 0
    assert result.maximum_checkpoint_count > 0
    assert result.connected_at_every_epoch


def test_physical_drift_creates_loss_and_recovery_without_protocol_failure():
    result = run_v14_physical_visibility_transition_experiment(
        node_count=5, seeds=1, duration=120.0, dt=4.0,
    )

    for case in (result.loss, result.recovery):
        assert case.visibility_summary.overall.visibility_rate < 1.0
        assert case.visibility_summary.overall.rejection_counts == {
            "range_exceeded": case.visibility_summary.overall.opportunity_count
            - case.visibility_summary.overall.visible_count
        }
        assert case.message_acceptance_rate == 1.0
        assert case.message_rejection_count == 0
        assert case.psd_failure_count == 0
    loss_switches = result.loss.visibility_summary.availability_switch_count_by_edge_and_modality
    recovery_switches = result.recovery.visibility_summary.availability_switch_count_by_edge_and_modality
    assert all(value == 1 for value in loss_switches.values())
    assert all(value == 1 for value in recovery_switches.values())


def test_staggered_two_hop_losses_keep_ten_satellite_graph_connected():
    result = run_v14_ten_satellite_staggered_visibility_scan(
        seeds=1, duration=120.0, dt=4.0,
    )

    assert result.baseline.connected_at_every_epoch
    assert len(result.transition_timestamps) >= 2
    assert result.minimum_visible_directed_edges >= 2 * (10 - 1)
    assert result.maximum_visible_directed_edges > result.minimum_visible_directed_edges
    assert result.baseline.message_acceptance_rate == 1.0
    assert result.baseline.message_rejection_count == 0
    assert result.baseline.psd_failure_count == 0


def test_topology_comparison_uses_identical_nodes_and_more_two_hop_data():
    result = run_v14_ten_satellite_topology_comparison(
        seeds=1, duration=4.0, dt=2.0,
    )

    nearest = result.nearest_neighbor_chain
    two_hop = result.two_hop_chain
    assert set(nearest.mean_nees_by_node) == set(two_hop.mean_nees_by_node)
    assert set(nearest.mean_position_rmse_by_node) == set(
        two_hop.mean_position_rmse_by_node
    )
    assert (
        two_hop.observation_count_by_modality_per_run["RADAR"]
        > nearest.observation_count_by_modality_per_run["RADAR"]
    )
    assert nearest.psd_failure_count == two_hop.psd_failure_count == 0
