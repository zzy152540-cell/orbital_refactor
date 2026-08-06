from experiments.v14_walker_dynamic_topology import (
    build_v14_walker_dynamic_topology_plan,
    run_v14_walker_dynamic_filter_smoke,
    run_v14_walker_online_dynamic_filter_smoke,
)
from experiments.v14_walker_geometry_audit import _component_sizes


def test_walker_dynamic_selector_keeps_bounded_connected_topology():
    plan = build_v14_walker_dynamic_topology_plan(
        duration=600.0, dt=10.0, maximum_degree=3,
    )

    assert plan.minimum_candidate_edge_count >= 20
    assert plan.minimum_active_edge_count == plan.maximum_active_edge_count == 19
    assert plan.maximum_selected_node_degree <= 3
    assert plan.topology_change_count > 0
    for record in plan.epoch_records:
        assert _component_sizes(
            plan.scenario.node_ids, record.active_undirected_edges,
        ) == (20,)
        assert set(record.active_undirected_edges) <= set(
            tuple(sorted((node, neighbor)))
            for node in plan.scenario.node_ids
            for neighbor in plan.topology_by_timestamp[record.timestamp].neighbors(node)
        )


def test_walker_offline_dynamic_filter_exposes_required_resynchronization():
    result = run_v14_walker_dynamic_filter_smoke(duration=60.0, dt=2.0)

    assert result.topology_change_count >= 1
    assert result.configured_union_edge_count > 19
    assert result.maximum_configured_node_degree >= 3
    assert set(result.mean_nis_by_modality) == {
        "RADAR", "INFRARED", "OPTICAL",
    }
    assert 0.95 < result.message_acceptance_rate < 1.0
    assert result.message_rejection_count > 0
    assert set(result.rejection_counts) == {
        "reference_covariance_mismatch", "history_unavailable",
    }
    assert result.psd_failure_count == 0
    assert result.minimum_joint_eigenvalue >= -1e-8


def test_walker_online_dynamic_filter_resynchronizes_without_protocol_rejection():
    result = run_v14_walker_online_dynamic_filter_smoke(
        duration=60.0, dt=2.0,
    )

    assert result.topology_change_count == 1
    assert result.resynchronization_count > 0
    assert all(":resync:" in lineage for _, _, lineage in result.resynchronized_links)
    assert result.rejected_message_count == 0
    assert result.rejection_counts_by_reason == {}
    assert result.stale_topology_message_count == 0
    assert result.protocol_rejected_message_count == 0
    assert result.dropped_message_count == 0
    assert result.psd_failure_count == 0
    assert result.minimum_joint_eigenvalue >= -1e-8


def test_walker_online_dynamic_filter_classifies_delayed_old_topology_messages():
    result = run_v14_walker_online_dynamic_filter_smoke(
        duration=60.0, dt=2.0,
        packet_loss_rate=0.1, communication_delay=2.0,
    )

    assert result.resynchronization_count == 4
    assert result.dropped_message_count > 0
    assert result.stale_topology_message_count > 0
    assert result.protocol_rejected_message_count == 0
    assert result.rejection_counts_by_reason == {
        "inactive_topology_link": result.stale_topology_message_count,
    }
    assert result.psd_failure_count == 0
