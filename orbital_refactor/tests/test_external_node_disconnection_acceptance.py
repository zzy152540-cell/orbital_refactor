import numpy as np

from cooperative.topology import ring_topology
from experiments.external_node_disconnection_acceptance import (
    isolate_nodes, run_external_node_disconnection_acceptance,
)


def test_isolation_removes_all_edges_of_selected_nodes():
    topology = ring_topology(tuple(f"sat_{index:02d}" for index in range(6)))
    result = isolate_nodes(topology, ("sat_01", "sat_04"))
    assert result.neighbors("sat_01") == ()
    assert result.neighbors("sat_04") == ()
    assert all(
        neighbor not in {"sat_01", "sat_04"}
        for node in result.node_ids for neighbor in result.neighbors(node)
    )


def test_short_r10_run_reports_fleet_affected_and_remaining_groups():
    report = run_external_node_disconnection_acceptance(
        seeds=(0,), duration=4.0, dt=2.0, formal_minimum_run_count=2,
    )
    assert report.walker_definition == (20, 10, 1)
    assert report.disconnected_node_count == 4
    assert not report.formal_sample_size_met
    assert not report.formal_duration_met
    assert not report.passed
    record = report.records[0]
    assert len(record.disconnected_nodes) == 4
    assert record.protocol_rejection_count == 0
    assert np.all(np.isfinite([
        record.normal_position_rmse_m, record.disconnected_position_rmse_m,
        record.normal_velocity_rmse_mps, record.disconnected_velocity_rmse_mps,
        record.normal_acceleration_rmse_mps2,
        record.disconnected_acceleration_rmse_mps2,
        record.normal_affected_position_rmse_m,
        record.disconnected_affected_position_rmse_m,
        record.normal_remaining_position_rmse_m,
        record.disconnected_remaining_position_rmse_m,
    ]))
