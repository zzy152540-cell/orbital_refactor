import pytest

from experiments.v14_online_topology_resynchronization import (
    run_v14_online_topology_resynchronization_experiment,
)
from experiments.v15_system_formal_matrix import run_v15_system_formal_matrix


def test_online_resynchronization_accepts_explicit_seed_values():
    result = run_v14_online_topology_resynchronization_experiment(
        seeds=99, seed_values=(7,), node_count=5, duration=4.0, dt=2.0,
        topology_inactive_windows_by_undirected_edge={},
    )
    assert result.run_count == 1


def test_formal_matrix_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        run_v15_system_formal_matrix(node_counts=(2,), seed_values=(0,))
    with pytest.raises(ValueError):
        run_v15_system_formal_matrix(node_counts=(5,), seed_values=(0, 0))
    with pytest.raises(ValueError):
        run_v15_system_formal_matrix(
            node_counts=(5,), seed_values=(0,), conditions=("unknown",),
        )


def test_formal_matrix_runs_normal_and_recovery_conditions():
    report = run_v15_system_formal_matrix(
        node_counts=(5,), seed_values=(3,),
        conditions=("normal", "link_outage_recovery"),
        duration=8.0, dt=2.0,
    )
    assert report.passed
    assert len(report.runs) == 2
    assert {run.condition for run in report.runs} == {
        "normal", "link_outage_recovery",
    }
    normal = next(run for run in report.runs if run.condition == "normal")
    assert normal.suppressed_absolute_observation_count == 0


def test_formal_matrix_runs_packet_loss():
    report = run_v15_system_formal_matrix(
        node_counts=(5,), seed_values=(4,),
        conditions=("packet_loss",),
        duration=8.0, dt=2.0,
    )
    assert report.runs[0].dropped_message_count > 0


def test_formal_matrix_accepts_delay_with_sufficient_history_window():
    report = run_v15_system_formal_matrix(
        node_counts=(5,), seed_values=(0,),
        conditions=("communication_delay",),
        duration=20.0, dt=2.0,
    )
    assert report.passed
    assert report.runs[0].protocol_rejection_count == 0
    assert report.runs[0].resynchronization_count == 0
    assert not report.runs[0].failure_reasons


def test_formal_matrix_suppresses_target_absolute_navigation():
    report = run_v15_system_formal_matrix(
        node_counts=(5,), seed_values=(0,),
        conditions=("absolute_navigation_dropout",),
        duration=8.0, dt=2.0,
    )
    assert report.passed
    assert report.runs[0].suppressed_absolute_observation_count == 3
    assert report.runs[0].target_pre_dropout_rmse_m is not None
    assert report.runs[0].target_dropout_rmse_m is not None
    assert report.runs[0].target_post_recovery_rmse_m is not None


def test_formal_matrix_suppresses_two_navigation_nodes():
    report = run_v15_system_formal_matrix(
        node_counts=(5,), seed_values=(0,),
        conditions=("multi_node_navigation_dropout",),
        duration=8.0, dt=2.0,
    )
    run = report.runs[0]
    assert report.passed
    assert run.navigation_dropout_node_count == 2
    assert run.suppressed_absolute_observation_count == 6
    assert run.affected_nodes_dropout_rmse_m is not None


def test_multi_node_navigation_reference_keeps_measurements():
    report = run_v15_system_formal_matrix(
        node_counts=(5,), seed_values=(0,),
        conditions=("multi_node_navigation_reference",),
        duration=8.0, dt=2.0,
    )
    run = report.runs[0]
    assert report.passed
    assert run.navigation_dropout_node_count == 0
    assert run.suppressed_absolute_observation_count == 0
    assert run.affected_nodes_dropout_rmse_m is not None


def test_combined_navigation_and_communication_degradation():
    report = run_v15_system_formal_matrix(
        node_counts=(5,), seed_values=(0,),
        conditions=("navigation_communication_degradation",),
        duration=8.0, dt=2.0,
    )
    run = report.runs[0]
    assert report.passed
    assert run.navigation_dropout_node_count == 2
    assert run.suppressed_absolute_observation_count == 6
    assert run.dropped_message_count > 0
    assert run.protocol_rejection_count == 0
