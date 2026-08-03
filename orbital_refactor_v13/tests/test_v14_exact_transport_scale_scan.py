from experiments.v14_exact_transport_scale_scan import run_v14_exact_transport_smoke_scan
from experiments.v14_exact_transport_scale_scan import run_v14_exact_transport_topology_scan


def test_production_api_smoke_scan_reports_all_modes_and_safe_history_failure():
    result = run_v14_exact_transport_smoke_scan(seeds=1, duration=8.0, dt=2.0)
    assert len(result.summary_by_scenario_and_mode) == 10
    ideal = result.summary_by_scenario_and_mode[("ideal", "exact_transport_event_replay")]
    assert ideal.message_acceptance_rate == 1.0
    assert ideal.psd_failure_count == 0
    insufficient = result.summary_by_scenario_and_mode[("insufficient_history", "exact_transport_event_replay")]
    assert insufficient.message_rejection_count > 0
    assert any(
        "consecutive_losses_before_delivery" in record
        for record in result.diagnostic_records
    )
    assert any(
        record["reason"] == "history_unavailable"
        for record in result.diagnostic_records
    )


def test_five_node_topology_scan_uses_all_public_topologies():
    result = run_v14_exact_transport_topology_scan(
        node_count=5, topology_types=("chain", "ring", "star"),
        seeds=1, duration=4.0, dt=2.0,
    )
    assert set(result.result_by_topology) == {"chain", "ring", "star"}
    for topology_type, scan in result.result_by_topology.items():
        ideal = scan.summary_by_scenario_and_mode[("ideal", "exact_transport_event_replay")]
        assert ideal.node_count == 5
        assert ideal.topology_type == topology_type
        assert ideal.psd_failure_count == 0
