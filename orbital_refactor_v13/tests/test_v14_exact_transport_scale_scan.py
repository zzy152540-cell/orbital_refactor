from experiments.v14_exact_transport_scale_scan import run_v14_exact_transport_smoke_scan


def test_production_api_smoke_scan_reports_all_modes_and_safe_history_failure():
    result = run_v14_exact_transport_smoke_scan(seeds=1, duration=8.0, dt=2.0)
    assert len(result.summary_by_scenario_and_mode) == 10
    ideal = result.summary_by_scenario_and_mode[("ideal", "exact_transport_event_replay")]
    assert ideal.message_acceptance_rate == 1.0
    assert ideal.psd_failure_count == 0
    insufficient = result.summary_by_scenario_and_mode[("insufficient_history", "exact_transport_event_replay")]
    assert insufficient.message_rejection_count > 0
