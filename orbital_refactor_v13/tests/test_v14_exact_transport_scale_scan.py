from experiments.v14_exact_transport_scale_scan import run_v14_exact_transport_smoke_scan
from experiments.v14_exact_transport_scale_scan import run_v14_exact_transport_topology_scan
from scenarios.measurement_visibility import VisibilityConfig


def test_production_api_smoke_scan_reports_all_modes_and_safe_history_failure():
    result = run_v14_exact_transport_smoke_scan(seeds=1, duration=8.0, dt=2.0)
    assert len(result.summary_by_scenario_and_mode) == 10
    ideal = result.summary_by_scenario_and_mode[("ideal", "exact_transport_event_replay")]
    assert ideal.message_acceptance_rate == 1.0
    assert ideal.psd_failure_count == 0
    assert ideal.mean_run_seconds > 0.0
    assert ideal.replay_count > 0
    assert ideal.total_replay_seconds > 0.0
    assert ideal.maximum_batch_size >= 1
    assert ideal.maximum_remote_event_count > 0
    assert ideal.maximum_observation_count > 0
    assert ideal.maximum_checkpoint_count > 0
    assert ideal.maximum_retained_journal_count >= ideal.maximum_remote_event_count
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


def test_topology_scan_can_select_formal_scenarios_and_modes():
    result = run_v14_exact_transport_topology_scan(
        node_count=3, topology_types=("chain",), seeds=1,
        duration=4.0, dt=2.0, scenario_names=("ideal",),
        modes=("exact_transport_event_replay",),
    )
    summaries = result.result_by_topology["chain"].summary_by_scenario_and_mode
    assert set(summaries) == {("ideal", "exact_transport_event_replay")}


def test_optional_visibility_filters_range_measurements_and_reports_summary():
    baseline = run_v14_exact_transport_smoke_scan(
        node_count=5, topology_type="star", seeds=1, duration=4.0, dt=2.0,
        scenario_names=("ideal",), modes=("propagate_only",),
    )
    visible = run_v14_exact_transport_smoke_scan(
        node_count=5, topology_type="star", seeds=1, duration=4.0, dt=2.0,
        scenario_names=("ideal",), modes=("propagate_only",),
        visibility_by_modality={
            "RANGE": VisibilityConfig(maximum_range=1500.0)
        },
    )

    assert baseline.visibility_summary is None
    assert visible.visibility_summary is not None
    assert 0.0 < visible.visibility_summary.overall.visibility_rate < 1.0
    baseline_nis = baseline.summary_by_scenario_and_mode[("ideal", "propagate_only")].mean_nis
    visible_nis = visible.summary_by_scenario_and_mode[("ideal", "propagate_only")].mean_nis
    assert baseline_nis != visible_nis


def test_scale_scan_visibility_rejects_unsupported_modalities():
    try:
        run_v14_exact_transport_smoke_scan(
            seeds=1, duration=4.0, dt=2.0,
            scenario_names=("ideal",), modes=("propagate_only",),
            visibility_by_modality={"AZ_EL": VisibilityConfig()},
        )
    except ValueError as error:
        assert "must match enabled relative modalities" in str(error)
    else:
        raise AssertionError("Expected unsupported visibility modality rejection.")


def test_scale_scan_supports_visibility_filtered_range_and_range_rate():
    result = run_v14_exact_transport_smoke_scan(
        node_count=3, topology_type="chain", seeds=1, duration=4.0, dt=2.0,
        scenario_names=("ideal",), modes=("exact_transport_event_replay",),
        relative_modalities=("RANGE", "RANGE_RATE"),
        visibility_by_modality={
            "RANGE": VisibilityConfig(maximum_range=1500.0),
            "RANGE_RATE": VisibilityConfig(maximum_range=1500.0),
        },
    )

    assert result.visibility_summary is not None
    assert set(result.visibility_summary.by_modality) == {"RANGE", "RANGE_RATE"}
    summary = result.summary_by_scenario_and_mode[
        ("ideal", "exact_transport_event_replay")
    ]
    assert summary.mean_nis > 0.0
    assert summary.psd_failure_count == 0


def test_scale_scan_supports_eci_az_el_with_dimension_aware_nis():
    result = run_v14_exact_transport_smoke_scan(
        node_count=3, topology_type="chain", seeds=1, duration=4.0, dt=2.0,
        scenario_names=("ideal",), modes=("exact_transport_event_replay",),
        relative_modalities=("RANGE", "AZ_EL"),
        visibility_by_modality={
            "RANGE": VisibilityConfig(maximum_range=3000.0),
            "AZ_EL": VisibilityConfig(maximum_range=3000.0),
        },
    )

    summary = result.summary_by_scenario_and_mode[
        ("ideal", "exact_transport_event_replay")
    ]
    assert set(summary.mean_nis_by_modality) == {"RANGE", "AZ_EL"}
    assert set(summary.mean_nis_95_coverage_by_modality) == {"RANGE", "AZ_EL"}
    assert 0.0 <= summary.mean_nis_95_coverage <= 1.0
    assert summary.psd_failure_count == 0
