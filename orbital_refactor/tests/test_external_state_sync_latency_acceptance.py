from experiments.external_state_sync_latency_acceptance import (
    run_external_state_sync_latency_acceptance,
    save_external_state_sync_latency_report,
)
import json


def test_short_p11_scan_separates_wall_clock_and_simulated_delay():
    report = run_external_state_sync_latency_acceptance(
        seeds=(0,), duration=0.6, dt=0.2,
    )
    assert report.formal_category == "ordinary_update"
    assert report.formal_sample_size_met is False
    assert report.formal_duration_met is False
    assert report.passed is False
    assert report.protocol_clean is True
    groups = {item.category: item for item in report.groups}
    assert "ordinary_update" in groups
    assert "historical_replay" in groups
    assert groups["ordinary_update"].mean_simulated_link_delay_seconds == 0.0
    assert groups["historical_replay"].mean_simulated_link_delay_seconds > 0.0
    assert all(item.creation_to_applied_ms >= 0.0 for item in report.records)
    assert all(item.reason for item in report.records)


def test_p11_summary_references_csv_without_duplicating_records(tmp_path):
    report = run_external_state_sync_latency_acceptance(
        seeds=(0,), duration=0.2, dt=0.2,
    )
    output = save_external_state_sync_latency_report(report, tmp_path)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["records_file"] == "communication_latency.csv"
    assert "records" not in summary
    assert (output / "communication_latency.csv").is_file()
