import json

from experiments.multitarget_cooperative_sync_latency_acceptance import (
    run_multitarget_cooperative_sync_latency_acceptance,
    save_multitarget_cooperative_sync_latency_report,
)


def test_integrated_sync_includes_real_ring_consensus_ci():
    report = run_multitarget_cooperative_sync_latency_acceptance(
        frame_count=2,
        friendly_observer_count=3,
        hostile_target_count=2,
        sample_interval_seconds=0.5,
        consensus_grid_points=5,
    )

    assert report.consensus_ci_included is True
    assert report.local_estimation_included is False
    assert report.sender_receiver_process_separated is True
    assert report.protocol_clean is True
    assert len(report.records) == 2
    assert all(item.datagram_count == 3 for item in report.records)
    assert all(item.estimate_count == 6 for item in report.records)
    assert all(item.received_message_count == 12 for item in report.records)
    assert all(item.consensus_ci_ms > 0.0 for item in report.records)
    assert all(
        item.report_ready_to_consensus_applied_ms >= item.consensus_ci_ms
        for item in report.records
    )


def test_integrated_sync_report_writes_summary_and_csv(tmp_path):
    report = run_multitarget_cooperative_sync_latency_acceptance(
        frame_count=1,
        friendly_observer_count=3,
        hostile_target_count=1,
        consensus_grid_points=3,
    )
    output = save_multitarget_cooperative_sync_latency_report(report, tmp_path)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))

    assert summary["records_file"] == "cooperative_sync_latency.csv"
    assert summary["consensus_ci_included"] is True
    assert "records" not in summary
    assert (output / "cooperative_sync_latency.csv").is_file()
