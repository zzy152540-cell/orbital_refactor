import json

from experiments.multitarget_state_sync_latency_acceptance import (
    run_multitarget_state_sync_latency_acceptance,
    save_multitarget_state_sync_latency_report,
)


def test_cross_process_udp_prescan_separates_latency_stages():
    report = run_multitarget_state_sync_latency_acceptance(frame_count=4)
    assert report.transport == (
        "SAME_MACHINE_CROSS_PROCESS_IPV4_UDP_TARGET_REPORT_JSON"
    )
    assert report.friendly_observer_count == 15
    assert report.hostile_target_count == 15
    assert report.frame_count == 4
    assert report.algorithm_processing_included is False
    assert report.sender_receiver_process_separated is True
    assert report.protocol_clean is True
    assert len(report.records) == 4
    assert all(item.target_count == 15 for item in report.records)
    assert all(item.payload_bytes > 0 for item in report.records)
    assert all(item.construction_to_applied_ms >= 0.0 for item in report.records)
    assert report.mean_construction_to_applied_ms >= 0.0


def test_multitarget_latency_summary_references_csv(tmp_path):
    report = run_multitarget_state_sync_latency_acceptance(frame_count=2)
    output = save_multitarget_state_sync_latency_report(report, tmp_path)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["records_file"] == "state_sync_latency.csv"
    assert "records" not in summary
    assert (output / "state_sync_latency.csv").is_file()
