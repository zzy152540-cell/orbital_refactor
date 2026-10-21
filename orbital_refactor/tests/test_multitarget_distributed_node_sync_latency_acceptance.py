import json

from experiments.multitarget_distributed_node_sync_latency_acceptance import (
    run_multitarget_distributed_node_sync_latency_acceptance,
    save_multitarget_distributed_node_sync_latency_report,
)


def test_independent_node_processes_exchange_and_apply_ring_consensus():
    report = run_multitarget_distributed_node_sync_latency_acceptance(
        frame_count=2,
        friendly_observer_count=3,
        hostile_target_count=2,
        sample_interval_seconds=0.5,
        consensus_grid_points=5,
    )

    assert report.independent_node_processes is True
    assert report.process_count == 3
    assert report.worker_linear_algebra_threads == 1
    assert report.consensus_ci_included is True
    assert report.local_estimation_included is False
    assert report.protocol_clean is True
    assert report.node_epoch_sample_count == 6
    assert len(report.records) == 6
    assert all(item.neighbor_batch_count == 2 for item in report.records)
    assert all(item.estimate_count == 2 for item in report.records)
    assert all(item.consensus_ci_ms > 0.0 for item in report.records)
    assert report.mean_frame_all_nodes_applied_ms >= report.mean_sync_ms


def test_distributed_node_report_writes_summary_and_csv(tmp_path):
    report = run_multitarget_distributed_node_sync_latency_acceptance(
        frame_count=1,
        friendly_observer_count=3,
        hostile_target_count=1,
        consensus_grid_points=3,
    )
    output = save_multitarget_distributed_node_sync_latency_report(
        report, tmp_path
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))

    assert summary["records_file"] == "distributed_node_sync_latency.csv"
    assert summary["independent_node_processes"] is True
    assert "records" not in summary
    assert (output / "distributed_node_sync_latency.csv").is_file()
