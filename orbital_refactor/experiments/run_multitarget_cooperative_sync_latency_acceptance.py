"""CLI for the integrated 15-node P09 cooperative synchronization test."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.multitarget_cooperative_sync_latency_acceptance import (
    run_multitarget_cooperative_sync_latency_acceptance,
    save_multitarget_cooperative_sync_latency_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure report-ready through actual ring Consensus-CI application "
            "for 15 logical observer nodes and 15 targets."
        )
    )
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--threshold-ms", type=float, default=100.0)
    parser.add_argument("--consensus-grid-points", type=int, default=31)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/external_acceptance/"
            "p09_integrated_cooperative_sync_latency_120frames"
        ),
    )
    args = parser.parse_args()
    report = run_multitarget_cooperative_sync_latency_acceptance(
        frame_count=args.frames,
        threshold_ms=args.threshold_ms,
        consensus_grid_points=args.consensus_grid_points,
        realtime_pacing=args.realtime,
    )
    output = save_multitarget_cooperative_sync_latency_report(report, args.output)
    print(
        f"report_ready_to_consensus_applied mean={report.mean_sync_ms:.3f} ms "
        f"p95={report.p95_sync_ms:.3f} ms "
        f"p99={report.p99_sync_ms:.3f} ms "
        f"max={report.maximum_sync_ms:.3f} ms"
    )
    print(
        f"serialization={report.mean_serialization_ms:.3f} ms "
        f"send={report.mean_socket_send_ms:.3f} ms "
        f"receiver={report.mean_receiver_processing_ms:.3f} ms "
        f"consensus_ci={report.mean_consensus_ci_ms:.3f} ms"
    )
    print(
        f"protocol_clean={report.protocol_clean} "
        f"threshold_met={report.threshold_met} wrote={output}"
    )


if __name__ == "__main__":
    main()
