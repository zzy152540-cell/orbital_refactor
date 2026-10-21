"""CLI for independent-node P09 synchronization latency testing."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.multitarget_distributed_node_sync_latency_acceptance import (
    run_multitarget_distributed_node_sync_latency_acceptance,
    save_multitarget_distributed_node_sync_latency_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one process per logical observer node, exchange target reports "
            "over ring-neighbor UDP links, and apply local Consensus-CI."
        )
    )
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--threshold-ms", type=float, default=100.0)
    parser.add_argument("--consensus-grid-points", type=int, default=31)
    parser.add_argument("--worker-linear-algebra-threads", type=int, default=1)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/external_acceptance/"
            "p09_distributed_node_sync_latency_120frames"
        ),
    )
    args = parser.parse_args()
    report = run_multitarget_distributed_node_sync_latency_acceptance(
        frame_count=args.frames,
        threshold_ms=args.threshold_ms,
        consensus_grid_points=args.consensus_grid_points,
        worker_linear_algebra_threads=args.worker_linear_algebra_threads,
        realtime_pacing=args.realtime,
    )
    output = save_multitarget_distributed_node_sync_latency_report(
        report, args.output
    )
    print(
        f"node_sync mean={report.mean_sync_ms:.3f} ms "
        f"p95={report.p95_sync_ms:.3f} ms "
        f"p99={report.p99_sync_ms:.3f} ms "
        f"max={report.maximum_sync_ms:.3f} ms"
    )
    print(
        f"all_nodes_applied mean={report.mean_frame_all_nodes_applied_ms:.3f} ms "
        f"p95={report.p95_frame_all_nodes_applied_ms:.3f} ms "
        f"max={report.maximum_frame_all_nodes_applied_ms:.3f} ms"
    )
    print(
        f"serialization={report.mean_serialization_ms:.3f} ms "
        f"send={report.mean_socket_send_ms:.3f} ms "
        f"receive_decode={report.mean_neighbor_receive_and_decode_ms:.3f} ms "
        f"consensus_ci={report.mean_consensus_ci_ms:.3f} ms"
    )
    print(
        f"protocol_clean={report.protocol_clean} "
        f"threshold_met={report.threshold_met} wrote={output}"
    )


if __name__ == "__main__":
    main()
