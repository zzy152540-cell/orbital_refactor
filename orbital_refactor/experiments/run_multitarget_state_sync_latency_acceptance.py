"""Command-line entry point for the P09 multi-target UDP latency prescan."""

import argparse
from pathlib import Path

from experiments.multitarget_state_sync_latency_acceptance import (
    run_multitarget_state_sync_latency_acceptance,
    save_multitarget_state_sync_latency_report,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the 15-observer/15-target node-to-node P09 UDP latency prescan."
        )
    )
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--threshold-ms", type=float, default=100.0)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "results/external_acceptance/"
            "p09_multitarget_state_sync_latency_prescan"
        ),
    )
    args = parser.parse_args()
    report = run_multitarget_state_sync_latency_acceptance(
        frame_count=args.frames,
        threshold_ms=args.threshold_ms,
        realtime_pacing=args.realtime,
    )
    output = save_multitarget_state_sync_latency_report(report, args.output)
    print(
        f"construction_to_applied mean="
        f"{report.mean_construction_to_applied_ms:.3f} ms "
        f"p95={report.p95_construction_to_applied_ms:.3f} ms "
        f"p99={report.p99_construction_to_applied_ms:.3f} ms "
        f"max={report.maximum_construction_to_applied_ms:.3f} ms"
    )
    print(
        f"serialization={report.mean_serialization_ms:.3f} ms "
        f"send_to_receive={report.mean_send_to_receive_ms:.3f} ms "
        f"receive_to_applied={report.mean_receive_to_applied_ms:.3f} ms"
    )
    print(
        f"payload={report.mean_payload_bytes:.1f} bytes "
        f"protocol_clean={report.protocol_clean} "
        f"threshold_met={report.threshold_met}"
    )
    print(output)


if __name__ == "__main__":
    main()
