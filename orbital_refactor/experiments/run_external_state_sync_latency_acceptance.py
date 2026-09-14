import argparse
from pathlib import Path

from experiments.external_state_sync_latency_acceptance import (
    run_external_state_sync_latency_acceptance,
    save_external_state_sync_latency_report,
)


def main():
    parser = argparse.ArgumentParser(description="Run Walker-20 P11 latency scan.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/external_acceptance/p11_state_sync_latency_prescan"),
    )
    args = parser.parse_args()
    report = run_external_state_sync_latency_acceptance(
        seeds=args.seeds, duration=args.duration, dt=args.dt,
    )
    output = save_external_state_sync_latency_report(report, args.output)
    for group in report.groups:
        print(
            f"{group.category}: mean={group.mean_creation_to_applied_ms:.3f} ms, "
            f"p95={group.p95_creation_to_applied_ms:.3f} ms, "
            f"n={group.sample_count}"
        )
    print(f"threshold_met={report.threshold_met}")
    print(f"formal_passed={report.passed}")
    print(output)


if __name__ == "__main__":
    main()
