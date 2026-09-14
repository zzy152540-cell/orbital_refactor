import argparse
from pathlib import Path

from experiments.external_update_frequency_acceptance import (
    run_external_update_frequency_acceptance, save_external_update_frequency_report,
)


def main():
    parser = argparse.ArgumentParser(description="Run Walker-20 P06 timing pre-scan.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=Path("results/external_acceptance/p06_update_frequency_prescan.json"))
    args = parser.parse_args()
    report = run_external_update_frequency_acceptance(seeds=args.seeds, duration=args.duration, dt=args.dt)
    path = save_external_update_frequency_report(report, args.output)
    for group in report.groups:
        print(
            f"{group.profile}: {group.mean_update_frequency_hz:.3f} Hz, "
            f"{group.mean_epoch_latency_ms:.3f} ms"
        )
    print(f"development_threshold_met={report.threshold_met}")
    print(f"formal_passed={report.passed}")
    print(path)


if __name__ == "__main__":
    main()
