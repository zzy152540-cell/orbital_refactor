from __future__ import annotations

import argparse
from pathlib import Path

from experiments.infrared_filter_outage_shadow import (
    run_infrared_filter_outage_shadow,
    save_infrared_filter_outage_shadow,
)


def main():
    parser = argparse.ArgumentParser(description="Run paired IR outage shadow.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--calibration-samples", type=int, default=300)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/external_acceptance/p08_ir_filter_outage_shadow"),
    )
    args = parser.parse_args()
    report = run_infrared_filter_outage_shadow(
        seeds=args.seeds, duration=args.duration, dt=args.dt,
        calibration_samples=args.calibration_samples,
    )
    print(*save_infrared_filter_outage_shadow(report, args.output), sep="\n")
    for group in report.groups:
        print(
            group.arm,
            f"mean_increase={group.mean_position_rmse_increase_percent:.6g}%",
            f"worst={group.worst_position_rmse_increase_percent:.6g}%",
        )


if __name__ == "__main__":
    main()
