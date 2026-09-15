from __future__ import annotations

import argparse
from pathlib import Path

from experiments.infrared_precision_robustness_shadow import (
    run_infrared_precision_robustness_shadow,
    save_infrared_precision_robustness_shadow,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run paired optical-outage infrared precision shadow scan."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument(
        "--infrared-angle-sigmas-deg", nargs="+", type=float,
        default=[0.05, 0.025, 0.0125, 0.00625, 0.005],
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/external_acceptance/p08_ir_precision_shadow"),
    )
    args = parser.parse_args()
    report = run_infrared_precision_robustness_shadow(
        seeds=args.seeds, duration=args.duration, dt=args.dt,
        infrared_angle_sigmas_deg=args.infrared_angle_sigmas_deg,
    )
    paths = save_infrared_precision_robustness_shadow(report, args.output)
    for group in report.groups:
        print(
            f"sigma={group.infrared_angle_sigma_deg:g} deg "
            f"mean_increase={group.mean_position_rmse_increase_percent:.3f}% "
            f"threshold_met={group.threshold_met}"
        )
    print(*(str(path) for path in paths), sep="\n")


if __name__ == "__main__":
    main()
