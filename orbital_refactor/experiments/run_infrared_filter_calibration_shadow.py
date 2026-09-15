from __future__ import annotations

import argparse
from pathlib import Path

from experiments.infrared_filter_calibration_shadow import (
    run_infrared_filter_calibration_shadow,
    save_infrared_filter_calibration_shadow,
)


def main():
    parser = argparse.ArgumentParser(description="Run infrared filter shadow.")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-samples", type=int, default=300)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/external_acceptance/p08_ir_filter_calibration_shadow"),
    )
    args = parser.parse_args()
    report = run_infrared_filter_calibration_shadow(
        duration=args.duration, dt=args.dt, seed=args.seed,
        calibration_samples=args.calibration_samples,
    )
    print(save_infrared_filter_calibration_shadow(report, args.output))
    for arm in report.arms:
        print(
            arm.name, f"rmse={arm.position_rmse_m:.6g}",
            f"ir_nis={arm.infrared_mean_nis:.6g}",
        )


if __name__ == "__main__":
    main()
