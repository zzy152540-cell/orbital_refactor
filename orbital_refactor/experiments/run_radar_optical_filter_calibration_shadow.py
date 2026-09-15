from __future__ import annotations

import argparse
from pathlib import Path

from experiments.radar_optical_filter_calibration_shadow import (
    run_radar_optical_filter_calibration_shadow,
    save_radar_optical_filter_calibration_shadow,
)


def main():
    parser = argparse.ArgumentParser(description="Run radar/optical filter shadow.")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-samples", type=int, default=300)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/external_acceptance/p08_radar_optical_filter_shadow"),
    )
    args = parser.parse_args()
    report = run_radar_optical_filter_calibration_shadow(
        duration=args.duration, dt=args.dt, seed=args.seed,
        calibration_samples=args.calibration_samples,
    )
    path = save_radar_optical_filter_calibration_shadow(report, args.output)
    for arm in report.arms:
        print(
            arm.modality, arm.name,
            f"position_rmse={arm.position_rmse_m:.3f}",
            f"relative_rmse={arm.relative_position_rmse_m:.3f}",
            f"nis={arm.mean_nis:.3f}",
            f"valid={arm.valid_measurement_count}",
        )
    print(path)


if __name__ == "__main__":
    main()
