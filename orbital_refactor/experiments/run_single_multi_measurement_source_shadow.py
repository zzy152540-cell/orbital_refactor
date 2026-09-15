from __future__ import annotations

import argparse
from pathlib import Path

from experiments.single_multi_measurement_source_shadow import (
    build_moderate_calibrated_measurement_source,
    run_single_multi_measurement_source_shadow,
    save_single_multi_measurement_source_shadow,
)


def main():
    parser = argparse.ArgumentParser(description="Compare analytic and raw sources.")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibrated-moderate", action="store_true")
    parser.add_argument("--calibration-samples", type=int, default=300)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/external_acceptance/measurement_source_shadow"),
    )
    args = parser.parse_args()
    sensors = None
    calibrations = None
    if args.calibrated_moderate:
        sensors, calibrations = build_moderate_calibrated_measurement_source(
            calibration_seed=701,
            calibration_samples=args.calibration_samples,
        )
    report = run_single_multi_measurement_source_shadow(
        duration=args.duration, dt=args.dt, seed=args.seed, sensors=sensors,
        covariance_calibration_by_modality=calibrations,
    )
    path = save_single_multi_measurement_source_shadow(report, args.output)
    for arm in report.arms:
        print(
            arm.scope, arm.source, f"rmse={arm.position_rmse_m:.3f}",
            f"valid={arm.valid_radar_count}/{arm.valid_infrared_count}/"
            f"{arm.valid_optical_count}",
            f"nis={arm.radar_mean_nis}/{arm.infrared_mean_nis}/"
            f"{arm.optical_mean_nis}",
        )
    print(path)


if __name__ == "__main__":
    main()
