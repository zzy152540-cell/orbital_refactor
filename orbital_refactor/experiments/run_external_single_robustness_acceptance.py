from __future__ import annotations

import argparse
from pathlib import Path

from experiments.external_single_robustness_acceptance import (
    run_external_single_robustness_acceptance,
    save_external_single_robustness_acceptance,
)
from adapters.multimodal_sensor_simulator import MultimodalMeasurementSourceConfig
from experiments.single_multi_measurement_source_shadow import (
    build_moderate_calibrated_measurement_source,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run the paired external-requirement P08 pilot.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--calibrated-raw", action="store_true")
    parser.add_argument("--calibration-samples", type=int, default=300)
    parser.add_argument("--infrared-extent", action="store_true")
    parser.add_argument("--target-diameter", type=float, default=10.0)
    parser.add_argument("--extent-fractional-sigma", type=float, default=0.1)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/external_acceptance/p08_single_robustness"),
    )
    arguments = parser.parse_args()
    source_config = None
    if arguments.calibrated_raw:
        sensors, calibrations = build_moderate_calibrated_measurement_source(
            calibration_samples=arguments.calibration_samples,
        )
        source_config = MultimodalMeasurementSourceConfig(
            source="raw_frontend", sensors=sensors,
            covariance_calibration_by_modality=calibrations,
        )
    report = run_external_single_robustness_acceptance(
        seeds=arguments.seeds, duration=arguments.duration, dt=arguments.dt,
        measurement_source_config=source_config,
        infrared_extent_enabled=arguments.infrared_extent,
        infrared_effective_target_diameter_m=arguments.target_diameter,
        infrared_extent_fractional_sigma=arguments.extent_fractional_sigma,
    )
    paths = save_external_single_robustness_acceptance(report, arguments.output)
    print(f"passed={report.passed}")
    for group in report.groups:
        print(
            f"missing_{group.missing_modality}: "
            f"mean_increase={group.mean_position_rmse_increase_percent:.6f}% "
            f"threshold_met={group.threshold_met}"
        )
    print(*(str(path) for path in paths), sep="\n")
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
