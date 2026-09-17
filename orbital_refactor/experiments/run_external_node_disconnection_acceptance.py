from __future__ import annotations

import argparse
from pathlib import Path

from adapters.multimodal_sensor_simulator import MultimodalMeasurementSourceConfig
from experiments.external_node_disconnection_acceptance import (
    run_external_node_disconnection_acceptance,
    save_external_node_disconnection_report,
)
from experiments.single_multi_measurement_source_shadow import (
    build_moderate_calibrated_measurement_source,
)


def main():
    parser = argparse.ArgumentParser(description="Run paired Walker-20 R10 acceptance.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--calibrated-raw", action="store_true")
    parser.add_argument("--calibration-samples", type=int, default=300)
    parser.add_argument(
        "--selection-pattern",
        choices=("dispersed", "adjacent", "random"),
        default="dispersed",
    )
    parser.add_argument(
        "--selection-seed", type=int, default=0,
        help="Seed used only to choose the fixed four-node random group.",
    )
    parser.add_argument("--output", type=Path, default=Path("results/external_acceptance/r10_node_disconnection"))
    args = parser.parse_args()
    source_config = None
    if args.calibrated_raw:
        sensors, calibrations = build_moderate_calibrated_measurement_source(
            calibration_samples=args.calibration_samples,
        )
        source_config = MultimodalMeasurementSourceConfig(
            source="raw_frontend", sensors=sensors,
            covariance_calibration_by_modality=calibrations,
        )
    report = run_external_node_disconnection_acceptance(
        seeds=args.seeds, duration=args.duration, dt=args.dt,
        selection_pattern=args.selection_pattern,
        selection_seed=args.selection_seed,
        measurement_source_config=source_config,
    )
    paths = save_external_node_disconnection_report(report, args.output)
    print(f"paired_mean_position_rmse_increase_percent={report.paired_mean_position_rmse_increase_percent:.6f}")
    print(f"threshold_met={report.threshold_met}")
    print(f"formal_sample_size_met={report.formal_sample_size_met}")
    print(f"formal_duration_met={report.formal_duration_met}")
    print(*(str(path) for path in paths), sep="\n")


if __name__ == "__main__":
    main()
