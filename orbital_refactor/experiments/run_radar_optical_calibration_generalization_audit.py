from __future__ import annotations

import argparse
from pathlib import Path

from experiments.radar_optical_calibration_generalization_audit import (
    run_radar_optical_calibration_generalization_audit,
    save_radar_optical_calibration_generalization_audit,
)


def main():
    parser = argparse.ArgumentParser(description="Audit calibration generalization.")
    parser.add_argument("--calibration-samples", type=int, default=300)
    parser.add_argument("--validation-samples", type=int, default=200)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/external_acceptance/p08_calibration_generalization"),
    )
    args = parser.parse_args()
    report = run_radar_optical_calibration_generalization_audit(
        calibration_samples=args.calibration_samples,
        validation_samples=args.validation_samples,
    )
    paths = save_radar_optical_calibration_generalization_audit(
        report, args.output,
    )
    worst = sorted(
        report.records, key=lambda record: record.mean_nis_distance_from_two,
        reverse=True,
    )[:6]
    for record in worst:
        print(
            record.modality, record.validation_profile,
            record.operating_point, f"nis={record.calibrated_mean_nis:.3f}",
            f"exceed={record.calibrated_nis_95_exceedance_fraction:.3f}",
        )
    print(*(str(path) for path in paths), sep="\n")


if __name__ == "__main__":
    main()
