from __future__ import annotations

import argparse
from pathlib import Path

from experiments.radar_optical_covariance_consistency_audit import (
    run_radar_optical_covariance_consistency_audit,
    save_radar_optical_covariance_consistency_audit,
)


def main():
    parser = argparse.ArgumentParser(description="Audit held-out radar/optical NIS.")
    parser.add_argument("--calibration-samples", type=int, default=300)
    parser.add_argument("--validation-samples", type=int, default=300)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/external_acceptance/p08_radar_optical_covariance"),
    )
    args = parser.parse_args()
    report = run_radar_optical_covariance_consistency_audit(
        calibration_samples=args.calibration_samples,
        validation_samples=args.validation_samples,
    )
    paths = save_radar_optical_covariance_consistency_audit(report, args.output)
    for record in report.records:
        print(
            record.modality, record.operating_point,
            f"fixed={record.fixed_mean_nis:.3f}",
            f"empirical={record.empirical_mean_nis:.3f}",
            f"corrected={record.bias_corrected_mean_nis:.3f}",
        )
    print(*(str(path) for path in paths), sep="\n")


if __name__ == "__main__":
    main()
