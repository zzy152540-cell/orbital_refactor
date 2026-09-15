from __future__ import annotations

import argparse
from pathlib import Path

from experiments.infrared_covariance_consistency_audit import (
    run_infrared_covariance_consistency_audit,
    save_infrared_covariance_consistency_audit,
)


def main():
    parser = argparse.ArgumentParser(description="Audit infrared covariance NIS.")
    parser.add_argument("--calibration-samples", type=int, default=300)
    parser.add_argument("--validation-samples", type=int, default=300)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/external_acceptance/p08_ir_covariance_consistency"),
    )
    args = parser.parse_args()
    report = run_infrared_covariance_consistency_audit(
        calibration_samples=args.calibration_samples,
        validation_samples=args.validation_samples,
    )
    paths = save_infrared_covariance_consistency_audit(report, args.output)
    print(f"records={len(report.records)}")
    print(*(str(path) for path in paths), sep="\n")


if __name__ == "__main__":
    main()
