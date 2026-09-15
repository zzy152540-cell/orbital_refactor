from __future__ import annotations

import argparse
from pathlib import Path

from experiments.infrared_image_precision_audit import (
    run_infrared_image_precision_audit,
    save_infrared_image_precision_audit,
)


def main():
    parser = argparse.ArgumentParser(description="Audit infrared image precision.")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/external_acceptance/p08_ir_image_precision_audit"),
    )
    args = parser.parse_args()
    report = run_infrared_image_precision_audit(
        monte_carlo_samples=args.samples
    )
    paths = save_infrared_image_precision_audit(report, args.output)
    feasible = [record for record in report.records if record.jointly_feasible]
    print(f"records={len(report.records)} jointly_feasible={len(feasible)}")
    print(*(str(path) for path in paths), sep="\n")


if __name__ == "__main__":
    main()
