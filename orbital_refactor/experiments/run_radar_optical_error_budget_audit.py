from __future__ import annotations

import argparse
from pathlib import Path

from experiments.radar_optical_error_budget_audit import (
    run_radar_optical_error_budget_audit,
    save_radar_optical_error_budget_audit,
)


def main():
    parser = argparse.ArgumentParser(description="Audit radar and optical errors.")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/external_acceptance/p08_radar_optical_error_budget"),
    )
    args = parser.parse_args()
    report = run_radar_optical_error_budget_audit(
        monte_carlo_samples=args.samples,
    )
    paths = save_radar_optical_error_budget_audit(report, args.output)
    print(f"records={len(report.records)}")
    print(*(str(path) for path in paths), sep="\n")


if __name__ == "__main__":
    main()
