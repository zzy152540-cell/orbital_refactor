from __future__ import annotations

import argparse
from pathlib import Path

from experiments.infrared_error_budget_audit import (
    run_infrared_error_budget_audit,
    save_infrared_error_budget_audit,
)


def main():
    parser = argparse.ArgumentParser(description="Audit infrared image errors.")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/external_acceptance/p08_ir_error_budget_audit"),
    )
    args = parser.parse_args()
    report = run_infrared_error_budget_audit(monte_carlo_samples=args.samples)
    paths = save_infrared_error_budget_audit(report, args.output)
    passed = sum(record.angular_target_met for record in report.records)
    print(f"records={len(report.records)} angular_target_met={passed}")
    print(*(str(path) for path in paths), sep="\n")


if __name__ == "__main__":
    main()
