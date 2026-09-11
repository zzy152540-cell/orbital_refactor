from __future__ import annotations

import argparse
from pathlib import Path

from experiments.external_single_accuracy_acceptance import (
    run_external_single_accuracy_acceptance,
    save_external_single_accuracy_acceptance,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run the paired external-requirement P07 pilot.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/external_acceptance/p07_single_accuracy"),
    )
    arguments = parser.parse_args()
    report = run_external_single_accuracy_acceptance(
        seeds=arguments.seeds, duration=arguments.duration, dt=arguments.dt,
    )
    paths = save_external_single_accuracy_acceptance(report, arguments.output)
    print(f"passed={report.passed}")
    print(
        "mean_position_improvement_over_best_single_percent="
        f"{report.mean_position_improvement_over_best_single_percent:.6f}"
    )
    print(*(str(path) for path in paths), sep="\n")
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
