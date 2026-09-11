from __future__ import annotations

import argparse
from pathlib import Path

from experiments.external_single_robustness_acceptance import (
    run_external_single_robustness_acceptance,
    save_external_single_robustness_acceptance,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run the paired external-requirement P08 pilot.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/external_acceptance/p08_single_robustness"),
    )
    arguments = parser.parse_args()
    report = run_external_single_robustness_acceptance(
        seeds=arguments.seeds, duration=arguments.duration, dt=arguments.dt,
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
