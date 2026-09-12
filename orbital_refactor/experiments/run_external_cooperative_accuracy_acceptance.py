from __future__ import annotations

import argparse
from pathlib import Path

from experiments.external_cooperative_accuracy_acceptance import (
    run_external_cooperative_accuracy_acceptance,
    save_external_cooperative_accuracy_report,
)


def main():
    parser = argparse.ArgumentParser(description="Run paired Walker-20 P09 acceptance.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/external_acceptance/p09_cooperative_accuracy"),
    )
    arguments = parser.parse_args()
    report = run_external_cooperative_accuracy_acceptance(
        seeds=arguments.seeds, duration=arguments.duration, dt=arguments.dt,
    )
    paths = save_external_cooperative_accuracy_report(report, arguments.output)
    print(f"paired_mean_improvement_percent={report.paired_mean_improvement_percent:.6f}")
    print(f"threshold_met={report.threshold_met}")
    print(f"formal_sample_size_met={report.formal_sample_size_met}")
    print(f"formal_duration_met={report.formal_duration_met}")
    print(*(str(path) for path in paths), sep="\n")


if __name__ == "__main__":
    main()
