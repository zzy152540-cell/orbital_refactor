from __future__ import annotations

import argparse
from pathlib import Path

from experiments.external_node_disconnection_acceptance import (
    run_external_node_disconnection_acceptance,
    save_external_node_disconnection_report,
)


def main():
    parser = argparse.ArgumentParser(description="Run paired Walker-20 R10 acceptance.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--selection-pattern", choices=("dispersed", "adjacent"), default="dispersed")
    parser.add_argument("--output", type=Path, default=Path("results/external_acceptance/r10_node_disconnection"))
    args = parser.parse_args()
    report = run_external_node_disconnection_acceptance(
        seeds=args.seeds, duration=args.duration, dt=args.dt,
        selection_pattern=args.selection_pattern,
    )
    paths = save_external_node_disconnection_report(report, args.output)
    print(f"paired_mean_position_rmse_increase_percent={report.paired_mean_position_rmse_increase_percent:.6f}")
    print(f"threshold_met={report.threshold_met}")
    print(f"formal_sample_size_met={report.formal_sample_size_met}")
    print(f"formal_duration_met={report.formal_duration_met}")
    print(*(str(path) for path in paths), sep="\n")


if __name__ == "__main__":
    main()
