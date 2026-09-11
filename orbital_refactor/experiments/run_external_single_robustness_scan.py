from __future__ import annotations

import argparse
from pathlib import Path

from experiments.external_single_robustness_scan import (
    run_external_single_robustness_scan,
    save_external_single_robustness_scan,
)


def main():
    parser = argparse.ArgumentParser(description="Run the P08 CI sensitivity scan.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument(
        "--outage-durations", nargs="+", type=float,
        default=[10.0, 20.0, 40.0, 60.0],
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/external_acceptance/p08_robustness_scan"),
    )
    arguments = parser.parse_args()
    report = run_external_single_robustness_scan(
        seeds=arguments.seeds, duration=arguments.duration, dt=arguments.dt,
        outage_durations=arguments.outage_durations,
    )
    paths = save_external_single_robustness_scan(report, arguments.output)
    print(f"groups={len(report.groups)}")
    print(*(str(path) for path in paths), sep="\n")


if __name__ == "__main__":
    main()
