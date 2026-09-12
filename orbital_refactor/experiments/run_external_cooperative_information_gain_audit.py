from __future__ import annotations

import argparse
from pathlib import Path

from experiments.external_cooperative_information_gain_audit import (
    run_external_cooperative_information_gain_audit,
    save_external_cooperative_information_gain_audit,
)


def main():
    parser = argparse.ArgumentParser(description="Run read-only P09 information audit.")
    parser.add_argument("--node", default="sat_p03_s01")
    parser.add_argument("--modality", default="RADAR")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/external_acceptance/p09_information_gain_audit"),
    )
    arguments = parser.parse_args()
    report = run_external_cooperative_information_gain_audit(
        node_id=arguments.node, modality=arguments.modality,
        seeds=arguments.seeds, duration=arguments.duration, dt=arguments.dt,
    )
    paths = save_external_cooperative_information_gain_audit(
        report, arguments.output
    )
    print(f"samples={report.sample_count}")
    print(
        "predicted_reduction_to_actual_change_correlation="
        f"{report.predicted_reduction_to_actual_change_correlation:.6f}"
    )
    print(*(str(path) for path in paths), sep="\n")


if __name__ == "__main__":
    main()
