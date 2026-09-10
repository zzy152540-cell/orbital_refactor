from __future__ import annotations

import argparse
from pathlib import Path

from experiments.v15_system_formal_matrix import (
    run_v15_system_formal_matrix,
    save_formal_matrix_report,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the V15 formal pilot matrix.")
    parser.add_argument("--nodes", nargs="+", type=int, default=(5, 10, 20))
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument(
        "--conditions", nargs="+",
        choices=(
            "normal", "link_outage_recovery", "packet_loss",
            "communication_delay", "absolute_navigation_dropout",
            "multi_node_navigation_dropout",
            "multi_node_navigation_reference",
            "navigation_communication_degradation",
        ),
        default=(
            "normal", "link_outage_recovery", "packet_loss",
            "communication_delay", "absolute_navigation_dropout",
            "multi_node_navigation_dropout",
            "navigation_communication_degradation",
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/v15_system_formal_pilot.json"),
    )
    args = parser.parse_args(argv)
    report = run_v15_system_formal_matrix(
        node_counts=args.nodes, seed_values=args.seeds,
        conditions=args.conditions, duration=args.duration, dt=args.dt,
    )
    json_path, csv_path = save_formal_matrix_report(
        report, json_path=args.output, csv_path=args.output.with_suffix(".csv"),
    )
    for group in report.groups:
        print(
            f"{group.condition:22s} N={group.node_count:2d} "
            f"pass={group.pass_count}/{group.run_count} "
            f"RMSE={group.mean_position_rmse_m:.4f} +/- "
            f"{group.position_rmse_95_half_width_m:.4f} m"
        )
    print(f"saved report to {json_path}")
    print(f"saved run table to {csv_path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
