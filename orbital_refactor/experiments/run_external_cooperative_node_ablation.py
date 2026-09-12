from __future__ import annotations

import argparse
from pathlib import Path

from experiments.external_cooperative_node_ablation import (
    run_external_cooperative_node_ablation,
    save_external_cooperative_node_ablation,
)


def main():
    parser = argparse.ArgumentParser(description="Run targeted P09 node ablation.")
    parser.add_argument("--node", default="sat_p03_s01")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/external_acceptance/p09_node_ablation"),
    )
    arguments = parser.parse_args()
    report = run_external_cooperative_node_ablation(
        node_id=arguments.node, seeds=arguments.seeds,
        duration=arguments.duration, dt=arguments.dt,
    )
    paths = save_external_cooperative_node_ablation(report, arguments.output)
    print("mean steady improvement by variant:")
    for name, value in report.mean_steady_improvement_by_variant.items():
        print(f"  {name}: {value:.6f}%")
    print(*(str(path) for path in paths), sep="\n")


if __name__ == "__main__":
    main()
