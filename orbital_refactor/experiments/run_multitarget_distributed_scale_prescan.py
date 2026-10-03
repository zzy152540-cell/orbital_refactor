from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.multitarget_distributed_scale_prescan import (
    run_multitarget_distributed_scale_prescan,
)


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(
        description="Run the target-centric distributed CI scale prescan."
    )
    parser.add_argument("--node-counts", type=int, nargs="+", default=(4, 6, 10))
    parser.add_argument("--target-count", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--topologies", nargs="+", default=("chain", "fully_connected"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/multitarget_distributed_scale_prescan.json"),
    )
    arguments = parser.parse_args(argv)
    summary = run_multitarget_distributed_scale_prescan(
        node_counts=tuple(arguments.node_counts),
        target_count=arguments.target_count,
        epochs=arguments.epochs,
        topologies=tuple(arguments.topologies),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return arguments.output


if __name__ == "__main__":
    main()
