from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.topology_control_baselines import HierarchicalGNNPolicy
from experiments.variable_scale_counterfactual_prescan import (
    run_variable_scale_counterfactual_prescan,
)


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(
        description="Run the bounded V15 5/10/20 local counterfactual prescan."
    )
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--condition-seeds", type=int, nargs="+",
        default=(320, 321, 322),
    )
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--horizon-decisions", type=int, default=2)
    parser.add_argument("--maximum-actions-per-kind", type=int, default=4)
    arguments = parser.parse_args(argv)
    summary = run_variable_scale_counterfactual_prescan(
        condition_seeds=tuple(arguments.condition_seeds),
        noise_seed=arguments.noise_seed,
        horizon_decisions=arguments.horizon_decisions,
        maximum_actions_per_kind=arguments.maximum_actions_per_kind,
        reference_policy=HierarchicalGNNPolicy(arguments.reference_checkpoint),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return arguments.output


if __name__ == "__main__":
    main()
