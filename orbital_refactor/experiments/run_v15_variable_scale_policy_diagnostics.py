from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.topology_ppo_stage1 import Stage1PenaltyWeights
from experiments.variable_scale_policy_diagnostics import (
    generate_policy_diagnostic_figure,
    load_variable_scale_model,
    run_variable_scale_policy_diagnostics,
    write_policy_diagnostic_csv,
)
from experiments.variable_scale_topology_curriculum import (
    VariableScaleTopologyCurriculum,
)


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(
        description="Audit V15 reward terms, action values, and scale baselines."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--randomize-walker-initialization", action="store_true")
    parser.add_argument("--branch", choices=("warm_start", "random_init"), default="warm_start")
    parser.add_argument("--condition-seeds", type=int, nargs="+", default=tuple(range(1500, 1507)))
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=(0,))
    parser.add_argument("--decision-indices", type=int, nargs="+", default=(0, 3, 6, 9))
    parser.add_argument("--horizon-decisions", type=int, default=2)
    parser.add_argument("--maximum-actions-per-kind", type=int, default=2)
    parser.add_argument(
        "--trajectories", choices=("keep", "policy"), nargs="+",
        default=("keep", "policy"),
    )
    arguments = parser.parse_args(argv)
    model, configuration = load_variable_scale_model(
        arguments.checkpoint, branch=arguments.branch,
        condition_seed=arguments.condition_seeds[0],
    )
    weights = Stage1PenaltyWeights(**configuration.get("penalty_weights", {}))
    summary = run_variable_scale_policy_diagnostics(
        model,
        condition_seeds=tuple(arguments.condition_seeds),
        noise_seeds=tuple(arguments.noise_seeds),
        decision_indices=tuple(arguments.decision_indices),
        horizon_decisions=arguments.horizon_decisions,
        maximum_actions_per_kind=arguments.maximum_actions_per_kind,
        curriculum=VariableScaleTopologyCurriculum(
            randomize_walker_initialization=(
                arguments.randomize_walker_initialization
            )
        ),
        trajectories=tuple(arguments.trajectories),
        gamma=float(configuration.get("gamma", 0.99)),
        gae_lambda=float(configuration.get("gae_lambda", 0.95)),
        penalty_weights=weights,
    )
    summary["source_checkpoint"] = str(arguments.checkpoint)
    summary["branch"] = arguments.branch
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    if arguments.csv is not None:
        write_policy_diagnostic_csv(summary, arguments.csv)
    if arguments.figure is not None:
        generate_policy_diagnostic_figure(summary, arguments.figure)
    print(json.dumps({
        "output": str(arguments.output),
        "baseline_by_node_count": summary["baseline_by_node_count"],
        "action_by_node_count": summary["action_by_node_count"],
        "selected_advantage_by_node_count": summary[
            "selected_advantage_by_node_count"
        ],
        "availability_by_node_count_and_trajectory": summary[
            "availability_by_node_count_and_trajectory"
        ],
    }, indent=2, sort_keys=True))
    return arguments.output


if __name__ == "__main__":
    main()
