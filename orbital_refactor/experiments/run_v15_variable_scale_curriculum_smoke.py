from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.counterfactual_action_value import ACTION_KINDS
from experiments.topology_control_baselines import (
    AlwaysKeepPolicy,
    run_topology_control_baseline_episode,
)
from experiments.topology_ppo import (
    build_warm_started_actor_critic,
    collect_topology_rollout,
)
from experiments.topology_ppo_stage1 import build_stage1_environment
from experiments.variable_scale_topology_curriculum import (
    VariableScaleTopologyCurriculum,
)


def run_variable_scale_curriculum_smoke(
    warm_start: str | Path,
    *,
    condition_seeds: tuple[int, ...] = (300, 301, 302),
    noise_seed: int = 0,
) -> dict[str, object]:
    """Run one full-horizon frozen-actor episode at each requested scale."""

    curriculum = VariableScaleTopologyCurriculum()
    model = build_warm_started_actor_critic(
        warm_start, reset_type_head=False,
    )
    records = []
    for condition_seed in condition_seeds:
        configuration = curriculum.configuration_for_condition(condition_seed)
        keep_environment = build_stage1_environment(configuration)
        keep = run_topology_control_baseline_episode(
            keep_environment, AlwaysKeepPolicy(),
            seed=noise_seed, condition_seed=condition_seed,
        )
        model_environment = build_stage1_environment(configuration)
        rollout = collect_topology_rollout(
            model_environment, model, seed=noise_seed,
            condition_seed=condition_seed, deterministic=True,
        )
        kinds = Counter(
            ACTION_KINDS[int(
                transition.group.action_kind_index[transition.action_index]
            )]
            for transition in rollout.transitions
        )
        model_rmse = float(model_environment._metrics()[0])
        conditions = model_environment._episode_conditions
        records.append({
            "condition_seed": int(condition_seed),
            "node_count": configuration.node_count,
            "scenario_type": configuration.scenario_type,
            "walker_plane_count": configuration.walker_plane_count,
            "decision_count": len(rollout.transitions),
            "dynamic_undirected_link_event_count": (
                len(conditions["dynamic_link_events_by_link"]) // 2
            ),
            "keep_final_position_rmse": keep.final_position_rmse,
            "model_final_position_rmse": model_rmse,
            "model_rmse_improvement": keep.final_position_rmse - model_rmse,
            "model_action_kind_counts": dict(sorted(kinds.items())),
            "model_topology_switches": float(sum(
                transition.costs[4] for transition in rollout.transitions
            )),
            "model_resynchronizations": float(sum(
                transition.costs[3] for transition in rollout.transitions
            )),
            "model_fallbacks": float(sum(
                transition.costs[5] for transition in rollout.transitions
            )),
        })
    return {
        "condition_seeds": list(condition_seeds),
        "noise_seed": int(noise_seed),
        "records": records,
    }


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(
        description="Smoke-test one shared GNN actor on the V15 5/10/20 curriculum."
    )
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition-seeds", type=int, nargs="+", default=(300, 301, 302))
    parser.add_argument("--noise-seed", type=int, default=0)
    arguments = parser.parse_args(argv)
    summary = run_variable_scale_curriculum_smoke(
        arguments.warm_start,
        condition_seeds=tuple(arguments.condition_seeds),
        noise_seed=arguments.noise_seed,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return arguments.output


if __name__ == "__main__":
    main()
