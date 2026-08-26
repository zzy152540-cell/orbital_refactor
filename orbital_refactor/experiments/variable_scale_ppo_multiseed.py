from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev


BRANCHES = ("random_init", "warm_start")
NODE_COUNTS = (5, 10, 20)
ACTION_KINDS = ("keep", "add", "swap", "remove")


def summarize_variable_scale_ppo_seeds(
    summary_paths: tuple[str | Path, ...],
) -> dict:
    """Aggregate frozen-condition PPO comparisons across policy seeds."""

    if len(summary_paths) < 2:
        raise ValueError("A multi-seed summary requires at least two runs.")
    runs = []
    reference_configuration = None
    reference_conditions = None
    for path in map(Path, summary_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        configuration = payload["configuration"]
        conditions = payload["evaluation_conditions"]
        frozen_configuration = dict(configuration)
        policy_seed = int(frozen_configuration.pop("policy_seed"))
        if reference_configuration is None:
            reference_configuration = frozen_configuration
            reference_conditions = conditions
        elif (
            frozen_configuration != reference_configuration
            or conditions != reference_conditions
        ):
            raise ValueError(
                "Multi-seed runs must differ only by policy seed."
            )
        branches = {
            branch: _summarize_branch(payload[branch]) for branch in BRANCHES
        }
        runs.append({
            "policy_seed": policy_seed,
            "source": str(path),
            "branches": branches,
        })
    runs.sort(key=lambda item: item["policy_seed"])
    if len({item["policy_seed"] for item in runs}) != len(runs):
        raise ValueError("Policy seeds must be unique.")
    return {
        "role": "variable_scale_ppo_multiseed_development_summary",
        "frozen_configuration_without_policy_seed": reference_configuration,
        "evaluation_conditions": reference_conditions,
        "runs": runs,
        "aggregate": {
            branch: _aggregate_branch(runs, branch) for branch in BRANCHES
        },
    }


def write_variable_scale_ppo_multiseed_summary(
    summary_paths: tuple[str | Path, ...], output_path: str | Path,
) -> Path:
    output = Path(output_path)
    summary = summarize_variable_scale_ppo_seeds(summary_paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
    )
    return output


def generate_variable_scale_ppo_multiseed_visualization(
    summary: dict | str | Path, output_path: str | Path,
) -> Path:
    """Render seed stability, scale response, actions, and Critic diagnostics."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not isinstance(summary, dict):
        summary = json.loads(Path(summary).read_text(encoding="utf-8"))
    output = Path(output_path)
    runs = summary["runs"]
    seeds = [item["policy_seed"] for item in runs]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    width = 0.36
    positions = np.arange(len(seeds))
    for offset, branch, label, color in (
        (-width / 2, "random_init", "random", "tab:gray"),
        (width / 2, "warm_start", "warm start", "tab:blue"),
    ):
        values = [item["branches"][branch]["mean_rmse_improvement"]
                  for item in runs]
        axes[0, 0].bar(positions + offset, values, width, label=label, color=color)
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set(title="Mean deterministic RMSE improvement by seed",
                   xlabel="policy seed", ylabel="RMSE improvement (m)")
    axes[0, 0].set_xticks(positions, seeds)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3, axis="y")

    warm_scales = [[item["branches"]["warm_start"]["by_node_count"][str(n)]
                    ["mean_rmse_improvement"] for item in runs]
                   for n in NODE_COUNTS]
    axes[0, 1].boxplot(warm_scales, tick_labels=[str(n) for n in NODE_COUNTS])
    for index, values in enumerate(warm_scales, start=1):
        axes[0, 1].scatter([index] * len(values), values, color="tab:blue", zorder=3)
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set(title="Warm-start scale stability", xlabel="node count",
                   ylabel="RMSE improvement (m)")
    axes[0, 1].grid(alpha=0.3, axis="y")

    for branch, label, color in (
        ("random_init", "random", "tab:gray"),
        ("warm_start", "warm start", "tab:blue"),
    ):
        values = [item["branches"][branch]["final_critic_explained_variance"]
                  for item in runs]
        axes[1, 0].plot(seeds, values, marker="o", label=label, color=color)
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set(title="Final Critic explained variance",
                   xlabel="policy seed", ylabel="explained variance")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    bottom = np.zeros(len(seeds))
    colors = ("tab:gray", "tab:green", "tab:orange", "tab:red")
    for kind, color in zip(ACTION_KINDS, colors):
        values = [item["branches"]["warm_start"]["action_kind_counts"][kind]
                  for item in runs]
        axes[1, 1].bar(positions, values, bottom=bottom, label=kind, color=color)
        bottom += np.asarray(values)
    axes[1, 1].set(title="Warm-start deterministic actions",
                   xlabel="policy seed", ylabel="decisions")
    axes[1, 1].set_xticks(positions, seeds)
    axes[1, 1].legend()

    warm = summary["aggregate"]["warm_start"]
    figure.suptitle(
        "Variable-scale PPO multi-seed development summary\n"
        f"mean improvement={warm['mean_rmse_improvement']:.5f} m, "
        f"improved={warm['improved_episode_count']}/{warm['episode_count']}, "
        f"worst={warm['worst_rmse_improvement']:.5f} m",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)
    return output


def _summarize_branch(branch: dict) -> dict:
    scales = branch["evaluation_by_node_count"]
    episode_count = sum(int(item["episode_count"]) for item in scales.values())
    weighted_improvement = sum(
        float(item["mean_rmse_improvement"]) * int(item["episode_count"])
        for item in scales.values()
    )
    return {
        "episode_count": episode_count,
        "improved_episode_count": sum(
            int(item["improved_episode_count"]) for item in scales.values()
        ),
        "mean_rmse_improvement": weighted_improvement / episode_count,
        "worst_rmse_improvement": min(
            float(item["worst_rmse_improvement"]) for item in scales.values()
        ),
        "final_critic_explained_variance": float(
            branch["batch_diagnostics"][-1]["update"]
            ["explained_variance_before_update"]
        ),
        "action_kind_counts": {
            kind: sum(
                int(item["action_kind_counts"].get(kind, 0))
                for item in scales.values()
            )
            for kind in ACTION_KINDS
        },
        "by_node_count": {
            str(node_count): scales[str(node_count)] for node_count in NODE_COUNTS
        },
    }


def _aggregate_branch(runs: list[dict], branch: str) -> dict:
    summaries = [item["branches"][branch] for item in runs]
    improvements = [item["mean_rmse_improvement"] for item in summaries]
    return {
        "seed_count": len(runs),
        "episode_count": sum(item["episode_count"] for item in summaries),
        "improved_episode_count": sum(
            item["improved_episode_count"] for item in summaries
        ),
        "mean_rmse_improvement": mean(improvements),
        "seed_standard_deviation_rmse_improvement": stdev(improvements),
        "worst_rmse_improvement": min(
            item["worst_rmse_improvement"] for item in summaries
        ),
        "final_critic_explained_variance_by_seed": [
            item["final_critic_explained_variance"] for item in summaries
        ],
        "action_kind_counts": {
            kind: sum(item["action_kind_counts"][kind] for item in summaries)
            for kind in ACTION_KINDS
        },
        "mean_rmse_improvement_by_node_count": {
            str(node_count): mean(
                item["by_node_count"][str(node_count)]["mean_rmse_improvement"]
                for item in summaries
            )
            for node_count in NODE_COUNTS
        },
    }
