from __future__ import annotations

import json
from pathlib import Path

import numpy as np


BRANCHES = (("random_init", "random", "tab:gray"),
            ("warm_start", "warm start", "tab:blue"))


def generate_variable_scale_ppo_training_visualization(
    summary_path: str | Path, output_path: str | Path,
) -> Path:
    """Plot PPO optimization, scale-wise evaluation, and action diagnostics."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_path, output_path = Path(summary_path), Path(output_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    _plot_batches(axes[0, 0], summary, "explained_variance_before_update",
                  "Critic explained variance")
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    _plot_losses(axes[0, 1], summary)
    _plot_batches(axes[0, 2], summary, "approximate_kl", "Approximate KL")
    _plot_episode_returns(axes[1, 0], summary)
    _plot_evaluation(axes[1, 1], summary)
    _plot_evaluation_actions(axes[1, 2], summary)
    configuration = summary.get("configuration", {})
    figure.suptitle(
        "Variable-scale PPO training diagnostics\n"
        f"episodes={configuration.get('training_episodes', '?')}, "
        f"counterfactual={configuration.get('counterfactual_keep_reward', False)}, "
        f"scale calibration={bool(configuration.get('critic_scale_calibration_node_counts'))}",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    return output_path


def _plot_batches(axis, summary, field, title):
    for key, label, color in BRANCHES:
        batches = summary[key]["batch_diagnostics"]
        x = [int(item["batch_end"]) for item in batches]
        y = [float(item["update"][field]) for item in batches]
        axis.plot(x, y, marker="o", label=label, color=color)
    axis.set(title=title, xlabel="training episodes")
    axis.grid(alpha=0.3)
    axis.legend()


def _plot_losses(axis, summary):
    for key, label, color in BRANCHES:
        batches = summary[key]["batch_diagnostics"]
        x = [int(item["batch_end"]) for item in batches]
        value = [max(abs(float(item["update"]["value_loss"])), 1e-8)
                 for item in batches]
        policy = [max(abs(float(item["update"]["policy_loss"])), 1e-8)
                  for item in batches]
        axis.plot(x, value, marker="o", color=color, label=f"{label} value")
        axis.plot(x, policy, marker="x", linestyle="--", color=color,
                  label=f"{label} |policy|")
    axis.set(title="PPO losses", xlabel="training episodes", yscale="log")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)


def _plot_episode_returns(axis, summary):
    found = False
    for key, label, color in BRANCHES:
        episodes = summary[key].get("episode_diagnostics", [])
        if not episodes:
            continue
        found = True
        x = [int(item["episode"]) + 1 for item in episodes]
        y = [float(item["task_return"]) for item in episodes]
        axis.plot(x, y, alpha=0.45, color=color, label=label)
        axis.plot(x, _moving_average(y, 5), color=color, linewidth=2)
    if not found:
        scales = (5, 10, 20)
        width = 0.36
        for offset, (key, label, color) in zip((-width / 2, width / 2), BRANCHES):
            values = [float(summary[key]["training"][str(n)]["mean_task_return"])
                      for n in scales]
            axis.bar(np.arange(3) + offset, values, width, label=label, color=color)
        axis.set_xticks(range(3), [str(value) for value in scales])
        axis.set_xlabel("node count (legacy summary)")
    else:
        axis.set_xlabel("episode")
    axis.set_title("Training task return")
    axis.grid(alpha=0.3)
    axis.legend()


def _plot_evaluation(axis, summary):
    scales, width = (5, 10, 20), 0.36
    for offset, (key, label, color) in zip((-width / 2, width / 2), BRANCHES):
        values = [float(summary[key]["evaluation_by_node_count"][str(n)]
                        ["mean_rmse_improvement"]) for n in scales]
        axis.bar(np.arange(3) + offset, values, width, label=label, color=color)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(title="Deterministic evaluation", xlabel="node count",
             ylabel="RMSE improvement (m)")
    axis.set_xticks(range(3), [str(value) for value in scales])
    axis.grid(alpha=0.3, axis="y")
    axis.legend()


def _plot_evaluation_actions(axis, summary):
    kinds = ("keep", "add", "swap", "remove")
    bottom = np.zeros(2)
    colors = ("tab:gray", "tab:green", "tab:orange", "tab:red")
    for kind, color in zip(kinds, colors):
        counts = []
        for key, _label, _branch_color in BRANCHES:
            total = sum(
                int(scale["action_kind_counts"].get(kind, 0))
                for scale in summary[key]["evaluation_by_node_count"].values()
            )
            counts.append(total)
        axis.bar((0, 1), counts, bottom=bottom, label=kind, color=color)
        bottom += np.asarray(counts)
    axis.set(title="Evaluation action counts", ylabel="decisions")
    axis.set_xticks((0, 1), ("random", "warm start"))
    axis.legend(fontsize=8)


def _moving_average(values, window):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    result = np.convolve(values, np.ones(window) / window, mode="valid")
    return np.concatenate((np.full(window - 1, np.nan), result))
