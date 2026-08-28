from __future__ import annotations

import csv
from pathlib import Path


REQUIRED_POLICIES = ("keep", "information_greedy")


def load_deterministic_topology_comparison(path: str | Path) -> dict:
    """Load paired deterministic-policy rows and derive presentation metrics."""

    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        rows = tuple(csv.DictReader(stream))
    paired = {}
    for row in rows:
        seed = int(row["seed"])
        policy = row["policy"]
        if policy in REQUIRED_POLICIES:
            paired.setdefault(seed, {})[policy] = row
    if not paired or any(set(items) != set(REQUIRED_POLICIES)
                         for items in paired.values()):
        raise ValueError("Each seed requires keep and information_greedy rows.")
    seeds = tuple(sorted(paired))
    records = []
    for seed in seeds:
        keep, greedy = (paired[seed][name] for name in REQUIRED_POLICIES)
        keep_rmse = float(keep["final_position_rmse"])
        greedy_rmse = float(greedy["final_position_rmse"])
        records.append({
            "seed": seed,
            "keep_rmse": keep_rmse,
            "greedy_rmse": greedy_rmse,
            "rmse_improvement": keep_rmse - greedy_rmse,
            "rmse_improvement_percent": 100.0 * (keep_rmse - greedy_rmse)
            / keep_rmse,
            "keep_penalized_return": float(keep["cumulative_penalized_return"]),
            "greedy_penalized_return": float(
                greedy["cumulative_penalized_return"]
            ),
            "transmitted_messages": float(greedy["transmitted_messages"]),
            "dropped_messages": float(greedy["dropped_messages"]),
            "replay_count": float(greedy["replay_count"]),
            "resynchronization_count": float(
                greedy["resynchronization_count"]
            ),
            "topology_switch_count": float(greedy["topology_switch_count"]),
            "keep_transmitted_messages": float(keep["transmitted_messages"]),
            "keep_dropped_messages": float(keep["dropped_messages"]),
            "keep_replay_count": float(keep["replay_count"]),
        })
    return {"seeds": seeds, "records": records}


def generate_deterministic_topology_visualization(
    csv_path: str | Path, output_path: str | Path,
) -> Path:
    """Plot a V14-style four-panel deterministic topology comparison."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    comparison = load_deterministic_topology_comparison(csv_path)
    records = comparison["records"]
    seeds = np.asarray(comparison["seeds"])
    output = Path(output_path)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    keep_rmse = [item["keep_rmse"] for item in records]
    greedy_rmse = [item["greedy_rmse"] for item in records]
    for index, seed in enumerate(seeds):
        axes[0, 0].plot(
            (0, 1), (keep_rmse[index], greedy_rmse[index]), marker="o",
            color=f"C{index}", alpha=0.8, label=f"seed {seed}",
        )
    axes[0, 0].set_xticks((0, 1), ("always keep", "information greedy"))
    axes[0, 0].set(title="Final fleet position RMSE", ylabel="RMSE (m)")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    improvements = [item["rmse_improvement_percent"] for item in records]
    axes[0, 1].bar(seeds, improvements, color="tab:blue", alpha=0.85)
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].axhline(
        float(np.mean(improvements)), color="tab:orange", linestyle="--",
        label=f"mean = {np.mean(improvements):.2f}%",
    )
    axes[0, 1].set(title="RMSE improvement over keep", xlabel="noise seed",
                   ylabel="improvement (%)")
    axes[0, 1].set_xticks(seeds)
    axes[0, 1].grid(alpha=0.3, axis="y")
    axes[0, 1].legend()

    resources = (
        ("transmitted", "keep_transmitted_messages", "transmitted_messages"),
        ("dropped", "keep_dropped_messages", "dropped_messages"),
        ("replay", "keep_replay_count", "replay_count"),
    )
    positions, width = np.arange(len(resources)), 0.36
    keep_resources = [np.mean([item[keep] for item in records])
                      for _, keep, _ in resources]
    greedy_resources = [np.mean([item[greedy] for item in records])
                        for _, _, greedy in resources]
    axes[1, 0].bar(positions - width / 2, keep_resources, width,
                   label="always keep", color="tab:gray")
    axes[1, 0].bar(positions + width / 2, greedy_resources, width,
                   label="information greedy", color="tab:blue")
    axes[1, 0].set_xticks(positions, [item[0] for item in resources])
    axes[1, 0].set(title="Mean communication and replay load",
                   ylabel="cumulative count")
    axes[1, 0].grid(alpha=0.3, axis="y")
    axes[1, 0].legend()

    switches = [item["topology_switch_count"] for item in records]
    resyncs = [item["resynchronization_count"] for item in records]
    axes[1, 1].bar(seeds - width / 2, switches, width,
                   label="topology switches", color="tab:purple")
    axes[1, 1].bar(seeds + width / 2, resyncs, width,
                   label="resynchronizations", color="tab:red")
    axes[1, 1].set(title="Topology-control cost", xlabel="noise seed",
                   ylabel="event count")
    axes[1, 1].set_xticks(seeds)
    axes[1, 1].grid(alpha=0.3, axis="y")
    axes[1, 1].legend(loc="upper left")
    return_axis = axes[1, 1].twinx()
    return_delta = [
        item["greedy_penalized_return"] - item["keep_penalized_return"]
        for item in records
    ]
    return_axis.plot(seeds, return_delta, marker="D", linestyle="--",
                     color="tab:orange", label="penalized return delta")
    return_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    return_axis.set_ylabel("greedy - keep penalized return")
    return_axis.legend(loc="lower right")

    figure.suptitle(
        "V15 Walker-20 deterministic topology policy comparison\n"
        "60 filter epochs | RADAR + INFRARED + OPTICAL | 10% loss | 1 s delay",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)
    return output
