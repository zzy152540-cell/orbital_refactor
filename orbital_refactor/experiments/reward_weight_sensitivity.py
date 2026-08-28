from __future__ import annotations

from collections import Counter, defaultdict
import csv
from pathlib import Path

import numpy as np


ACTION_KINDS = ("keep", "add", "swap", "remove")
RESOURCE_FORMS = ("absolute", "difference_from_keep")


def scan_reward_weight_sensitivity(
    records: tuple[dict[str, object], ...],
    *,
    communication_multipliers=(0.0, 0.25, 0.5, 1.0, 2.0),
    resynchronization_multipliers=(0.0, 0.5, 1.0, 2.0),
    switch_multipliers=(0.0, 0.25, 0.5, 1.0, 2.0),
    resource_forms=RESOURCE_FORMS,
) -> dict[str, object]:
    """Re-score frozen same-state actions under a reward-weight grid."""

    grouped = _group_states(records)
    configurations = []
    for form in resource_forms:
        if form not in RESOURCE_FORMS:
            raise ValueError(f"Unknown resource reward form {form!r}.")
        for communication in communication_multipliers:
            for resynchronization in resynchronization_multipliers:
                for switch in switch_multipliers:
                    configurations.append(_configuration_summary(
                        grouped, resource_form=form,
                        communication_multiplier=float(communication),
                        resynchronization_multiplier=float(resynchronization),
                        switch_multiplier=float(switch),
                    ))
    reference = next(
        item for item in configurations
        if item["resource_form"] == "absolute"
        and item["communication_multiplier"] == 1.0
        and item["resynchronization_multiplier"] == 1.0
        and item["switch_multiplier"] == 1.0
    )
    reference_types = reference["_best_types"]
    for item in configurations:
        best_types = item.pop("_best_types")
        item["best_type_match_fraction_to_current"] = float(np.mean([
            best_types[key] == reference_types[key] for key in reference_types
        ]))
    return {
        "role": "reward_weight_sensitivity_scan",
        "state_count": len(grouped),
        "configuration_count": len(configurations),
        "communication_multipliers": list(communication_multipliers),
        "resynchronization_multipliers": list(resynchronization_multipliers),
        "switch_multipliers": list(switch_multipliers),
        "resource_forms": list(resource_forms),
        "configurations": configurations,
    }


def read_policy_diagnostic_csv(path: str | Path) -> tuple[dict[str, object], ...]:
    integer_fields = {"condition_seed", "noise_seed", "node_count", "decision_index", "action_id"}
    string_fields = {"trajectory", "action_kind"}
    records = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            records.append({
                key: (
                    value if key in string_fields
                    else int(value) if key in integer_fields
                    else float(value)
                )
                for key, value in row.items()
            })
    return tuple(records)


def write_reward_weight_scan_csv(summary, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for configuration in summary["configurations"]:
        common = {
            key: configuration[key] for key in (
                "resource_form", "communication_multiplier",
                "resynchronization_multiplier", "switch_multiplier",
                "mean_best_gain_over_keep",
                "positive_best_nonkeep_fraction",
                "best_type_match_fraction_to_current",
            )
        }
        for node_count, values in configuration["by_node_count"].items():
            rows.append({
                **common, "node_count": node_count,
                "state_count": values["state_count"],
                "mean_best_gain_over_keep": values["mean_best_gain_over_keep"],
                "positive_best_nonkeep_fraction": values[
                    "positive_best_nonkeep_fraction"
                ],
                **{
                    f"best_{kind}_count": values["best_action_kind_counts"].get(kind, 0)
                    for kind in ACTION_KINDS
                },
            })
    fields = tuple(rows[0])
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def generate_reward_weight_scan_figure(summary, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    switch_values = summary["switch_multipliers"]
    communication_values = summary["communication_multipliers"]
    figure, axes = plt.subplots(
        2, 3, figsize=(15, 8), sharex=True, sharey=True,
        layout="constrained",
    )
    for row, form in enumerate(RESOURCE_FORMS):
        for column, node_count in enumerate((5, 10, 20)):
            matrix = np.full((len(switch_values), len(communication_values)), np.nan)
            for item in summary["configurations"]:
                if (
                    item["resource_form"] == form
                    and item["resynchronization_multiplier"] == 1.0
                ):
                    y = switch_values.index(item["switch_multiplier"])
                    x = communication_values.index(item["communication_multiplier"])
                    matrix[y, x] = item["by_node_count"][str(node_count)][
                        "positive_best_nonkeep_fraction"
                    ]
            image = axes[row, column].imshow(
                matrix, origin="lower", vmin=0.0, vmax=1.0, cmap="viridis",
                aspect="auto",
            )
            axes[row, column].set_title(f"{form}, N={node_count}")
            axes[row, column].set_xticks(range(len(communication_values)), communication_values)
            axes[row, column].set_yticks(range(len(switch_values)), switch_values)
            axes[row, column].set_xlabel("Communication weight multiplier")
            axes[row, column].set_ylabel("Switch weight multiplier")
    figure.colorbar(
        image, ax=axes.ravel().tolist(),
        label="States with beneficial non-keep action", shrink=0.85,
    )
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def _group_states(records):
    grouped = defaultdict(list)
    for record in records:
        key = tuple(record[name] for name in (
            "condition_seed", "noise_seed", "node_count", "trajectory", "decision_index"
        ))
        grouped[key].append(record)
    if not grouped or any(not any(
        item["action_kind"] == "keep" for item in values
    ) for values in grouped.values()):
        raise ValueError("Every scanned state must include a keep action.")
    return grouped


def _configuration_summary(
    grouped, *, resource_form, communication_multiplier,
    resynchronization_multiplier, switch_multiplier,
):
    best_types, best_gains = {}, []
    best_nonkeep_positive = []
    by_node = defaultdict(list)
    for key, actions in grouped.items():
        keep = next(item for item in actions if item["action_kind"] == "keep")
        scored = []
        for action in actions:
            if resource_form == "absolute":
                resource = (
                    communication_multiplier * action["communication_penalty"]
                    + resynchronization_multiplier * action["resynchronization_penalty"]
                    + switch_multiplier * action["topology_switch_penalty"]
                )
            else:
                resource = (
                    communication_multiplier * (
                        action["communication_penalty"] - keep["communication_penalty"]
                    )
                    + resynchronization_multiplier * (
                        action["resynchronization_penalty"] - keep["resynchronization_penalty"]
                    )
                    + switch_multiplier * (
                        action["topology_switch_penalty"] - keep["topology_switch_penalty"]
                    )
                )
            scored.append((
                float(action["counterfactual_task_gain"] - resource),
                action["action_kind"], int(action["action_id"]),
            ))
        best = max(scored, key=lambda value: (value[0], -value[2]))
        keep_score = next(value[0] for value in scored if value[1] == "keep")
        nonkeep_scores = [value[0] for value in scored if value[1] != "keep"]
        nonkeep_score = max(nonkeep_scores) if nonkeep_scores else -np.inf
        gain = best[0] - keep_score
        best_types[key] = best[1]
        best_gains.append(gain)
        best_nonkeep_positive.append(nonkeep_score > keep_score)
        by_node[key[2]].append((best[1], gain, nonkeep_score > keep_score))
    return {
        "resource_form": resource_form,
        "communication_multiplier": communication_multiplier,
        "resynchronization_multiplier": resynchronization_multiplier,
        "switch_multiplier": switch_multiplier,
        "mean_best_gain_over_keep": float(np.mean(best_gains)),
        "positive_best_nonkeep_fraction": float(np.mean(best_nonkeep_positive)),
        "best_action_kind_counts": dict(sorted(Counter(best_types.values()).items())),
        "by_node_count": {
            str(node_count): {
                "state_count": len(values),
                "mean_best_gain_over_keep": float(np.mean([item[1] for item in values])),
                "positive_best_nonkeep_fraction": float(np.mean([item[2] for item in values])),
                "best_action_kind_counts": dict(sorted(Counter(
                    item[0] for item in values
                ).items())),
            }
            for node_count, values in sorted(by_node.items())
        },
        "_best_types": best_types,
    }
