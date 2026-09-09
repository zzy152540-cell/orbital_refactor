from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from experiments.topology_snapshot_counterfactual import (
    SnapshotActionTensorDataset,
)


@dataclass(frozen=True)
class SnapshotCostAwareWeights:
    communication: float = 0.0025
    topology_switch: float = 0.001
    resynchronization: float = 0.001
    communication_scale: float = 4.0
    topology_switch_scale: float = 1.0
    resynchronization_scale: float = 2.0

    def validate(self):
        penalties = (
            self.communication, self.topology_switch,
            self.resynchronization,
        )
        scales = (
            self.communication_scale, self.topology_switch_scale,
            self.resynchronization_scale,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in penalties):
            raise ValueError("Snapshot penalty weights must be nonnegative.")
        if any(not np.isfinite(value) or value <= 0.0 for value in scales):
            raise ValueError("Snapshot cost scales must be positive.")


@dataclass(frozen=True)
class SnapshotCostAwareOracleSummary:
    group_count: int
    keep_oracle_rate: float
    non_keep_oracle_rate: float
    mean_oracle_utility: float
    action_kind_rates: tuple[tuple[str, float], ...]


def apply_stage1_cost_aware_utility(
    dataset: SnapshotActionTensorDataset, *, weights=None,
) -> SnapshotActionTensorDataset:
    """Replace target column zero with PPO-aligned improvement over keep."""
    selected = weights or SnapshotCostAwareWeights()
    selected.validate()
    required = (
        "position_rmse_reduction_vs_keep", "transmitted_messages",
        "resynchronization_count", "topology_switch_count",
    )
    if any(name not in dataset.target_names for name in required):
        raise ValueError("Snapshot dataset lacks cost-aware utility targets.")
    index = {name: dataset.target_names.index(name) for name in required}
    groups = []
    for group in dataset.groups:
        keep_indices = [i for i, kind in enumerate(group.action_kinds)
                        if kind == "keep"]
        if len(keep_indices) != 1:
            raise ValueError("Each snapshot requires exactly one keep action.")
        keep = keep_indices[0]
        targets = group.targets.copy()
        utility = targets[:, index["position_rmse_reduction_vs_keep"]].copy()
        utility -= selected.communication * (
            targets[:, index["transmitted_messages"]]
            - targets[keep, index["transmitted_messages"]]
        ) / selected.communication_scale
        utility -= selected.resynchronization * (
            targets[:, index["resynchronization_count"]]
            - targets[keep, index["resynchronization_count"]]
        ) / selected.resynchronization_scale
        utility -= selected.topology_switch * (
            targets[:, index["topology_switch_count"]]
            - targets[keep, index["topology_switch_count"]]
        ) / selected.topology_switch_scale
        targets[:, 0] = utility
        targets.setflags(write=False)
        groups.append(replace(group, targets=targets))
    names = list(dataset.target_names)
    names[0] = "stage1_penalized_improvement_vs_keep"
    return replace(
        dataset,
        feature_version=dataset.feature_version + "-stage1-utility",
        target_names=tuple(names), groups=tuple(groups),
    )


def apply_stage1_robust_lcb_utility(
    dataset: SnapshotActionTensorDataset, *, risk_coefficient: float,
    weights=None,
) -> SnapshotActionTensorDataset:
    """Use robust gain mean minus beta times its standard deviation."""
    beta = float(risk_coefficient)
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("Risk coefficient must be finite and nonnegative.")
    mean_name = "position_rmse_reduction_mean_vs_keep"
    deviation_name = "position_rmse_reduction_standard_deviation"
    if mean_name not in dataset.target_names or deviation_name not in (
        dataset.target_names
    ):
        raise ValueError("Robust LCB requires gain mean and deviation targets.")
    mean_index = dataset.target_names.index(mean_name)
    deviation_index = dataset.target_names.index(deviation_name)
    groups = []
    for group in dataset.groups:
        targets = group.targets.copy()
        targets[:, 0] = (
            targets[:, mean_index] - beta * targets[:, deviation_index]
        )
        targets.setflags(write=False)
        groups.append(replace(group, targets=targets))
    lcb = replace(dataset, groups=tuple(groups))
    result = apply_stage1_cost_aware_utility(lcb, weights=weights)
    names = list(result.target_names)
    names[0] = "stage1_robust_lcb_improvement_vs_keep"
    return replace(
        result, feature_version=(
            dataset.feature_version + f"-stage1-lcb-beta{beta:g}"
        ), target_names=tuple(names),
    )


def summarize_cost_aware_oracles(dataset):
    if not dataset.groups:
        raise ValueError("Oracle summary requires a nonempty dataset.")
    kinds, gains = [], []
    for group in dataset.groups:
        oracle = int(np.argmax(group.targets[:, 0]))
        kinds.append(group.action_kinds[oracle])
        gains.append(float(group.targets[oracle, 0]))
    labels = ("keep", "add", "swap", "remove")
    rates = tuple(
        (label, float(np.mean([kind == label for kind in kinds])))
        for label in labels
    )
    rate = dict(rates)["keep"]
    return SnapshotCostAwareOracleSummary(
        group_count=len(dataset.groups), keep_oracle_rate=rate,
        non_keep_oracle_rate=1.0 - rate,
        mean_oracle_utility=float(np.mean(gains)),
        action_kind_rates=rates,
    )


def scan_stage1_robust_lcb_risk(dataset, risk_coefficients):
    values = tuple(float(value) for value in risk_coefficients)
    if not values or len(set(values)) != len(values):
        raise ValueError("Risk scan coefficients must be unique and nonempty.")
    return tuple((value, summarize_cost_aware_oracles(
        apply_stage1_robust_lcb_utility(
            dataset, risk_coefficient=value,
        )
    )) for value in values)
