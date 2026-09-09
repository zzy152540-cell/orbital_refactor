from __future__ import annotations

from dataclasses import dataclass

from experiments.cann_snapshot_information_audit import (
    strip_cann_snapshot_features,
)
from experiments.topology_snapshot_counterfactual import (
    SnapshotActionTensorDataset,
    split_topology_snapshot_dataset_by_seed,
)
from experiments.training.graph_action_gnn import train_snapshot_action_network


@dataclass(frozen=True)
class CANNGNNAblationRun:
    random_seed: int
    base_exact_action_match_rate: float
    cann_exact_action_match_rate: float
    base_action_kind_match_rate: float
    cann_action_kind_match_rate: float
    base_mean_oracle_regret: float
    cann_mean_oracle_regret: float


def run_cann_snapshot_gnn_ablation(
    dataset: SnapshotActionTensorDataset, *, training_seeds,
    validation_seeds, random_seeds=(0, 1, 2), epochs=80,
    patience=20, hidden_size=24,
):
    """Compare paired v15.0/v15.1 inputs under identical labels and splits."""
    base = strip_cann_snapshot_features(dataset)
    base_split = split_topology_snapshot_dataset_by_seed(
        base, training_seeds=training_seeds,
        validation_seeds=validation_seeds,
    )
    cann_split = split_topology_snapshot_dataset_by_seed(
        dataset, training_seeds=training_seeds,
        validation_seeds=validation_seeds,
    )
    results = []
    for seed in random_seeds:
        common = dict(
            epochs=epochs, patience=patience, hidden_size=hidden_size,
            random_seed=int(seed), loss_mode="decision",
        )
        base_result = train_snapshot_action_network(
            base_split.training, base_split.validation, **common,
        )
        cann_result = train_snapshot_action_network(
            cann_split.training, cann_split.validation, **common,
        )
        left, right = base_result.best_validation, cann_result.best_validation
        results.append(CANNGNNAblationRun(
            random_seed=int(seed),
            base_exact_action_match_rate=left.exact_action_match_rate,
            cann_exact_action_match_rate=right.exact_action_match_rate,
            base_action_kind_match_rate=left.action_kind_match_rate,
            cann_action_kind_match_rate=right.action_kind_match_rate,
            base_mean_oracle_regret=left.mean_oracle_regret,
            cann_mean_oracle_regret=right.mean_oracle_regret,
        ))
    return tuple(results)
