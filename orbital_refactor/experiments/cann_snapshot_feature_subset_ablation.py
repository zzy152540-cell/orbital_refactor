from __future__ import annotations

from dataclasses import dataclass

from cooperative.v15_cann_policy_tensor import CANN_NODE_FEATURE_NAMES
from experiments.cann_snapshot_information_audit import (
    mask_cann_snapshot_features,
    shuffle_cann_snapshot_features,
    strip_cann_snapshot_features,
)
from experiments.topology_snapshot_counterfactual import (
    split_topology_snapshot_dataset_by_seed,
)
from experiments.training.graph_action_gnn import train_snapshot_action_network


CANN_CONTINUOUS_FEATURE_NAMES = CANN_NODE_FEATURE_NAMES[:6]


@dataclass(frozen=True)
class CANNFeatureSubsetRun:
    random_seed: int
    variant: str
    exact_action_match_rate: float
    action_kind_match_rate: float
    mean_oracle_regret: float


def run_cann_snapshot_feature_subset_ablation(
    dataset, *, training_seeds, validation_seeds,
    random_seeds=(0, 1, 2), epochs=120, patience=30, hidden_size=24,
    shuffle_random_seeds=(20260909,),
):
    """Compare base, zero, subset, shuffled, and aligned CANN inputs."""
    variants = [
        ("v15_base", strip_cann_snapshot_features(dataset)),
        ("equal_width_zero", mask_cann_snapshot_features(dataset)),
        ("continuous_only", mask_cann_snapshot_features(
            dataset, retained_feature_names=CANN_CONTINUOUS_FEATURE_NAMES,
        )),
    ]
    variants.extend((
        f"shuffled_cann_{int(shuffle_seed)}",
        shuffle_cann_snapshot_features(dataset, random_seed=shuffle_seed),
    ) for shuffle_seed in shuffle_random_seeds)
    variants.append(("full_cann", dataset))
    splits = tuple((name, split_topology_snapshot_dataset_by_seed(
        values, training_seeds=training_seeds,
        validation_seeds=validation_seeds,
    )) for name, values in variants)
    results = []
    for seed in random_seeds:
        for name, split in splits:
            result = train_snapshot_action_network(
                split.training, split.validation,
                epochs=epochs, patience=patience, hidden_size=hidden_size,
                random_seed=int(seed), loss_mode="decision",
            )
            metrics = result.best_validation
            results.append(CANNFeatureSubsetRun(
                random_seed=int(seed), variant=name,
                exact_action_match_rate=metrics.exact_action_match_rate,
                action_kind_match_rate=metrics.action_kind_match_rate,
                mean_oracle_regret=metrics.mean_oracle_regret,
            ))
    return tuple(results)
