from __future__ import annotations

from dataclasses import dataclass

from experiments.cann_snapshot_feature_subset_ablation import (
    run_cann_snapshot_feature_subset_ablation,
)
from experiments.run_v15_cann_physical_expansion_batch import (
    physical_expansion_cross_validation_folds,
)


@dataclass(frozen=True)
class CANNSnapshotCrossValidationRun:
    fold_index: int
    random_seed: int
    variant: str
    exact_action_match_rate: float
    action_kind_match_rate: float
    mean_oracle_regret: float


def run_cann_snapshot_cross_validation(
    dataset, *, random_seeds=(0,), shuffle_random_seeds=(20260909,),
    epochs=120, patience=30, hidden_size=24,
) -> tuple[CANNSnapshotCrossValidationRun, ...]:
    """Run paired feature ablations over frozen leave-one-batch-out folds."""
    records = []
    for fold in physical_expansion_cross_validation_folds():
        runs = run_cann_snapshot_feature_subset_ablation(
            dataset,
            training_seeds=fold.training,
            validation_seeds=fold.validation,
            random_seeds=random_seeds,
            shuffle_random_seeds=shuffle_random_seeds,
            epochs=epochs,
            patience=patience,
            hidden_size=hidden_size,
        )
        records.extend(CANNSnapshotCrossValidationRun(
            fold_index=fold.fold_index,
            random_seed=run.random_seed,
            variant=run.variant,
            exact_action_match_rate=run.exact_action_match_rate,
            action_kind_match_rate=run.action_kind_match_rate,
            mean_oracle_regret=run.mean_oracle_regret,
        ) for run in runs)
    return tuple(records)
