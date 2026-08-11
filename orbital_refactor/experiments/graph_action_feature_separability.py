from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.counterfactual_action_value import (
    CounterfactualActionValueDataset,
)


@dataclass(frozen=True)
class FeatureSeparability:
    feature: str
    sample_count: int
    safe_positive_rate: float
    spearman_position_gain: float | None
    safe_positive_auc: float | None
    direction_free_auc: float | None


@dataclass(frozen=True)
class ActionFeatureSeparabilityReport:
    action_kinds: tuple[str, ...]
    decision_epochs: tuple[int, ...] | None
    sample_count: int
    safe_positive_rate: float
    features: tuple[FeatureSeparability, ...]

    def strongest_by_auc(self, count: int = 10):
        available = [
            value for value in self.features
            if value.direction_free_auc is not None
        ]
        return tuple(sorted(
            available,
            key=lambda value: value.direction_free_auc,
            reverse=True,
        )[:count])

    def strongest_by_gain_rank(self, count: int = 10):
        available = [
            value for value in self.features
            if value.spearman_position_gain is not None
        ]
        return tuple(sorted(
            available,
            key=lambda value: abs(value.spearman_position_gain),
            reverse=True,
        )[:count])


def analyze_action_feature_separability(
    dataset: CounterfactualActionValueDataset,
    *,
    action_kinds: tuple[str, ...] = ("remove", "swap"),
    decision_epochs: tuple[int, ...] | None = None,
) -> ActionFeatureSeparabilityReport:
    """Audit causal scalar features before fitting another action model."""

    kinds = tuple(dict.fromkeys(action_kinds))
    if not kinds:
        raise ValueError("At least one action kind is required.")
    epochs = None if decision_epochs is None else tuple(dict.fromkeys(
        int(value) for value in decision_epochs
    ))
    selected = np.asarray([
        record.action_kind in kinds
        and (epochs is None or record.decision_epoch in epochs)
        for record in dataset.records
    ])
    if not np.any(selected):
        raise ValueError("No action rows match the separability slice.")

    gains = dataset.position_rmse_reduction[selected]
    records = [
        record for record, include in zip(dataset.records, selected) if include
    ]
    safe_positive = np.asarray([
        gain > 0.0
        and record.nees_calibration_improvement >= 0.0
        and record.nees_coverage_calibration_improvement >= 0.0
        for gain, record in zip(gains, records)
    ])
    matrix = dataset.features[selected]
    values = []
    for index, name in enumerate(dataset.feature_names):
        feature = matrix[:, index]
        spearman = _correlation(_average_ranks(feature), _average_ranks(gains))
        auc = binary_auc(feature, safe_positive)
        values.append(FeatureSeparability(
            feature=name,
            sample_count=len(feature),
            safe_positive_rate=float(np.mean(safe_positive)),
            spearman_position_gain=spearman,
            safe_positive_auc=auc,
            direction_free_auc=(
                None if auc is None else max(auc, 1.0 - auc)
            ),
        ))
    return ActionFeatureSeparabilityReport(
        action_kinds=kinds,
        decision_epochs=epochs,
        sample_count=int(np.sum(selected)),
        safe_positive_rate=float(np.mean(safe_positive)),
        features=tuple(values),
    )


def binary_auc(scores, labels):
    labels = np.asarray(labels, dtype=bool)
    positive = int(np.sum(labels))
    negative = len(labels) - positive
    if positive == 0 or negative == 0 or np.ptp(scores) == 0.0:
        return None
    ranks = _average_ranks(np.asarray(scores, dtype=float))
    rank_sum = float(np.sum(ranks[labels]))
    return (
        rank_sum - positive * (positive + 1) / 2.0
    ) / (positive * negative)


def _correlation(left, right):
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _average_ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks
