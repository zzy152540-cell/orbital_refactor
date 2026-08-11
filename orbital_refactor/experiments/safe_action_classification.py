from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.counterfactual_action_value import (
    CounterfactualActionValueDataset,
    LinearActionValueModel,
    ShallowTreeActionValueModel,
    _fit_regression_tree,
)
from experiments.graph_action_feature_separability import binary_auc


@dataclass(frozen=True)
class SafeActionClassificationMetrics:
    sample_count: int
    positive_rate: float
    predicted_positive_rate: float
    auc: float | None
    balanced_accuracy: float
    precision: float | None
    recall: float | None


@dataclass(frozen=True)
class SafeActionClassificationResult:
    action_kind: str
    model_kind: str
    training_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    threshold: float
    validation: SafeActionClassificationMetrics
    test: SafeActionClassificationMetrics


def fit_safe_action_classifier(
    dataset: CounterfactualActionValueDataset,
    *,
    action_kind: str,
    model_kind: str,
    training_seeds: tuple[int, ...],
    validation_seeds: tuple[int, ...],
    test_seeds: tuple[int, ...],
    ridge: float = 1.0,
    maximum_depth: int = 3,
    minimum_leaf_size: int = 8,
) -> SafeActionClassificationResult:
    """Fit and calibrate a causal classifier with strict seed isolation."""

    partitions = tuple(map(set, (
        training_seeds, validation_seeds, test_seeds,
    )))
    if any(not values for values in partitions) or any(
        partitions[left] & partitions[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise ValueError("Training, validation, and test seeds must be disjoint.")
    if set().union(*partitions) - set(dataset.seed_by_row):
        raise ValueError("Requested seeds are absent from the dataset.")
    if model_kind not in {"linear", "tree"}:
        raise ValueError("model_kind must be linear or tree.")

    indices = {
        name: np.asarray([
            index for index, (seed, record) in enumerate(zip(
                dataset.seed_by_row, dataset.records
            ))
            if seed in seeds and record.action_kind == action_kind
        ], dtype=int)
        for name, seeds in zip(
            ("training", "validation", "test"), partitions
        )
    }
    if any(not len(value) for value in indices.values()):
        raise ValueError("Every split requires rows for the requested action kind.")
    labels = np.asarray([
        record.position_rmse_reduction > 0.0
        and record.nees_calibration_improvement >= 0.0
        and record.nees_coverage_calibration_improvement >= 0.0
        for record in dataset.records
    ], dtype=float)
    train = indices["training"]
    if model_kind == "linear":
        model = _fit_linear_classifier(
            dataset, train, labels[train], ridge=ridge
        )
    else:
        root = _fit_regression_tree(
            dataset.features[train], labels[train],
            depth=0, maximum_depth=maximum_depth,
            minimum_leaf_size=minimum_leaf_size,
        )
        model = ShallowTreeActionValueModel(
            dataset.feature_names, root, maximum_depth, minimum_leaf_size
        )

    validation_scores = model.predict(dataset.features[indices["validation"]])
    threshold = _calibrate_threshold(
        validation_scores, labels[indices["validation"]]
    )
    return SafeActionClassificationResult(
        action_kind=action_kind,
        model_kind=model_kind,
        training_seeds=tuple(training_seeds),
        validation_seeds=tuple(validation_seeds),
        test_seeds=tuple(test_seeds),
        threshold=threshold,
        validation=_classification_metrics(
            validation_scores, labels[indices["validation"]], threshold
        ),
        test=_classification_metrics(
            model.predict(dataset.features[indices["test"]]),
            labels[indices["test"]],
            threshold,
        ),
    )


def _fit_linear_classifier(dataset, indices, labels, *, ridge):
    if ridge < 0.0:
        raise ValueError("ridge cannot be negative.")
    features = dataset.features[indices]
    mean = np.mean(features, axis=0)
    scale = np.where(np.std(features, axis=0) > 1e-12,
                     np.std(features, axis=0), 1.0)
    design = np.column_stack((np.ones(len(features)), (features - mean) / scale))
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    parameters = np.linalg.pinv(
        design.T @ design + penalty
    ) @ design.T @ labels
    return LinearActionValueModel(
        feature_names=dataset.feature_names,
        coefficients=tuple(parameters[1:]),
        intercept=float(parameters[0]),
        feature_mean=tuple(mean),
        feature_scale=tuple(scale),
        ridge=float(ridge),
    )


def _calibrate_threshold(scores, labels):
    candidates = (-np.inf, *sorted(set(float(value) for value in scores)), np.inf)
    evaluated = [
        (_classification_metrics(scores, labels, threshold), threshold)
        for threshold in candidates
    ]
    best, threshold = max(evaluated, key=lambda item: (
        item[0].balanced_accuracy,
        -item[0].predicted_positive_rate,
        item[1],
    ))
    return float(threshold)


def _classification_metrics(scores, labels, threshold):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    predicted = scores > threshold
    true_positive = int(np.sum(predicted & labels))
    false_positive = int(np.sum(predicted & ~labels))
    positive = int(np.sum(labels))
    negative = len(labels) - positive
    true_negative = int(np.sum(~predicted & ~labels))
    recall = None if positive == 0 else true_positive / positive
    specificity = None if negative == 0 else true_negative / negative
    balanced = np.mean([
        value for value in (recall, specificity) if value is not None
    ])
    return SafeActionClassificationMetrics(
        sample_count=len(labels),
        positive_rate=float(np.mean(labels)),
        predicted_positive_rate=float(np.mean(predicted)),
        auc=binary_auc(scores, labels),
        balanced_accuracy=float(balanced),
        precision=(
            None if true_positive + false_positive == 0
            else true_positive / (true_positive + false_positive)
        ),
        recall=recall,
    )
