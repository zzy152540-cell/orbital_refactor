from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cooperative.topology_policy import GraphObservation, TopologyAction
from experiments.action_graph_features import (
    ACTION_GRAPH_FEATURE_NAMES,
    ACTION_PAIR_FEATURE_NAMES,
    action_graph_metrics,
    action_pair_metrics,
)
from experiments.short_horizon_counterfactual_study import (
    ShortHorizonActionRecord,
    ShortHorizonCounterfactualStudy,
)


ACTION_KINDS = ("keep", "add", "swap", "remove")
SCORE_FEATURES = (
    "endpoint_uncertainty_score_gain",
    "projected_uncertainty_score_gain",
    "approximate_trace_score_gain",
    "approximate_logdet_score_gain",
    "negative_distance_score_gain",
    "observation_age_score_gain",
    "recent_nis_score_gain",
    "negative_recent_nis_score_gain",
    "nis_calibration_score_gain",
    "nis_sample_count_score_gain",
    "negative_anomaly_score_gain",
    "communication_risk_score_gain",
)
FEATURE_NAMES = (
    "node_count", "decision_epoch", "horizon_epochs",
    "active_edge_count", "added_edge_count", "removed_edge_count",
    *(f"action_{kind}" for kind in ACTION_KINDS),
    *SCORE_FEATURES,
    *(f"graph_{name}_gain" for name in ACTION_GRAPH_FEATURE_NAMES),
    *(f"pair_{name}" for name in ACTION_PAIR_FEATURE_NAMES),
)


@dataclass(frozen=True)
class CounterfactualActionValueDataset:
    feature_names: tuple[str, ...]
    features: np.ndarray
    position_rmse_reduction: np.ndarray
    nees_calibration_improvement: np.ndarray
    transmitted_message_cost: np.ndarray
    replay_cost: np.ndarray
    topology_change_cost: np.ndarray
    seed_by_row: tuple[int, ...]
    group_by_row: tuple[tuple[int, int, int, int], ...]
    observation_by_group: tuple[
        tuple[tuple[int, int, int, int], GraphObservation], ...
    ]
    records: tuple[ShortHorizonActionRecord, ...]

    def __post_init__(self) -> None:
        row_count = len(self.records)
        if self.features.shape != (row_count, len(self.feature_names)):
            raise ValueError("Action-value feature matrix has the wrong shape.")
        if any(len(value) != row_count for value in (
            self.position_rmse_reduction,
            self.nees_calibration_improvement,
            self.transmitted_message_cost,
            self.replay_cost,
            self.topology_change_cost,
            self.seed_by_row,
            self.group_by_row,
        )):
            raise ValueError("Action-value labels must align with feature rows.")
        if set(self.group_by_row) != {
            group for group, _ in self.observation_by_group
        }:
            raise ValueError("Every action group requires one graph observation.")


@dataclass(frozen=True)
class LinearActionValueModel:
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    ridge: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=float)
        normalized = (
            matrix - np.asarray(self.feature_mean)
        ) / np.asarray(self.feature_scale)
        return self.intercept + normalized @ np.asarray(self.coefficients)


@dataclass(frozen=True)
class ActionValueHoldoutSummary:
    training_seeds: tuple[int, ...]
    holdout_seeds: tuple[int, ...]
    group_count: int
    action_match_rate: float
    action_kind_match_rate: float
    mean_position_rmse_reduction: float
    mean_oracle_regret: float
    positive_rmse_gain_rate: float
    nees_calibration_violation_rate: float
    keep_rate: float
    add_rate: float
    swap_rate: float
    remove_rate: float


@dataclass(frozen=True)
class ActionValueFitResult:
    model: LinearActionValueModel
    holdout: ActionValueHoldoutSummary


@dataclass(frozen=True)
class RegressionTreeNode:
    value: float
    feature_index: int | None = None
    threshold: float | None = None
    left: "RegressionTreeNode | None" = None
    right: "RegressionTreeNode | None" = None


@dataclass(frozen=True)
class ShallowTreeActionValueModel:
    feature_names: tuple[str, ...]
    root: RegressionTreeNode
    maximum_depth: int
    minimum_leaf_size: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=float)
        return np.asarray([self._predict_row(row) for row in matrix])

    def _predict_row(self, row: np.ndarray) -> float:
        node = self.root
        while node.feature_index is not None:
            node = (
                node.left if row[node.feature_index] <= float(node.threshold)
                else node.right
            )
        return float(node.value)


@dataclass(frozen=True)
class TreeActionValueFitResult:
    model: ShallowTreeActionValueModel
    holdout: ActionValueHoldoutSummary


@dataclass(frozen=True)
class FrozenLinearActionValuePolicy:
    model: LinearActionValueModel
    keep_threshold: float
    training_seeds: tuple[int, ...]
    maximum_training_nees_violation_rate: float


@dataclass(frozen=True)
class FrozenPolicyValidation:
    summary: ActionValueHoldoutSummary
    worst_position_rmse_reduction: float
    tenth_percentile_position_rmse_reduction: float
    positive_mean_gain_seed_rate: float


@dataclass(frozen=True)
class ActionValueCrossValidationSummary:
    fold_count: int
    group_count: int
    mean_action_match_rate: float
    mean_action_kind_match_rate: float
    mean_position_rmse_reduction: float
    mean_oracle_regret: float
    positive_rmse_gain_rate: float
    nees_calibration_violation_rate: float
    keep_rate: float
    add_rate: float
    swap_rate: float
    remove_rate: float


def build_counterfactual_action_value_dataset(
    study: ShortHorizonCounterfactualStudy,
) -> CounterfactualActionValueDataset:
    """Flatten causal decision/action features while retaining all outcomes."""

    rows = []
    groups = []
    observation_by_group = dict(study.decision_observations)
    baseline_metrics_by_group = {}
    for record in study.records:
        group = (
            record.node_count, record.seed,
            record.decision_epoch, record.horizon_epochs,
        )
        observation = observation_by_group[group]
        if group not in baseline_metrics_by_group:
            baseline_metrics_by_group[group] = action_graph_metrics(
                observation,
                TopologyAction("baseline", observation.previous_active_edges),
            )
        baseline = baseline_metrics_by_group[group]
        current = action_graph_metrics(
            observation,
            TopologyAction("candidate", record.active_edges),
        )
        pair = action_pair_metrics(
            observation, TopologyAction("candidate", record.active_edges)
        )
        rows.append((
            float(record.node_count),
            float(record.decision_epoch),
            float(record.horizon_epochs),
            float(len(record.active_edges)),
            float(len(record.added_edges)),
            float(len(record.removed_edges)),
            *(1.0 if record.action_kind == kind else 0.0
              for kind in ACTION_KINDS),
            *(float(getattr(record, name)) for name in SCORE_FEATURES),
            *(float(getattr(current, name) - getattr(baseline, name))
              for name in ACTION_GRAPH_FEATURE_NAMES),
            *(float(getattr(pair, name)) for name in ACTION_PAIR_FEATURE_NAMES),
        ))
        groups.append(group)
    records = tuple(study.records)
    return CounterfactualActionValueDataset(
        feature_names=FEATURE_NAMES,
        features=np.asarray(rows, dtype=float),
        position_rmse_reduction=np.asarray([
            record.position_rmse_reduction for record in records
        ]),
        nees_calibration_improvement=np.asarray([
            record.nees_calibration_improvement for record in records
        ]),
        transmitted_message_cost=np.asarray([
            record.transmitted_message_cost for record in records
        ]),
        replay_cost=np.asarray([record.replay_cost for record in records]),
        topology_change_cost=np.asarray([
            record.topology_change_cost for record in records
        ]),
        seed_by_row=tuple(record.seed for record in records),
        group_by_row=tuple(groups),
        observation_by_group=study.decision_observations,
        records=records,
    )


def fit_seed_holdout_linear_action_value(
    dataset: CounterfactualActionValueDataset,
    *,
    training_seeds: tuple[int, ...],
    holdout_seeds: tuple[int, ...],
    ridge: float = 1.0,
    nees_weight: float = 0.0,
    transmitted_message_weight: float = 0.0,
) -> ActionValueFitResult:
    """Fit a transparent ridge baseline with seed-disjoint evaluation."""

    if ridge < 0.0:
        raise ValueError("ridge cannot be negative.")
    training = set(training_seeds)
    holdout = set(holdout_seeds)
    if not training or not holdout or training & holdout:
        raise ValueError("Training and holdout seeds must be nonempty and disjoint.")
    known = set(dataset.seed_by_row)
    if (training | holdout) - known:
        raise ValueError("Requested seeds are absent from the dataset.")
    train_mask = np.asarray([seed in training for seed in dataset.seed_by_row])
    holdout_mask = np.asarray([seed in holdout for seed in dataset.seed_by_row])
    target = (
        dataset.position_rmse_reduction
        + nees_weight * dataset.nees_calibration_improvement
        - transmitted_message_weight * np.maximum(
            dataset.transmitted_message_cost, 0.0
        )
    )
    x_train = dataset.features[train_mask]
    y_train = target[train_mask]
    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    normalized = (x_train - mean) / scale
    design = np.column_stack((np.ones(len(normalized)), normalized))
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    parameters = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y_train
    model = LinearActionValueModel(
        feature_names=dataset.feature_names,
        coefficients=tuple(float(value) for value in parameters[1:]),
        intercept=float(parameters[0]),
        feature_mean=tuple(float(value) for value in mean),
        feature_scale=tuple(float(value) for value in scale),
        ridge=float(ridge),
    )
    holdout_indices = np.flatnonzero(holdout_mask)
    predictions = model.predict(dataset.features[holdout_indices])
    grouped: dict[tuple[int, int, int, int], list[tuple[int, float]]] = {}
    for row_index, prediction in zip(holdout_indices, predictions):
        grouped.setdefault(dataset.group_by_row[row_index], []).append(
            (int(row_index), float(prediction))
        )
    selected = []
    oracle = []
    matches = []
    kind_matches = []
    for candidates in grouped.values():
        selected_index = max(candidates, key=lambda value: value[1])[0]
        oracle_value = max(
            dataset.position_rmse_reduction[index] for index, _ in candidates
        )
        best_indices = {
            index for index, _ in candidates
            if np.isclose(dataset.position_rmse_reduction[index], oracle_value)
        }
        selected.append(selected_index)
        oracle.append(float(oracle_value))
        matches.append(selected_index in best_indices)
        kind_matches.append(dataset.records[selected_index].action_kind in {
            dataset.records[index].action_kind for index in best_indices
        })
    gains = np.asarray([
        dataset.position_rmse_reduction[index] for index in selected
    ])
    return ActionValueFitResult(
        model=model,
        holdout=ActionValueHoldoutSummary(
            training_seeds=tuple(training_seeds),
            holdout_seeds=tuple(holdout_seeds),
            group_count=len(grouped),
            action_match_rate=float(np.mean(matches)) if matches else 0.0,
            action_kind_match_rate=(
                float(np.mean(kind_matches)) if kind_matches else 0.0
            ),
            mean_position_rmse_reduction=(
                float(np.mean(gains)) if gains.size else 0.0
            ),
            mean_oracle_regret=(
                float(np.mean(np.asarray(oracle) - gains))
                if gains.size else 0.0
            ),
            positive_rmse_gain_rate=(
                float(np.mean(gains > 0.0)) if gains.size else 0.0
            ),
            nees_calibration_violation_rate=(
                float(np.mean([
                    dataset.nees_calibration_improvement[index] < 0.0
                    for index in selected
                ])) if selected else 0.0
            ),
            keep_rate=(
                float(np.mean([
                    dataset.records[index].action_kind == "keep"
                    for index in selected
                ])) if selected else 0.0
            ),
            add_rate=float(np.mean([
                dataset.records[index].action_kind == "add" for index in selected
            ])) if selected else 0.0,
            swap_rate=float(np.mean([
                dataset.records[index].action_kind == "swap" for index in selected
            ])) if selected else 0.0,
            remove_rate=float(np.mean([
                dataset.records[index].action_kind == "remove" for index in selected
            ])) if selected else 0.0,
        ),
    )


def cross_validate_seed_holdout_linear_action_value(
    dataset: CounterfactualActionValueDataset,
    *,
    seeds: tuple[int, ...] | None = None,
    ridge: float = 1.0,
    nees_weight: float = 0.0,
    transmitted_message_weight: float = 0.0,
) -> ActionValueCrossValidationSummary:
    """Evaluate every seed once without mixing it into its training fold."""

    seed_values = tuple(sorted(
        set(dataset.seed_by_row) if seeds is None else set(seeds)
    ))
    if len(seed_values) < 2:
        raise ValueError("Cross-validation requires at least two seeds.")
    folds = [
        fit_seed_holdout_linear_action_value(
            dataset,
            training_seeds=tuple(
                seed for seed in seed_values if seed != holdout_seed
            ),
            holdout_seeds=(holdout_seed,),
            ridge=ridge,
            nees_weight=nees_weight,
            transmitted_message_weight=transmitted_message_weight,
        ).holdout
        for holdout_seed in seed_values
    ]
    group_count = sum(fold.group_count for fold in folds)

    def weighted(name: str) -> float:
        return float(sum(
            getattr(fold, name) * fold.group_count for fold in folds
        ) / group_count) if group_count else 0.0

    return ActionValueCrossValidationSummary(
        fold_count=len(folds),
        group_count=group_count,
        mean_action_match_rate=weighted("action_match_rate"),
        mean_action_kind_match_rate=weighted("action_kind_match_rate"),
        mean_position_rmse_reduction=weighted(
            "mean_position_rmse_reduction"
        ),
        mean_oracle_regret=weighted("mean_oracle_regret"),
        positive_rmse_gain_rate=weighted("positive_rmse_gain_rate"),
        nees_calibration_violation_rate=weighted(
            "nees_calibration_violation_rate"
        ),
        keep_rate=weighted("keep_rate"),
        add_rate=weighted("add_rate"),
        swap_rate=weighted("swap_rate"),
        remove_rate=weighted("remove_rate"),
    )


def cross_validate_linear_action_value_with_abstention(
    dataset: CounterfactualActionValueDataset,
    *,
    seeds: tuple[int, ...] | None = None,
    ridge: float = 1.0,
    maximum_training_nees_violation_rate: float = 0.25,
) -> ActionValueCrossValidationSummary:
    """Calibrate a keep margin on training seeds inside every holdout fold."""

    if not 0.0 <= maximum_training_nees_violation_rate <= 1.0:
        raise ValueError("Training NEES violation limit must be in [0, 1].")
    seed_values = tuple(sorted(
        set(dataset.seed_by_row) if seeds is None else set(seeds)
    ))
    if len(seed_values) < 2:
        raise ValueError("Cross-validation requires at least two seeds.")
    folds = []
    for holdout_seed in seed_values:
        training_seeds = tuple(
            seed for seed in seed_values if seed != holdout_seed
        )
        fit = fit_seed_holdout_linear_action_value(
            dataset, training_seeds=training_seeds,
            holdout_seeds=(holdout_seed,), ridge=ridge,
        )
        training_indices = np.asarray([
            index for index, seed in enumerate(dataset.seed_by_row)
            if seed in training_seeds
        ])
        threshold = _calibrate_keep_threshold(
            dataset, fit.model, training_indices,
            maximum_nees_violation_rate=maximum_training_nees_violation_rate,
        )
        holdout_indices = np.asarray([
            index for index, seed in enumerate(dataset.seed_by_row)
            if seed == holdout_seed
        ])
        folds.append(_evaluate_model_with_keep_threshold(
            dataset, fit.model, holdout_indices, threshold,
            training_seeds=training_seeds, holdout_seeds=(holdout_seed,),
        ))
    count = sum(fold.group_count for fold in folds)

    def weighted(name):
        return float(sum(getattr(fold, name) * fold.group_count
                         for fold in folds) / count) if count else 0.0

    return ActionValueCrossValidationSummary(
        fold_count=len(folds), group_count=count,
        mean_action_match_rate=weighted("action_match_rate"),
        mean_action_kind_match_rate=weighted("action_kind_match_rate"),
        mean_position_rmse_reduction=weighted("mean_position_rmse_reduction"),
        mean_oracle_regret=weighted("mean_oracle_regret"),
        positive_rmse_gain_rate=weighted("positive_rmse_gain_rate"),
        nees_calibration_violation_rate=weighted(
            "nees_calibration_violation_rate"
        ),
        keep_rate=weighted("keep_rate"), add_rate=weighted("add_rate"),
        swap_rate=weighted("swap_rate"), remove_rate=weighted("remove_rate"),
    )


def fit_frozen_linear_action_value_policy(
    dataset: CounterfactualActionValueDataset,
    *,
    training_seeds: tuple[int, ...] | None = None,
    ridge: float = 1.0,
    maximum_training_nees_violation_rate: float = 0.0,
) -> FrozenLinearActionValuePolicy:
    """Fit once and freeze both coefficients and the training-only keep margin."""

    seed_values = tuple(sorted(
        set(dataset.seed_by_row)
        if training_seeds is None else set(training_seeds)
    ))
    if not seed_values:
        raise ValueError("Frozen policy requires training seeds.")
    if set(seed_values) - set(dataset.seed_by_row):
        raise ValueError("Requested training seeds are absent from the dataset.")
    if not 0.0 <= maximum_training_nees_violation_rate <= 1.0:
        raise ValueError("Training NEES violation limit must be in [0, 1].")
    indices = np.asarray([
        index for index, seed in enumerate(dataset.seed_by_row)
        if seed in set(seed_values)
    ])
    model = _fit_linear_model(
        dataset.features[indices], dataset.position_rmse_reduction[indices],
        feature_names=dataset.feature_names, ridge=ridge,
    )
    threshold = _calibrate_keep_threshold(
        dataset, model, indices,
        maximum_nees_violation_rate=maximum_training_nees_violation_rate,
    )
    return FrozenLinearActionValuePolicy(
        model=model,
        keep_threshold=float(threshold),
        training_seeds=seed_values,
        maximum_training_nees_violation_rate=(
            float(maximum_training_nees_violation_rate)
        ),
    )


def evaluate_frozen_linear_action_value_policy(
    policy: FrozenLinearActionValuePolicy,
    dataset: CounterfactualActionValueDataset,
) -> FrozenPolicyValidation:
    """Evaluate a frozen policy without recalibrating on validation outcomes."""

    if dataset.feature_names != policy.model.feature_names:
        raise ValueError("Training and validation feature schemas differ.")
    validation_seeds = tuple(sorted(set(dataset.seed_by_row)))
    if set(validation_seeds) & set(policy.training_seeds):
        raise ValueError("Frozen-policy validation seeds overlap training seeds.")
    indices = np.arange(len(dataset.records))
    predictions = policy.model.predict(dataset.features)
    groups = _prediction_groups(dataset, indices, predictions)
    selected = _select_with_keep_threshold(
        dataset, groups, policy.keep_threshold
    )
    adjusted = np.full(len(indices), -np.inf)
    for index in selected:
        adjusted[int(index)] = 1.0
    summary = _evaluate_action_predictions(
        dataset, indices, adjusted,
        training_seeds=policy.training_seeds, holdout_seeds=validation_seeds,
    )
    gains = np.asarray([
        dataset.position_rmse_reduction[index] for index in selected
    ])
    gains_by_seed = {
        seed: [
            dataset.position_rmse_reduction[index]
            for index in selected if dataset.records[index].seed == seed
        ]
        for seed in validation_seeds
    }
    return FrozenPolicyValidation(
        summary=summary,
        worst_position_rmse_reduction=(
            float(np.min(gains)) if gains.size else 0.0
        ),
        tenth_percentile_position_rmse_reduction=(
            float(np.percentile(gains, 10.0)) if gains.size else 0.0
        ),
        positive_mean_gain_seed_rate=float(np.mean([
            np.mean(values) > 0.0 for values in gains_by_seed.values()
        ])) if gains_by_seed else 0.0,
    )


def _fit_linear_model(features, target, *, feature_names, ridge):
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    normalized = (features - mean) / scale
    design = np.column_stack((np.ones(len(normalized)), normalized))
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    parameters = np.linalg.pinv(
        design.T @ design + penalty
    ) @ design.T @ target
    return LinearActionValueModel(
        feature_names=tuple(feature_names),
        coefficients=tuple(float(value) for value in parameters[1:]),
        intercept=float(parameters[0]),
        feature_mean=tuple(float(value) for value in mean),
        feature_scale=tuple(float(value) for value in scale),
        ridge=float(ridge),
    )


def _calibrate_keep_threshold(
    dataset, model, indices, *, maximum_nees_violation_rate,
) -> float:
    predictions = model.predict(dataset.features[indices])
    groups = _prediction_groups(dataset, indices, predictions)
    gaps = []
    for candidates in groups.values():
        keep = next(
            item for item in candidates
            if dataset.records[item[0]].action_kind == "keep"
        )
        alternative = max(
            (item for item in candidates
             if dataset.records[item[0]].action_kind != "keep"),
            key=lambda item: item[1],
        )
        gaps.append(alternative[1] - keep[1])
    thresholds = (-np.inf, *sorted(set(gaps)), np.inf)
    feasible = []
    for threshold in thresholds:
        selected = _select_with_keep_threshold(dataset, groups, threshold)
        violation = float(np.mean([
            dataset.nees_calibration_improvement[index] < 0.0
            for index in selected
        ]))
        if violation <= maximum_nees_violation_rate:
            gain = float(np.mean([
                dataset.position_rmse_reduction[index] for index in selected
            ]))
            feasible.append((gain, float(threshold)))
    return max(feasible, key=lambda item: (item[0], item[1]))[1]


def _evaluate_model_with_keep_threshold(
    dataset, model, indices, threshold, *, training_seeds, holdout_seeds,
):
    predictions = model.predict(dataset.features[indices])
    groups = _prediction_groups(dataset, indices, predictions)
    selected = _select_with_keep_threshold(dataset, groups, threshold)
    adjusted = np.full(len(indices), -np.inf)
    index_position = {int(index): position for position, index in enumerate(indices)}
    for index in selected:
        adjusted[index_position[index]] = 1.0
    return _evaluate_action_predictions(
        dataset, indices, adjusted,
        training_seeds=training_seeds, holdout_seeds=holdout_seeds,
    )


def _prediction_groups(dataset, indices, predictions):
    groups = {}
    for index, prediction in zip(indices, predictions):
        groups.setdefault(dataset.group_by_row[index], []).append(
            (int(index), float(prediction))
        )
    return groups


def _select_with_keep_threshold(dataset, groups, threshold):
    selected = []
    for candidates in groups.values():
        keep = next(
            item for item in candidates
            if dataset.records[item[0]].action_kind == "keep"
        )
        alternative = max(
            (item for item in candidates
             if dataset.records[item[0]].action_kind != "keep"),
            key=lambda item: item[1],
        )
        selected.append(
            alternative[0]
            if alternative[1] - keep[1] > threshold else keep[0]
        )
    return selected


def fit_seed_holdout_tree_action_value(
    dataset: CounterfactualActionValueDataset,
    *,
    training_seeds: tuple[int, ...],
    holdout_seeds: tuple[int, ...],
    maximum_depth: int = 3,
    minimum_leaf_size: int = 8,
    nees_weight: float = 0.0,
    transmitted_message_weight: float = 0.0,
    replay_weight: float = 0.0,
    topology_change_weight: float = 0.0,
) -> TreeActionValueFitResult:
    """Fit a deterministic CART-style tree and evaluate complete action groups."""

    if maximum_depth < 1 or minimum_leaf_size < 1:
        raise ValueError("Tree depth and leaf size must be positive.")
    training = set(training_seeds)
    holdout = set(holdout_seeds)
    if not training or not holdout or training & holdout:
        raise ValueError("Training and holdout seeds must be nonempty and disjoint.")
    if (training | holdout) - set(dataset.seed_by_row):
        raise ValueError("Requested seeds are absent from the dataset.")
    train_indices = np.asarray([
        index for index, seed in enumerate(dataset.seed_by_row)
        if seed in training
    ])
    holdout_indices = np.asarray([
        index for index, seed in enumerate(dataset.seed_by_row)
        if seed in holdout
    ])
    target = _composite_action_value(
        dataset,
        nees_weight=nees_weight,
        transmitted_message_weight=transmitted_message_weight,
        replay_weight=replay_weight,
        topology_change_weight=topology_change_weight,
    )
    root = _fit_regression_tree(
        dataset.features[train_indices], target[train_indices],
        depth=0, maximum_depth=maximum_depth,
        minimum_leaf_size=minimum_leaf_size,
    )
    model = ShallowTreeActionValueModel(
        dataset.feature_names, root, maximum_depth, minimum_leaf_size
    )
    summary = _evaluate_action_predictions(
        dataset, holdout_indices, model.predict(dataset.features[holdout_indices]),
        training_seeds=training_seeds, holdout_seeds=holdout_seeds,
    )
    return TreeActionValueFitResult(model=model, holdout=summary)


def cross_validate_seed_holdout_tree_action_value(
    dataset: CounterfactualActionValueDataset,
    *, seeds: tuple[int, ...] | None = None, **fit_options,
) -> ActionValueCrossValidationSummary:
    seed_values = tuple(sorted(
        set(dataset.seed_by_row) if seeds is None else set(seeds)
    ))
    if len(seed_values) < 2:
        raise ValueError("Cross-validation requires at least two seeds.")
    folds = [
        fit_seed_holdout_tree_action_value(
            dataset,
            training_seeds=tuple(seed for seed in seed_values
                                 if seed != holdout_seed),
            holdout_seeds=(holdout_seed,), **fit_options,
        ).holdout
        for holdout_seed in seed_values
    ]
    count = sum(fold.group_count for fold in folds)

    def weighted(name):
        return float(sum(getattr(fold, name) * fold.group_count
                         for fold in folds) / count) if count else 0.0

    return ActionValueCrossValidationSummary(
        fold_count=len(folds), group_count=count,
        mean_action_match_rate=weighted("action_match_rate"),
        mean_action_kind_match_rate=weighted("action_kind_match_rate"),
        mean_position_rmse_reduction=weighted("mean_position_rmse_reduction"),
        mean_oracle_regret=weighted("mean_oracle_regret"),
        positive_rmse_gain_rate=weighted("positive_rmse_gain_rate"),
        nees_calibration_violation_rate=weighted(
            "nees_calibration_violation_rate"
        ),
        keep_rate=weighted("keep_rate"),
        add_rate=weighted("add_rate"),
        swap_rate=weighted("swap_rate"),
        remove_rate=weighted("remove_rate"),
    )


def _composite_action_value(
    dataset, *, nees_weight, transmitted_message_weight,
    replay_weight, topology_change_weight,
):
    return (
        dataset.position_rmse_reduction
        + nees_weight * dataset.nees_calibration_improvement
        - transmitted_message_weight * np.maximum(
            dataset.transmitted_message_cost, 0.0
        )
        - replay_weight * np.maximum(dataset.replay_cost, 0.0)
        - topology_change_weight * np.maximum(
            dataset.topology_change_cost, 0.0
        )
    )


def _fit_regression_tree(
    features, target, *, depth, maximum_depth, minimum_leaf_size,
) -> RegressionTreeNode:
    value = float(np.mean(target))
    if depth >= maximum_depth or len(target) < 2 * minimum_leaf_size:
        return RegressionTreeNode(value)
    best = None
    for feature_index in range(features.shape[1]):
        unique = np.unique(features[:, feature_index])
        if unique.size < 2:
            continue
        for threshold in 0.5 * (unique[:-1] + unique[1:]):
            left = features[:, feature_index] <= threshold
            left_count = int(np.sum(left))
            if (
                left_count < minimum_leaf_size
                or len(target) - left_count < minimum_leaf_size
            ):
                continue
            loss = float(
                np.sum((target[left] - np.mean(target[left])) ** 2)
                + np.sum((target[~left] - np.mean(target[~left])) ** 2)
            )
            candidate = (loss, feature_index, float(threshold), left)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    if best is None:
        return RegressionTreeNode(value)
    _, feature_index, threshold, left = best
    return RegressionTreeNode(
        value=value, feature_index=feature_index, threshold=threshold,
        left=_fit_regression_tree(
            features[left], target[left], depth=depth + 1,
            maximum_depth=maximum_depth, minimum_leaf_size=minimum_leaf_size,
        ),
        right=_fit_regression_tree(
            features[~left], target[~left], depth=depth + 1,
            maximum_depth=maximum_depth, minimum_leaf_size=minimum_leaf_size,
        ),
    )


def _evaluate_action_predictions(
    dataset, indices, predictions, *, training_seeds, holdout_seeds,
) -> ActionValueHoldoutSummary:
    grouped = {}
    for index, prediction in zip(indices, predictions):
        grouped.setdefault(dataset.group_by_row[index], []).append(
            (int(index), float(prediction))
        )
    selected, oracle, matches, kind_matches = [], [], [], []
    for candidates in grouped.values():
        selected_index = max(candidates, key=lambda value: value[1])[0]
        oracle_value = max(
            dataset.position_rmse_reduction[index] for index, _ in candidates
        )
        selected.append(selected_index)
        oracle.append(float(oracle_value))
        matches.append(np.isclose(
            dataset.position_rmse_reduction[selected_index], oracle_value
        ))
        best_kinds = {
            dataset.records[index].action_kind for index, _ in candidates
            if np.isclose(dataset.position_rmse_reduction[index], oracle_value)
        }
        kind_matches.append(
            dataset.records[selected_index].action_kind in best_kinds
        )
    gains = np.asarray([
        dataset.position_rmse_reduction[index] for index in selected
    ])
    return ActionValueHoldoutSummary(
        training_seeds=tuple(training_seeds),
        holdout_seeds=tuple(holdout_seeds), group_count=len(grouped),
        action_match_rate=float(np.mean(matches)) if matches else 0.0,
        action_kind_match_rate=(
            float(np.mean(kind_matches)) if kind_matches else 0.0
        ),
        mean_position_rmse_reduction=float(np.mean(gains)) if gains.size else 0.0,
        mean_oracle_regret=float(np.mean(np.asarray(oracle) - gains))
        if gains.size else 0.0,
        positive_rmse_gain_rate=float(np.mean(gains > 0.0))
        if gains.size else 0.0,
        nees_calibration_violation_rate=float(np.mean([
            dataset.nees_calibration_improvement[index] < 0.0
            for index in selected
        ])) if selected else 0.0,
        keep_rate=float(np.mean([
            dataset.records[index].action_kind == "keep" for index in selected
        ])) if selected else 0.0,
        add_rate=float(np.mean([
            dataset.records[index].action_kind == "add" for index in selected
        ])) if selected else 0.0,
        swap_rate=float(np.mean([
            dataset.records[index].action_kind == "swap" for index in selected
        ])) if selected else 0.0,
        remove_rate=float(np.mean([
            dataset.records[index].action_kind == "remove" for index in selected
        ])) if selected else 0.0,
    )
