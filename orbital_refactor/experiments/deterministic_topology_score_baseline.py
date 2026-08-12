from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.short_horizon_counterfactual_study import (
    ShortHorizonActionRecord,
    ShortHorizonCounterfactualStudy,
)


@dataclass(frozen=True)
class DeterministicScoreSelectionRecord:
    seed: int
    decision_epoch: int
    horizon_epochs: int
    action_kind: str
    added_edges: tuple[tuple[str, str], ...]
    removed_edges: tuple[tuple[str, str], ...]
    score_margin: float
    calibrated_threshold: float
    position_rmse_reduction: float
    nees_calibration_improvement: float
    communication_risk_score_gain: float
    resynchronization_cost: int


@dataclass(frozen=True)
class DeterministicScoreCrossValidation:
    feature: str
    fold_count: int
    group_count: int
    mean_position_rmse_reduction: float
    positive_gain_rate: float
    nees_calibration_violation_rate: float
    mean_nees_calibration_improvement: float
    worst_nees_calibration_improvement: float
    position_rmse_reduction_confidence_interval: tuple[float, float]
    nees_calibration_violation_confidence_interval: tuple[float, float]
    keep_rate: float
    add_rate: float
    swap_rate: float
    remove_rate: float
    selections: tuple[DeterministicScoreSelectionRecord, ...] = ()


def cross_validate_deterministic_score_with_abstention(
    study: ShortHorizonCounterfactualStudy, *, feature: str,
    maximum_training_nees_violation_rate: float = 0.25,
    minimum_training_mean_gain: float = 0.0,
    communication_risk_weight: float = 0.0,
    resynchronization_weight: float = 0.0,
    topology_change_weight: float = 0.0,
    maximum_added_communication_risk: float | None = None,
    allowed_action_kinds: tuple[str, ...] = ("add", "swap"),
    nees_calibration_degradation_tolerance: float = 0.0,
) -> DeterministicScoreCrossValidation:
    """Calibrate a scalar action score and keep margin on disjoint seeds."""

    if not study.records or not hasattr(study.records[0], feature):
        raise ValueError("feature must name a ShortHorizonActionRecord field.")
    if not 0.0 <= maximum_training_nees_violation_rate <= 1.0:
        raise ValueError("NEES violation limit must be in [0, 1].")
    if nees_calibration_degradation_tolerance < 0.0:
        raise ValueError("NEES degradation tolerance cannot be negative.")
    if min(
        communication_risk_weight,
        resynchronization_weight,
        topology_change_weight,
    ) < 0.0:
        raise ValueError("Risk and cost weights cannot be negative.")
    if (
        maximum_added_communication_risk is not None
        and maximum_added_communication_risk < 0.0
    ):
        raise ValueError("Maximum added communication risk cannot be negative.")
    allowed = tuple(dict.fromkeys(str(kind) for kind in allowed_action_kinds))
    if not allowed or set(allowed) - {"add", "swap", "remove"}:
        raise ValueError("Allowed action kinds must use add, swap, or remove.")
    score = lambda record: (
        float(getattr(record, feature))
        - communication_risk_weight
        * max(float(record.communication_risk_score_gain), 0.0)
        - resynchronization_weight * float(record.resynchronization_cost)
        - topology_change_weight * float(record.topology_change_cost)
    )
    eligible = lambda record: (
        record.action_kind in allowed
        and (
            maximum_added_communication_risk is None
            or record.communication_risk_score_gain
            <= maximum_added_communication_risk
        )
    )
    seeds = tuple(sorted(set(record.seed for record in study.records)))
    if len(seeds) < 2:
        raise ValueError("Cross-validation requires at least two seeds.")
    groups = {}
    for record in study.records:
        key = (record.seed, record.decision_epoch, record.horizon_epochs)
        groups.setdefault(key, []).append(record)
    groups = {key: tuple(records) for key, records in groups.items()}
    selected = []
    selection_records = []
    for holdout_seed in seeds:
        threshold = _calibrate_threshold(
            tuple(
                records for key, records in groups.items()
                if key[0] != holdout_seed
            ),
            score=score,
            eligible=eligible,
            maximum_nees_violation_rate=(
                maximum_training_nees_violation_rate
            ),
            minimum_mean_gain=minimum_training_mean_gain,
            nees_degradation_tolerance=(
                nees_calibration_degradation_tolerance
            ),
        )
        for key, records in groups.items():
            if key[0] != holdout_seed:
                continue
            keep, alternative = _keep_and_best_alternative(
                records, score, eligible
            )
            margin = score(alternative) - score(keep)
            chosen = alternative if margin > threshold else keep
            selected.append(chosen)
            selection_records.append(DeterministicScoreSelectionRecord(
                seed=chosen.seed,
                decision_epoch=chosen.decision_epoch,
                horizon_epochs=chosen.horizon_epochs,
                action_kind=chosen.action_kind,
                added_edges=chosen.added_edges,
                removed_edges=chosen.removed_edges,
                score_margin=float(margin),
                calibrated_threshold=float(threshold),
                position_rmse_reduction=float(
                    chosen.position_rmse_reduction
                ),
                nees_calibration_improvement=float(
                    chosen.nees_calibration_improvement
                ),
                communication_risk_score_gain=float(
                    chosen.communication_risk_score_gain
                ),
                resynchronization_cost=int(chosen.resynchronization_cost),
            ))
    gains = np.asarray([
        record.position_rmse_reduction for record in selected
    ])
    nees_improvements = np.asarray([
        record.nees_calibration_improvement for record in selected
    ])
    violations = np.asarray([
        record.nees_calibration_improvement
        < -nees_calibration_degradation_tolerance
        for record in selected
    ])
    return DeterministicScoreCrossValidation(
        feature=feature, fold_count=len(seeds), group_count=len(groups),
        mean_position_rmse_reduction=float(np.mean(gains)),
        positive_gain_rate=float(np.mean(gains > 0.0)),
        nees_calibration_violation_rate=float(np.mean(violations)),
        mean_nees_calibration_improvement=float(np.mean(nees_improvements)),
        worst_nees_calibration_improvement=float(np.min(nees_improvements)),
        position_rmse_reduction_confidence_interval=(
            _bootstrap_mean_interval(gains)
        ),
        nees_calibration_violation_confidence_interval=(
            _wilson_interval(int(np.sum(violations)), len(violations))
        ),
        keep_rate=_kind_rate(selected, "keep"),
        add_rate=_kind_rate(selected, "add"),
        swap_rate=_kind_rate(selected, "swap"),
        remove_rate=_kind_rate(selected, "remove"),
        selections=tuple(sorted(
            selection_records,
            key=lambda item: (
                item.seed, item.decision_epoch, item.horizon_epochs
            ),
        )),
    )


def _calibrate_threshold(
    training_groups, *, score, eligible, maximum_nees_violation_rate,
    minimum_mean_gain, nees_degradation_tolerance,
):
    choices = tuple(
        _keep_and_best_alternative(records, score, eligible)
        for records in training_groups
    )
    gaps = tuple(
        score(alternative) - score(keep)
        for keep, alternative in choices
    )
    feasible = []
    for threshold in (-np.inf, *sorted(set(gaps)), np.inf):
        selected = tuple(
            alternative if gap > threshold else keep
            for (keep, alternative), gap in zip(choices, gaps)
        )
        violation = float(np.mean([
            record.nees_calibration_improvement
            < -nees_degradation_tolerance
            for record in selected
        ]))
        gain = float(np.mean([
            record.position_rmse_reduction for record in selected
        ]))
        if (
            violation <= maximum_nees_violation_rate
            and gain >= minimum_mean_gain
        ):
            feasible.append((gain, float(threshold)))
    if not feasible:
        return np.inf
    return max(feasible, key=lambda item: (item[0], item[1]))[1]


def _select(records, score, eligible, threshold):
    keep, alternative = _keep_and_best_alternative(records, score, eligible)
    return (
        alternative
        if score(alternative) - score(keep) > threshold
        else keep
    )


def _keep_and_best_alternative(records, score, eligible):
    keep = next(record for record in records if record.action_kind == "keep")
    alternatives = tuple(
        record for record in records
        if record.action_kind != "keep" and eligible(record)
    )
    alternative = max(alternatives, key=score) if alternatives else keep
    return keep, alternative


def _kind_rate(records: list[ShortHorizonActionRecord], kind: str) -> float:
    return float(np.mean([record.action_kind == kind for record in records]))


def _bootstrap_mean_interval(
    values: np.ndarray, *, confidence: float = 0.95,
    resamples: int = 4000,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(20270812)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    means = np.mean(values[indices], axis=1)
    tail = 0.5 * (1.0 - confidence)
    return tuple(float(value) for value in np.quantile(
        means, (tail, 1.0 - tail)
    ))


def _wilson_interval(
    successes: int, total: int, *, z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson interval counts are invalid.")
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    radius = z * np.sqrt(
        proportion * (1.0 - proportion) / total
        + z**2 / (4.0 * total**2)
    ) / denominator
    return float(center - radius), float(center + radius)
