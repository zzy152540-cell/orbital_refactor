from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace

import numpy as np

from experiments.short_horizon_counterfactual_study import (
    ShortHorizonActionRecord,
    ShortHorizonCounterfactualStudy,
)


@dataclass(frozen=True)
class ActionCostWeights:
    transmitted_message: float = 0.0
    replay: float = 0.0
    topology_change: float = 0.0

    def __post_init__(self) -> None:
        if min(
            self.transmitted_message, self.replay, self.topology_change
        ) < 0.0:
            raise ValueError("Action cost weights cannot be negative.")


@dataclass(frozen=True)
class OraclePolicySummary:
    policy: str
    group_count: int
    non_keep_action_rate: float
    positive_gain_rate: float
    mean_position_rmse_reduction: float
    median_position_rmse_reduction: float
    worst_position_rmse_reduction: float
    mean_worst_node_rmse_reduction: float
    nees_calibration_violation_rate: float
    coverage_calibration_violation_rate: float
    mean_transmitted_message_cost: float
    mean_replay_cost: float
    mean_topology_change_cost: float
    action_kind_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class GraphActionLearnabilityReport:
    group_count: int
    action_count: int
    mean_action_count_per_group: float
    safe_positive_action_availability_rate: float
    summaries: tuple[OraclePolicySummary, ...]

    def summary(self, policy: str) -> OraclePolicySummary:
        return next(value for value in self.summaries if value.policy == policy)


@dataclass(frozen=True)
class GraphActionLearnabilityStratum:
    dimension: str
    value: int
    report: GraphActionLearnabilityReport


def analyze_graph_action_learnability(
    study: ShortHorizonCounterfactualStudy,
    *,
    cost_weights: ActionCostWeights = ActionCostWeights(),
    nees_tolerance: float = 0.0,
    coverage_tolerance: float = 0.0,
) -> GraphActionLearnabilityReport:
    """Measure actionable oracle headroom relative to the safe keep policy."""

    if nees_tolerance < 0.0 or coverage_tolerance < 0.0:
        raise ValueError("Consistency tolerances cannot be negative.")
    grouped = defaultdict(list)
    for record in study.records:
        grouped[_group_id(record)].append(record)
    if not grouped:
        raise ValueError("Learnability analysis requires action records.")

    selections = {
        "keep": [],
        "unconstrained_oracle": [],
        "consistency_safe_oracle": [],
        "cost_aware_safe_oracle": [],
    }
    safe_positive_available = []
    for records in grouped.values():
        keep = _unique_keep(records)
        safe = [
            record for record in records
            if _is_consistency_safe(
                record, nees_tolerance, coverage_tolerance
            )
        ]
        # Keep has zero relative consistency change and is always safe.
        if keep not in safe:
            safe.append(keep)
        selections["keep"].append(keep)
        selections["unconstrained_oracle"].append(
            _best(records, lambda record: record.position_rmse_reduction)
        )
        selections["consistency_safe_oracle"].append(
            _best(safe, lambda record: record.position_rmse_reduction)
        )
        selections["cost_aware_safe_oracle"].append(
            _best(safe, lambda record: _cost_aware_value(record, cost_weights))
        )
        safe_positive_available.append(any(
            record.action_kind != "keep"
            and record.position_rmse_reduction > 0.0
            for record in safe
        ))

    return GraphActionLearnabilityReport(
        group_count=len(grouped),
        action_count=len(study.records),
        mean_action_count_per_group=float(len(study.records) / len(grouped)),
        safe_positive_action_availability_rate=float(
            np.mean(safe_positive_available)
        ),
        summaries=tuple(
            _summarize(policy, selected)
            for policy, selected in selections.items()
        ),
    )


def stratify_graph_action_learnability(
    study: ShortHorizonCounterfactualStudy,
    *,
    dimension: str,
    cost_weights: ActionCostWeights = ActionCostWeights(),
    nees_tolerance: float = 0.0,
    coverage_tolerance: float = 0.0,
) -> tuple[GraphActionLearnabilityStratum, ...]:
    """Repeat the oracle analysis within a causal scenario dimension."""

    if dimension not in {"decision_epoch", "horizon_epochs", "node_count"}:
        raise ValueError("Unsupported learnability stratification dimension.")
    values = sorted({getattr(record, dimension) for record in study.records})
    return tuple(
        GraphActionLearnabilityStratum(
            dimension=dimension,
            value=int(value),
            report=analyze_graph_action_learnability(
                replace(
                    study,
                    records=tuple(
                        record for record in study.records
                        if getattr(record, dimension) == value
                    ),
                ),
                cost_weights=cost_weights,
                nees_tolerance=nees_tolerance,
                coverage_tolerance=coverage_tolerance,
            ),
        )
        for value in values
    )


def consistency_safe_action_kind_oracles(
    study: ShortHorizonCounterfactualStudy,
    *,
    nees_tolerance: float = 0.0,
    coverage_tolerance: float = 0.0,
) -> tuple[OraclePolicySummary, ...]:
    """Measure each action kind's headroom while allowing keep abstention."""

    if nees_tolerance < 0.0 or coverage_tolerance < 0.0:
        raise ValueError("Consistency tolerances cannot be negative.")
    grouped = defaultdict(list)
    for record in study.records:
        grouped[_group_id(record)].append(record)
    if not grouped:
        raise ValueError("Action-kind analysis requires action records.")

    action_kinds = sorted({
        record.action_kind for record in study.records
        if record.action_kind != "keep"
    })
    summaries = []
    for action_kind in action_kinds:
        selected = []
        for records in grouped.values():
            keep = _unique_keep(records)
            candidates = [
                record for record in records
                if record.action_kind == action_kind
                and _is_consistency_safe(
                    record, nees_tolerance, coverage_tolerance
                )
            ]
            selected.append(_best(
                [keep, *candidates],
                lambda record: record.position_rmse_reduction,
            ))
        summaries.append(_summarize(
            f"consistency_safe_{action_kind}_oracle", selected
        ))
    return tuple(summaries)


def _group_id(record: ShortHorizonActionRecord):
    return (
        record.node_count, record.seed,
        record.decision_epoch, record.horizon_epochs,
    )


def _unique_keep(records):
    keep = [record for record in records if record.action_kind == "keep"]
    if len(keep) != 1:
        raise ValueError("Every action group must contain exactly one keep action.")
    return keep[0]


def _is_consistency_safe(record, nees_tolerance, coverage_tolerance):
    return (
        record.nees_calibration_improvement >= -nees_tolerance
        and record.nees_coverage_calibration_improvement >= -coverage_tolerance
    )


def _cost_aware_value(record, weights):
    return (
        record.position_rmse_reduction
        - weights.transmitted_message * record.transmitted_message_cost
        - weights.replay * record.replay_cost
        - weights.topology_change * record.topology_change_cost
    )


def _best(records, value):
    # Prefer keep on an exact tie so an oracle does not claim needless churn.
    return max(records, key=lambda record: (
        value(record), record.action_kind == "keep"
    ))


def _summarize(policy, selected):
    gains = np.asarray([
        record.position_rmse_reduction for record in selected
    ], dtype=float)
    kinds = Counter(record.action_kind for record in selected)
    return OraclePolicySummary(
        policy=policy,
        group_count=len(selected),
        non_keep_action_rate=float(np.mean([
            record.action_kind != "keep" for record in selected
        ])),
        positive_gain_rate=float(np.mean(gains > 0.0)),
        mean_position_rmse_reduction=float(np.mean(gains)),
        median_position_rmse_reduction=float(np.median(gains)),
        worst_position_rmse_reduction=float(np.min(gains)),
        mean_worst_node_rmse_reduction=float(np.mean([
            record.worst_node_position_rmse_reduction for record in selected
        ])),
        nees_calibration_violation_rate=float(np.mean([
            record.nees_calibration_improvement < 0.0 for record in selected
        ])),
        coverage_calibration_violation_rate=float(np.mean([
            record.nees_coverage_calibration_improvement < 0.0
            for record in selected
        ])),
        mean_transmitted_message_cost=float(np.mean([
            record.transmitted_message_cost for record in selected
        ])),
        mean_replay_cost=float(np.mean([
            record.replay_cost for record in selected
        ])),
        mean_topology_change_cost=float(np.mean([
            record.topology_change_cost for record in selected
        ])),
        action_kind_counts=tuple(sorted(kinds.items())),
    )
