from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from experiments.topology_control_baselines import AlwaysKeepPolicy
from experiments.topology_ppo_stage1 import (
    Stage1Configuration,
    build_stage1_environment,
)
from experiments.topology_snapshot_counterfactual import (
    evaluate_topology_action_snapshot,
)


@dataclass(frozen=True)
class TopologyDecisionOpportunity:
    environment_seed: int
    decision_epoch: int
    legal_action_count: int
    best_raw_action_id: int
    best_raw_action_kind: str
    best_raw_action_signature: str
    best_action_id: int
    best_action_kind: str
    best_action_signature: str
    best_gain_over_keep: float
    best_penalized_gain_over_keep: float
    best_to_second_margin: float
    best_to_second_penalized_margin: float


@dataclass(frozen=True)
class TopologyTaskLearnabilityAudit:
    opportunities: tuple[TopologyDecisionOpportunity, ...]
    best_raw_kind_counts: tuple[tuple[str, int], ...]
    best_kind_counts: tuple[tuple[str, int], ...]
    best_raw_action_signature_counts: tuple[tuple[str, int], ...]
    best_action_signature_counts: tuple[tuple[str, int], ...]
    nontrivial_raw_fraction: float
    nontrivial_penalized_fraction: float
    median_best_gain_over_keep: float
    median_positive_gain_over_keep: float
    median_best_to_second_margin: float


@dataclass(frozen=True)
class TopologyHorizonStabilityAudit:
    decision_count: int
    action_agreement_by_horizon_pair: tuple[tuple[str, float], ...]
    kind_transition_counts_one_to_longest: tuple[tuple[str, int], ...]


def audit_stage1_task_learnability(
    configuration: Stage1Configuration,
    *, seeds,
    minimum_meaningful_gain: float = 1.0e-3,
) -> TopologyTaskLearnabilityAudit:
    """Enumerate legal actions along a common keep reference trajectory."""

    requested_seeds = tuple(int(seed) for seed in seeds)
    if not requested_seeds or len(set(requested_seeds)) != len(requested_seeds):
        raise ValueError("Learnability-audit seeds must be unique and nonempty.")
    if minimum_meaningful_gain < 0.0:
        raise ValueError("Minimum meaningful gain cannot be negative.")
    decision_count = int(np.ceil(
        configuration.episode_epochs / configuration.decision_interval_epochs
    ))
    opportunities = []
    for seed in requested_seeds:
        for decision_epoch in range(decision_count):
            records = evaluate_topology_action_snapshot(
                build_stage1_environment(configuration), seed=seed,
                decision_epoch=decision_epoch,
                baseline_policy=AlwaysKeepPolicy(), lookahead_steps=1,
            )
            keep = next(record for record in records if record.action_kind == "keep")
            scored = tuple(
                (record, _penalized_gain(record, keep, configuration))
                for record in records
            )
            raw_order = sorted(
                records,
                key=lambda record: (
                    record.position_rmse_reduction_vs_keep, -record.action_id
                ),
                reverse=True,
            )
            penalized_order = sorted(
                scored, key=lambda item: (item[1], -item[0].action_id),
                reverse=True,
            )
            best_raw = raw_order[0]
            best_penalized, best_penalized_gain = penalized_order[0]
            opportunities.append(TopologyDecisionOpportunity(
                environment_seed=seed, decision_epoch=decision_epoch,
                legal_action_count=len(records),
                best_raw_action_id=best_raw.action_id,
                best_raw_action_kind=best_raw.action_kind,
                best_raw_action_signature=_action_signature(best_raw),
                best_action_id=best_penalized.action_id,
                best_action_kind=best_penalized.action_kind,
                best_action_signature=_action_signature(best_penalized),
                best_gain_over_keep=float(
                    best_raw.position_rmse_reduction_vs_keep
                ),
                best_penalized_gain_over_keep=float(best_penalized_gain),
                best_to_second_margin=_margin(raw_order, lambda item: (
                    item.position_rmse_reduction_vs_keep
                )),
                best_to_second_penalized_margin=_margin(
                    penalized_order, lambda item: item[1]
                ),
            ))
    raw_kind_counts = Counter(item.best_raw_action_kind for item in opportunities)
    kind_counts = Counter(item.best_action_kind for item in opportunities)
    raw_signature_counts = Counter(
        item.best_raw_action_signature for item in opportunities
    )
    signature_counts = Counter(item.best_action_signature for item in opportunities)
    raw_gains = np.asarray([
        item.best_gain_over_keep for item in opportunities
    ])
    penalized_gains = np.asarray([
        item.best_penalized_gain_over_keep for item in opportunities
    ])
    positive = raw_gains[raw_gains > minimum_meaningful_gain]
    return TopologyTaskLearnabilityAudit(
        opportunities=tuple(opportunities),
        best_raw_kind_counts=tuple(sorted(raw_kind_counts.items())),
        best_kind_counts=tuple(sorted(kind_counts.items())),
        best_raw_action_signature_counts=tuple(sorted(
            raw_signature_counts.items()
        )),
        best_action_signature_counts=tuple(sorted(signature_counts.items())),
        nontrivial_raw_fraction=float(np.mean(
            raw_gains > minimum_meaningful_gain
        )),
        nontrivial_penalized_fraction=float(np.mean(
            penalized_gains > minimum_meaningful_gain
        )),
        median_best_gain_over_keep=float(np.median(raw_gains)),
        median_positive_gain_over_keep=(
            float(np.median(positive)) if len(positive) else 0.0
        ),
        median_best_to_second_margin=float(np.median([
            item.best_to_second_margin for item in opportunities
        ])),
    )


def audit_stage1_horizon_stability(
    configuration: Stage1Configuration,
    *, seeds, lookahead_steps=(1, 2, 3),
) -> TopologyHorizonStabilityAudit:
    """Measure whether counterfactual best actions survive horizon changes."""

    requested_seeds = tuple(int(seed) for seed in seeds)
    horizons = tuple(int(value) for value in lookahead_steps)
    if not requested_seeds or len(set(requested_seeds)) != len(requested_seeds):
        raise ValueError("Horizon-audit seeds must be unique and nonempty.")
    if len(horizons) < 2 or len(set(horizons)) != len(horizons) or min(horizons) < 1:
        raise ValueError("Horizon audit requires unique positive lookaheads.")
    decision_count = int(np.ceil(
        configuration.episode_epochs / configuration.decision_interval_epochs
    ))
    best_by_horizon = {horizon: [] for horizon in horizons}
    for seed in requested_seeds:
        for decision_epoch in range(decision_count):
            for horizon in horizons:
                records = evaluate_topology_action_snapshot(
                    build_stage1_environment(configuration), seed=seed,
                    decision_epoch=decision_epoch,
                    baseline_policy=AlwaysKeepPolicy(),
                    lookahead_steps=horizon,
                )
                best = max(records, key=lambda record: (
                    record.position_rmse_reduction_vs_keep, -record.action_id
                ))
                best_by_horizon[horizon].append(
                    (best.action_kind, best.action_id)
                )
    reference = horizons[0]
    agreements = []
    for horizon in horizons[1:]:
        agreements.append((
            f"{reference}_vs_{horizon}",
            float(np.mean([
                left == right for left, right in zip(
                    best_by_horizon[reference], best_by_horizon[horizon]
                )
            ])),
        ))
    transitions = Counter(
        f"{left[0]}->{right[0]}" for left, right in zip(
            best_by_horizon[reference], best_by_horizon[horizons[-1]]
        )
    )
    return TopologyHorizonStabilityAudit(
        decision_count=len(best_by_horizon[reference]),
        action_agreement_by_horizon_pair=tuple(agreements),
        kind_transition_counts_one_to_longest=tuple(sorted(transitions.items())),
    )


def _penalized_gain(record, keep, configuration):
    scales = configuration.cost_normalization
    weights = configuration.penalty_weights
    return float(
        record.position_rmse_reduction_vs_keep
        - weights.communication
        * (record.transmitted_messages - keep.transmitted_messages)
        / scales.transmitted_messages
        - weights.topology_switch
        * (record.topology_switch_count - keep.topology_switch_count)
        / scales.topology_switch
        - weights.resynchronization
        * (record.resynchronization_count - keep.resynchronization_count)
        / scales.resynchronization
    )


def _margin(ordered, value):
    if len(ordered) < 2:
        return 0.0
    return float(value(ordered[0]) - value(ordered[1]))


def _action_signature(record):
    return f"{record.action_kind}|+{record.added_edges}|-{record.removed_edges}"
