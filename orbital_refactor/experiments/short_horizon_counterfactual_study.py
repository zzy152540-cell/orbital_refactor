from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from cooperative.topology_policy import GraphObservation, UndirectedEdge
from experiments.decision_time_edge_scoring import range_topology_score
from experiments.short_horizon_topology_counterfactual import (
    run_short_horizon_topology_counterfactual,
)


@dataclass(frozen=True)
class ShortHorizonActionRecord:
    node_count: int
    seed: int
    decision_epoch: int
    horizon_epochs: int
    action_kind: str
    active_edges: tuple[UndirectedEdge, ...]
    added_edges: tuple[UndirectedEdge, ...]
    removed_edges: tuple[UndirectedEdge, ...]
    covariance_trace_reduction: float
    covariance_logdet_reduction: float
    position_rmse_reduction: float
    worst_node_position_rmse_reduction: float
    nees_calibration_improvement: float
    nees_coverage_calibration_improvement: float
    transmitted_message_cost: int
    replay_cost: int
    topology_change_cost: int
    endpoint_uncertainty_score_gain: float
    projected_uncertainty_score_gain: float
    approximate_trace_score_gain: float
    approximate_logdet_score_gain: float
    negative_distance_score_gain: float
    observation_age_score_gain: float
    recent_nis_score_gain: float
    negative_recent_nis_score_gain: float
    nis_calibration_score_gain: float
    nis_sample_count_score_gain: float
    negative_anomaly_score_gain: float
    communication_risk_score_gain: float = 0.0
    resynchronization_cost: int = 0


@dataclass(frozen=True)
class ActionKindSummary:
    action_kind: str
    sample_count: int
    positive_rmse_gain_rate: float
    mean_position_rmse_reduction: float
    median_position_rmse_reduction: float
    mean_covariance_trace_reduction: float
    mean_covariance_logdet_reduction: float
    mean_transmitted_message_cost: float
    nees_calibration_violation_rate: float
    coverage_calibration_violation_rate: float


@dataclass(frozen=True)
class SwapOracleSummary:
    group_count: int
    positive_best_swap_rate: float
    mean_best_position_rmse_reduction: float
    median_best_position_rmse_reduction: float


@dataclass(frozen=True)
class SwapPredictorSelectionSummary:
    predictor: str
    group_count: int
    best_swap_hit_rate: float
    mean_position_rmse_regret: float
    median_position_rmse_regret: float
    mean_selected_count: float
    mean_selected_position_rmse_reduction: float
    selected_positive_rmse_gain_rate: float
    positive_gain_precision: float | None
    selected_nees_calibration_violation_rate: float


@dataclass(frozen=True)
class SwapNisRetentionGateSummary:
    maximum_removed_edge_log_deviation: float
    group_count: int
    swap_execution_rate: float
    mean_position_rmse_reduction: float
    positive_rmse_gain_rate: float
    nees_calibration_violation_rate: float


@dataclass(frozen=True)
class SwapAbstentionSummary:
    predictor: str
    group_count: int
    swap_execution_rate: float
    mean_position_rmse_reduction: float
    positive_rmse_gain_rate: float
    nees_calibration_violation_rate: float


@dataclass(frozen=True)
class ShortHorizonCounterfactualStudy:
    node_counts: tuple[int, ...]
    seeds: tuple[int, ...]
    decision_epochs: tuple[int, ...]
    horizon_epochs: tuple[int, ...]
    relative_modalities: tuple[str, ...]
    decision_observations: tuple[
        tuple[tuple[int, int, int, int], GraphObservation], ...
    ]
    records: tuple[ShortHorizonActionRecord, ...]
    summaries_by_action_kind: tuple[ActionKindSummary, ...]
    swap_oracle_summary: SwapOracleSummary
    swap_predictor_summaries: tuple[SwapPredictorSelectionSummary, ...]
    swap_nis_retention_gate_summaries: tuple[
        SwapNisRetentionGateSummary, ...
    ]
    swap_abstention_summaries: tuple[SwapAbstentionSummary, ...]


def run_short_horizon_counterfactual_study(
    *, node_counts: tuple[int, ...] = (3,),
    seeds: Iterable[int] = range(5),
    decision_epochs: tuple[int, ...] = (1, 3, 5),
    horizon_epochs: tuple[int, ...] = (1, 3, 5),
    dt: float = 2.0,
    relative_modalities: tuple[str, ...] = ("RANGE",),
    backend: str = "offline_replay",
    packet_loss: float = 0.0,
    communication_delay: float = 0.0,
    packet_loss_by_edge: dict[UndirectedEdge, float] | None = None,
    communication_delay_by_edge: dict[UndirectedEdge, float] | None = None,
    future_batch_relative_observations: bool = False,
) -> ShortHorizonCounterfactualStudy:
    """Run a controlled action matrix and express every result versus keep."""

    seed_values = _unique_nonempty("seeds", tuple(int(value) for value in seeds))
    node_values = _unique_nonempty(
        "node_counts", tuple(int(value) for value in node_counts)
    )
    decision_values = _unique_nonempty(
        "decision_epochs", tuple(int(value) for value in decision_epochs)
    )
    horizon_values = _unique_nonempty(
        "horizon_epochs", tuple(int(value) for value in horizon_epochs)
    )
    if any(value not in {3, 5} for value in node_values):
        raise ValueError("node_counts values must be 3 or 5.")
    if any(value < 0 for value in decision_values):
        raise ValueError("decision_epochs values cannot be negative.")
    if any(value <= 0 for value in horizon_values):
        raise ValueError("horizon_epochs values must be positive.")

    records = []
    decision_observations = []
    for node_count in node_values:
        for seed in seed_values:
            for decision_epoch in decision_values:
                for horizon in horizon_values:
                    result = run_short_horizon_topology_counterfactual(
                        node_count=node_count,
                        seed=seed,
                        decision_epoch=decision_epoch,
                        horizon_epochs=horizon,
                        dt=dt,
                        relative_modalities=relative_modalities,
                        backend=backend,
                        packet_loss=packet_loss,
                        communication_delay=communication_delay,
                        packet_loss_by_edge=packet_loss_by_edge,
                        communication_delay_by_edge=(
                            communication_delay_by_edge
                        ),
                        future_batch_relative_observations=(
                            future_batch_relative_observations
                        ),
                    )
                    decision_observations.append((
                        (node_count, seed, decision_epoch, horizon),
                        result.decision_observation,
                    ))
                    keep = next(
                        rollout for rollout in result.rollouts
                        if rollout.action.kind == "keep"
                    )
                    keep_metrics = keep.metrics
                    keep_score = range_topology_score(
                        result.decision_observation,
                        keep.action.topology,
                    )
                    edge_by_nodes = {
                        edge.nodes: edge
                        for edge in result.decision_observation.candidate_edges
                    }

                    def communication_risk(active_edges):
                        return float(sum(
                            edge_by_nodes[edge].packet_loss_rate
                            + edge_by_nodes[edge].delay / dt
                            + (0.0 if edge_by_nodes[edge].communication_available
                               else 1.0)
                            for edge in active_edges
                        ))
                    keep_communication_risk = communication_risk(
                        keep.action.topology.active_edges
                    )
                    for rollout in result.rollouts:
                        metrics = rollout.metrics
                        action = rollout.action
                        action_score = range_topology_score(
                            result.decision_observation,
                            action.topology,
                        )
                        records.append(ShortHorizonActionRecord(
                            node_count=node_count,
                            seed=seed,
                            decision_epoch=decision_epoch,
                            horizon_epochs=horizon,
                            action_kind=action.kind,
                            active_edges=action.topology.active_edges,
                            added_edges=action.added_edges,
                            removed_edges=action.removed_edges,
                            covariance_trace_reduction=(
                                keep_metrics.mean_covariance_trace
                                - metrics.mean_covariance_trace
                            ),
                            covariance_logdet_reduction=(
                                keep_metrics.mean_covariance_logdet
                                - metrics.mean_covariance_logdet
                            ),
                            position_rmse_reduction=(
                                keep_metrics.position_rmse
                                - metrics.position_rmse
                            ),
                            worst_node_position_rmse_reduction=(
                                keep_metrics.worst_node_position_rmse
                                - metrics.worst_node_position_rmse
                            ),
                            nees_calibration_improvement=(
                                abs(keep_metrics.mean_nees - 6.0)
                                - abs(metrics.mean_nees - 6.0)
                            ),
                            nees_coverage_calibration_improvement=(
                                abs(keep_metrics.nees_95_coverage - 0.95)
                                - abs(metrics.nees_95_coverage - 0.95)
                            ),
                            transmitted_message_cost=(
                                metrics.transmitted_message_count
                                - keep_metrics.transmitted_message_count
                            ),
                            replay_cost=(
                                metrics.replay_count
                                - keep_metrics.replay_count
                            ),
                            topology_change_cost=(
                                metrics.topology_change_count
                                - keep_metrics.topology_change_count
                            ),
                            endpoint_uncertainty_score_gain=(
                                action_score.endpoint_position_uncertainty
                                - keep_score.endpoint_position_uncertainty
                            ),
                            projected_uncertainty_score_gain=(
                                action_score.projected_position_uncertainty
                                - keep_score.projected_position_uncertainty
                            ),
                            approximate_trace_score_gain=(
                                action_score.approximate_trace_reduction
                                - keep_score.approximate_trace_reduction
                            ),
                            approximate_logdet_score_gain=(
                                action_score.approximate_logdet_reduction
                                - keep_score.approximate_logdet_reduction
                            ),
                            negative_distance_score_gain=(
                                action_score.negative_distance
                                - keep_score.negative_distance
                            ),
                            observation_age_score_gain=(
                                action_score.observation_age
                                - keep_score.observation_age
                            ),
                            recent_nis_score_gain=(
                                action_score.recent_mean_nis
                                - keep_score.recent_mean_nis
                            ),
                            negative_recent_nis_score_gain=(
                                action_score.negative_recent_mean_nis
                                - keep_score.negative_recent_mean_nis
                            ),
                            nis_calibration_score_gain=(
                                action_score.nis_calibration_quality
                                - keep_score.nis_calibration_quality
                            ),
                            nis_sample_count_score_gain=(
                                action_score.nis_sample_count
                                - keep_score.nis_sample_count
                            ),
                            negative_anomaly_score_gain=(
                                action_score.negative_consecutive_anomaly_count
                                - keep_score.negative_consecutive_anomaly_count
                            ),
                            communication_risk_score_gain=(
                                communication_risk(
                                    action.topology.active_edges
                                ) - keep_communication_risk
                            ),
                            resynchronization_cost=(
                                metrics.resynchronization_count
                                - keep_metrics.resynchronization_count
                            ),
                        ))
    record_values = tuple(records)
    return ShortHorizonCounterfactualStudy(
        node_counts=node_values,
        seeds=seed_values,
        decision_epochs=decision_values,
        horizon_epochs=horizon_values,
        relative_modalities=tuple(relative_modalities),
        decision_observations=tuple(decision_observations),
        records=record_values,
        summaries_by_action_kind=action_kind_summaries(record_values),
        swap_oracle_summary=swap_oracle_summary(record_values),
        swap_predictor_summaries=swap_predictor_selection_summaries(
            record_values
        ),
        swap_nis_retention_gate_summaries=(
            swap_nis_retention_gate_summaries(record_values)
        ),
        swap_abstention_summaries=swap_abstention_summaries(record_values),
    )


def action_kind_summaries(
    records: tuple[ShortHorizonActionRecord, ...],
) -> tuple[ActionKindSummary, ...]:
    values = []
    for action_kind in ("keep", "add", "swap", "remove"):
        selected = tuple(
            record for record in records if record.action_kind == action_kind
        )
        if not selected:
            continue
        rmse = np.asarray([
            record.position_rmse_reduction for record in selected
        ])
        values.append(ActionKindSummary(
            action_kind=action_kind,
            sample_count=len(selected),
            positive_rmse_gain_rate=float(np.mean(rmse > 0.0)),
            mean_position_rmse_reduction=float(np.mean(rmse)),
            median_position_rmse_reduction=float(np.median(rmse)),
            mean_covariance_trace_reduction=float(np.mean([
                record.covariance_trace_reduction for record in selected
            ])),
            mean_covariance_logdet_reduction=float(np.mean([
                record.covariance_logdet_reduction for record in selected
            ])),
            mean_transmitted_message_cost=float(np.mean([
                record.transmitted_message_cost for record in selected
            ])),
            nees_calibration_violation_rate=float(np.mean([
                record.nees_calibration_improvement < 0.0
                for record in selected
            ])),
            coverage_calibration_violation_rate=float(np.mean([
                record.nees_coverage_calibration_improvement < 0.0
                for record in selected
            ])),
        ))
    return tuple(values)


def swap_oracle_summary(
    records: tuple[ShortHorizonActionRecord, ...],
) -> SwapOracleSummary:
    groups: dict[tuple[int, int, int, int], list[float]] = {}
    for record in records:
        if record.action_kind == "swap":
            groups.setdefault((
                record.node_count,
                record.seed,
                record.decision_epoch,
                record.horizon_epochs,
            ), []).append(record.position_rmse_reduction)
    best = np.asarray([max(values) for values in groups.values()], dtype=float)
    return SwapOracleSummary(
        group_count=len(groups),
        positive_best_swap_rate=(
            float(np.mean(best > 0.0)) if best.size else 0.0
        ),
        mean_best_position_rmse_reduction=(
            float(np.mean(best)) if best.size else 0.0
        ),
        median_best_position_rmse_reduction=(
            float(np.median(best)) if best.size else 0.0
        ),
    )


def swap_predictor_selection_summaries(
    records: tuple[ShortHorizonActionRecord, ...],
) -> tuple[SwapPredictorSelectionSummary, ...]:
    """Evaluate deployable decision-time scores against the realized swap oracle."""

    groups: dict[
        tuple[int, int, int, int], list[ShortHorizonActionRecord]
    ] = {}
    for record in records:
        if record.action_kind == "swap":
            groups.setdefault((
                record.node_count,
                record.seed,
                record.decision_epoch,
                record.horizon_epochs,
            ), []).append(record)
    predictors = (
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
    )
    summaries = []
    for predictor in predictors:
        hits = []
        regrets = []
        selected_counts = []
        selected_gains = []
        selected_values = []
        for values in groups.values():
            predicted_best = max(getattr(value, predictor) for value in values)
            realized_best = max(
                value.position_rmse_reduction for value in values
            )
            selected = [
                value for value in values
                if np.isclose(getattr(value, predictor), predicted_best)
            ]
            best_edges = {
                (value.added_edges, value.removed_edges)
                for value in values
                if np.isclose(value.position_rmse_reduction, realized_best)
            }
            hits.append(any(
                (value.added_edges, value.removed_edges) in best_edges
                for value in selected
            ))
            selected_gain = float(np.mean([
                value.position_rmse_reduction for value in selected
            ]))
            selected_counts.append(len(selected))
            selected_gains.append(selected_gain)
            regrets.append(realized_best - selected_gain)
            selected_values.extend(selected)
        predicted_positive = [
            value for value in selected_values
            if getattr(value, predictor) > 0.0
        ]
        summaries.append(SwapPredictorSelectionSummary(
            predictor=predictor,
            group_count=len(groups),
            best_swap_hit_rate=float(np.mean(hits)) if hits else 0.0,
            mean_position_rmse_regret=(
                float(np.mean(regrets)) if regrets else 0.0
            ),
            median_position_rmse_regret=(
                float(np.median(regrets)) if regrets else 0.0
            ),
            mean_selected_count=(
                float(np.mean(selected_counts)) if selected_counts else 0.0
            ),
            mean_selected_position_rmse_reduction=(
                float(np.mean(selected_gains)) if selected_gains else 0.0
            ),
            selected_positive_rmse_gain_rate=(
                float(np.mean(np.asarray(selected_gains) > 0.0))
                if selected_gains else 0.0
            ),
            positive_gain_precision=(
                float(np.mean([
                    value.position_rmse_reduction > 0.0
                    for value in predicted_positive
                ]))
                if predicted_positive else None
            ),
            selected_nees_calibration_violation_rate=(
                float(np.mean([
                    value.nees_calibration_improvement < 0.0
                    for value in selected_values
                ]))
                if selected_values else 0.0
            ),
        ))
    return tuple(summaries)


def swap_nis_retention_gate_summaries(
    records: tuple[ShortHorizonActionRecord, ...],
    *,
    maximum_removed_edge_log_deviations=(0.1, 0.2, 0.3, 0.5, 1.0),
) -> tuple[SwapNisRetentionGateSummary, ...]:
    """Scan a pre-action gate while keeping NIS ranking otherwise unchanged.

    For a swap, ``nis_calibration_score_gain`` equals the calibration
    deviation of the removed historical edge because the new edge has no NIS
    history. If every swap in a group is gated out, the safe action is keep.
    """

    groups: dict[
        tuple[int, int, int, int], list[ShortHorizonActionRecord]
    ] = {}
    for record in records:
        if record.action_kind == "swap":
            groups.setdefault((
                record.node_count, record.seed,
                record.decision_epoch, record.horizon_epochs,
            ), []).append(record)
    values = []
    for threshold in maximum_removed_edge_log_deviations:
        if threshold < 0.0:
            raise ValueError("NIS retention gate threshold cannot be negative.")
        selected = []
        for candidates in groups.values():
            allowed = [
                record for record in candidates
                if record.nis_calibration_score_gain <= threshold
            ]
            selected.append(
                max(allowed, key=lambda record: record.recent_nis_score_gain)
                if allowed else None
            )
        gains = np.asarray([
            0.0 if record is None else record.position_rmse_reduction
            for record in selected
        ])
        violations = np.asarray([
            False if record is None
            else record.nees_calibration_improvement < 0.0
            for record in selected
        ])
        values.append(SwapNisRetentionGateSummary(
            maximum_removed_edge_log_deviation=float(threshold),
            group_count=len(groups),
            swap_execution_rate=float(np.mean([
                record is not None for record in selected
            ])) if selected else 0.0,
            mean_position_rmse_reduction=(
                float(np.mean(gains)) if gains.size else 0.0
            ),
            positive_rmse_gain_rate=(
                float(np.mean(gains > 0.0)) if gains.size else 0.0
            ),
            nees_calibration_violation_rate=(
                float(np.mean(violations)) if violations.size else 0.0
            ),
        ))
    return tuple(values)


def swap_abstention_summaries(
    records: tuple[ShortHorizonActionRecord, ...],
) -> tuple[SwapAbstentionSummary, ...]:
    """Select the highest-scored swap only when its gain versus keep is positive."""

    groups: dict[
        tuple[int, int, int, int], list[ShortHorizonActionRecord]
    ] = {}
    for record in records:
        if record.action_kind == "swap":
            groups.setdefault((
                record.node_count, record.seed,
                record.decision_epoch, record.horizon_epochs,
            ), []).append(record)
    predictors = (
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
    )
    summaries = []
    for predictor in predictors:
        selected = []
        for candidates in groups.values():
            best = max(candidates, key=lambda record: getattr(record, predictor))
            selected.append(best if getattr(best, predictor) > 0.0 else None)
        gains = np.asarray([
            0.0 if record is None else record.position_rmse_reduction
            for record in selected
        ])
        violations = np.asarray([
            False if record is None
            else record.nees_calibration_improvement < 0.0
            for record in selected
        ])
        summaries.append(SwapAbstentionSummary(
            predictor=predictor,
            group_count=len(groups),
            swap_execution_rate=float(np.mean([
                record is not None for record in selected
            ])) if selected else 0.0,
            mean_position_rmse_reduction=(
                float(np.mean(gains)) if gains.size else 0.0
            ),
            positive_rmse_gain_rate=(
                float(np.mean(gains > 0.0)) if gains.size else 0.0
            ),
            nees_calibration_violation_rate=(
                float(np.mean(violations)) if violations.size else 0.0
            ),
        ))
    return tuple(summaries)


def _unique_nonempty(name: str, values: tuple[int, ...]) -> tuple[int, ...]:
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{name} must be nonempty and unique.")
    return values
