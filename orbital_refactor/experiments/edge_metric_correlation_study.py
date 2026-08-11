from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from cooperative.topology_policy import UndirectedEdge
from experiments.small_fleet_edge_marginal import (
    run_small_fleet_edge_marginal_experiment,
)


@dataclass(frozen=True)
class EdgeMetricRecord:
    node_count: int
    seed: int
    edge: UndirectedEdge
    covariance_trace_reduction: float
    covariance_logdet_reduction: float
    position_rmse_reduction: float
    worst_node_position_rmse_reduction: float
    nees_calibration_improvement: float
    nees_coverage_calibration_improvement: float
    transmitted_message_cost: int
    replay_cost: int
    resynchronization_cost: int


@dataclass(frozen=True)
class MetricCorrelation:
    predictor: str
    outcome: str
    sample_count: int
    pearson: float | None
    spearman: float | None


@dataclass(frozen=True)
class EdgeScoreSelectionSummary:
    predictor: str
    group_count: int
    best_edge_hit_rate: float
    mean_position_rmse_regret: float
    median_position_rmse_regret: float
    positive_gain_precision: float | None
    selected_nees_calibration_violation_rate: float
    selected_coverage_calibration_violation_rate: float


@dataclass(frozen=True)
class EdgeMetricCorrelationStudy:
    node_counts: tuple[int, ...]
    seeds: tuple[int, ...]
    records: tuple[EdgeMetricRecord, ...]
    pooled_correlations: tuple[MetricCorrelation, ...]
    correlations_by_node_count: dict[int, tuple[MetricCorrelation, ...]]
    selection_summaries: tuple[EdgeScoreSelectionSummary, ...]
    selection_summaries_by_node_count: dict[
        int, tuple[EdgeScoreSelectionSummary, ...]
    ]


def run_edge_metric_correlation_study(
    *, node_counts: tuple[int, ...] = (3, 5), seeds: Iterable[int] = range(5),
    duration: float = 10.0, dt: float = 2.0,
    relative_modalities: tuple[str, ...] = ("RANGE",),
) -> EdgeMetricCorrelationStudy:
    """Compare covariance proxies with realized accuracy and consistency gains."""

    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be nonempty and unique.")
    if not node_counts or len(set(node_counts)) != len(node_counts):
        raise ValueError("node_counts must be nonempty and unique.")
    records = []
    for node_count in node_counts:
        for seed in seed_values:
            result = run_small_fleet_edge_marginal_experiment(
                node_count=node_count, seed=seed, duration=duration, dt=dt,
                relative_modalities=relative_modalities,
            )
            for value in result.edge_marginals:
                with_metrics = value.metrics_with_edge
                without_metrics = value.metrics_without_edge
                records.append(EdgeMetricRecord(
                    node_count=node_count,
                    seed=seed,
                    edge=value.edge,
                    covariance_trace_reduction=value.covariance_trace_reduction,
                    covariance_logdet_reduction=value.covariance_logdet_reduction,
                    position_rmse_reduction=value.position_rmse_reduction,
                    worst_node_position_rmse_reduction=(
                        value.worst_node_position_rmse_reduction
                    ),
                    nees_calibration_improvement=(
                        abs(without_metrics.mean_nees - 6.0)
                        - abs(with_metrics.mean_nees - 6.0)
                    ),
                    nees_coverage_calibration_improvement=(
                        abs(without_metrics.nees_95_coverage - 0.95)
                        - abs(with_metrics.nees_95_coverage - 0.95)
                    ),
                    transmitted_message_cost=value.transmitted_message_cost,
                    replay_cost=value.replay_cost,
                    resynchronization_cost=value.resynchronization_cost,
                ))
    records_tuple = tuple(records)
    return EdgeMetricCorrelationStudy(
        node_counts=tuple(node_counts),
        seeds=seed_values,
        records=records_tuple,
        pooled_correlations=metric_correlations(records_tuple),
        correlations_by_node_count={
            node_count: metric_correlations(tuple(
                record for record in records_tuple
                if record.node_count == node_count
            ))
            for node_count in node_counts
        },
        selection_summaries=edge_score_selection_summaries(records_tuple),
        selection_summaries_by_node_count={
            node_count: edge_score_selection_summaries(tuple(
                record for record in records_tuple
                if record.node_count == node_count
            ))
            for node_count in node_counts
        },
    )


def metric_correlations(
    records: tuple[EdgeMetricRecord, ...],
) -> tuple[MetricCorrelation, ...]:
    """Calculate linear and rank correlations for the planned comparisons."""

    comparisons = (
        ("covariance_trace_reduction", "position_rmse_reduction"),
        ("covariance_logdet_reduction", "position_rmse_reduction"),
        ("covariance_trace_reduction", "worst_node_position_rmse_reduction"),
        ("covariance_logdet_reduction", "worst_node_position_rmse_reduction"),
        ("covariance_trace_reduction", "nees_calibration_improvement"),
        ("covariance_logdet_reduction", "nees_calibration_improvement"),
        (
            "covariance_trace_reduction",
            "nees_coverage_calibration_improvement",
        ),
        (
            "covariance_logdet_reduction",
            "nees_coverage_calibration_improvement",
        ),
    )
    values = []
    for predictor, outcome in comparisons:
        left = np.asarray([getattr(record, predictor) for record in records])
        right = np.asarray([getattr(record, outcome) for record in records])
        values.append(MetricCorrelation(
            predictor=predictor,
            outcome=outcome,
            sample_count=len(records),
            pearson=_correlation(left, right),
            spearman=_correlation(_average_ranks(left), _average_ranks(right)),
        ))
    return tuple(values)


def edge_score_selection_summaries(
    records: tuple[EdgeMetricRecord, ...],
) -> tuple[EdgeScoreSelectionSummary, ...]:
    """Measure whether a covariance proxy actually chooses the best RMSE edge."""

    groups: dict[tuple[int, int], list[EdgeMetricRecord]] = {}
    for record in records:
        groups.setdefault((record.node_count, record.seed), []).append(record)
    summaries = []
    for predictor in (
        "covariance_trace_reduction", "covariance_logdet_reduction"
    ):
        hits = []
        regrets = []
        selected_nees_violations = []
        selected_coverage_violations = []
        for values in groups.values():
            predicted_best = max(getattr(value, predictor) for value in values)
            realized_best = max(
                value.position_rmse_reduction for value in values
            )
            selected = [
                value for value in values
                if np.isclose(getattr(value, predictor), predicted_best)
            ]
            realized_best_edges = {
                value.edge for value in values
                if np.isclose(value.position_rmse_reduction, realized_best)
            }
            hits.append(any(
                value.edge in realized_best_edges for value in selected
            ))
            selected_gain = float(np.mean([
                value.position_rmse_reduction for value in selected
            ]))
            regrets.append(realized_best - selected_gain)
            selected_nees_violations.extend(
                value.nees_calibration_improvement < 0.0 for value in selected
            )
            selected_coverage_violations.extend(
                value.nees_coverage_calibration_improvement < 0.0
                for value in selected
            )
        predicted_positive = [
            record for record in records if getattr(record, predictor) > 0.0
        ]
        summaries.append(EdgeScoreSelectionSummary(
            predictor=predictor,
            group_count=len(groups),
            best_edge_hit_rate=float(np.mean(hits)) if hits else 0.0,
            mean_position_rmse_regret=(
                float(np.mean(regrets)) if regrets else 0.0
            ),
            median_position_rmse_regret=(
                float(np.median(regrets)) if regrets else 0.0
            ),
            positive_gain_precision=(
                float(np.mean([
                    value.position_rmse_reduction > 0.0
                    for value in predicted_positive
                ]))
                if predicted_positive else None
            ),
            selected_nees_calibration_violation_rate=(
                float(np.mean(selected_nees_violations))
                if selected_nees_violations else 0.0
            ),
            selected_coverage_calibration_violation_rate=(
                float(np.mean(selected_coverage_violations))
                if selected_coverage_violations else 0.0
            ),
        ))
    return tuple(summaries)


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or right.size != left.size:
        return None
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("Correlation inputs must be finite.")
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks
