from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from cooperative.topology_policy import (
    GraphObservation,
    TopologyAction,
    UndirectedEdge,
    normalized_undirected_edge,
)
from experiments.network_filter_metrics import (
    NEES_95_DOF6,
    modality_from_information_id,
    nis_interval,
)
from experiments.summary_statistics import interval_coverage
from orbital_core.metrics import compute_nees_history, compute_rmse


@dataclass(frozen=True)
class NodeModalityNisSummary:
    node_id: str
    modality: str
    sample_count: int
    mean_nis: float
    coverage_95: float
    upper_violation_rate: float


@dataclass(frozen=True)
class TopologyRolloutMetrics:
    """Comparable estimation and resource metrics from one fixed rollout."""

    mean_covariance_trace: float
    mean_covariance_logdet: float
    position_rmse: float
    mean_nees: float
    nees_95_coverage: float
    worst_node_position_rmse: float
    transmitted_message_count: int = 0
    topology_change_count: int = 0
    replay_count: int = 0
    resynchronization_count: int = 0
    position_rmse_by_node: tuple[tuple[str, float], ...] = ()
    mean_covariance_trace_by_node: tuple[tuple[str, float], ...] = ()
    mean_nees_by_node: tuple[tuple[str, float], ...] = ()
    nis_by_node_and_modality: tuple[NodeModalityNisSummary, ...] = ()


@dataclass(frozen=True)
class EdgeMarginalInformation:
    """Counterfactual value of keeping one edge, holding the case fixed."""

    edge: UndirectedEdge
    metrics_with_edge: TopologyRolloutMetrics
    metrics_without_edge: TopologyRolloutMetrics
    covariance_trace_reduction: float
    covariance_logdet_reduction: float
    position_rmse_reduction: float
    mean_nees_change: float
    nees_95_coverage_change: float
    worst_node_position_rmse_reduction: float
    transmitted_message_cost: int
    topology_change_cost: int
    replay_cost: int
    resynchronization_cost: int


RolloutEvaluator = Callable[[TopologyAction], TopologyRolloutMetrics]


def evaluate_candidate_edge_marginals(
    *, observation: GraphObservation, baseline_action: TopologyAction,
    evaluate_action: RolloutEvaluator,
    candidate_edges: Iterable[UndirectedEdge] | None = None,
) -> tuple[EdgeMarginalInformation, ...]:
    """Evaluate with/without-edge counterfactuals from identical pre-state.

    The callback must reset or clone its rollout state for every distinct action
    and keep truth, measurements, noise, delays, and random draws fixed.
    """

    available = {edge.nodes for edge in observation.candidate_edges}
    selected = set(baseline_action.active_edges)
    if selected - available:
        raise ValueError("baseline_action contains a non-candidate edge.")
    requested = (
        tuple(sorted(available)) if candidate_edges is None
        else tuple(normalized_undirected_edge(*edge) for edge in candidate_edges)
    )
    if len(set(requested)) != len(requested):
        raise ValueError("candidate_edges must be unique.")
    if set(requested) - available:
        raise ValueError("candidate_edges contains a non-candidate edge.")

    cache: dict[tuple[UndirectedEdge, ...], TopologyRolloutMetrics] = {}

    def metrics_for(edges: set[UndirectedEdge]) -> TopologyRolloutMetrics:
        normalized = tuple(sorted(edges))
        if normalized not in cache:
            cache[normalized] = evaluate_action(TopologyAction(
                policy_name="edge_counterfactual",
                active_edges=normalized,
            ))
        metrics = cache[normalized]
        _validate_metrics(metrics)
        return metrics

    values = []
    for edge in requested:
        with_edge = metrics_for(selected | {edge})
        without_edge = metrics_for(selected - {edge})
        values.append(EdgeMarginalInformation(
            edge=edge,
            metrics_with_edge=with_edge,
            metrics_without_edge=without_edge,
            covariance_trace_reduction=(
                without_edge.mean_covariance_trace
                - with_edge.mean_covariance_trace
            ),
            covariance_logdet_reduction=(
                without_edge.mean_covariance_logdet
                - with_edge.mean_covariance_logdet
            ),
            position_rmse_reduction=(
                without_edge.position_rmse - with_edge.position_rmse
            ),
            mean_nees_change=with_edge.mean_nees - without_edge.mean_nees,
            nees_95_coverage_change=(
                with_edge.nees_95_coverage - without_edge.nees_95_coverage
            ),
            worst_node_position_rmse_reduction=(
                without_edge.worst_node_position_rmse
                - with_edge.worst_node_position_rmse
            ),
            transmitted_message_cost=(
                with_edge.transmitted_message_count
                - without_edge.transmitted_message_count
            ),
            topology_change_cost=(
                with_edge.topology_change_count
                - without_edge.topology_change_count
            ),
            replay_cost=with_edge.replay_count - without_edge.replay_count,
            resynchronization_cost=(
                with_edge.resynchronization_count
                - without_edge.resynchronization_count
            ),
        ))
    return tuple(values)


def covariance_summary(covariance_by_node) -> tuple[float, float]:
    """Return fleet means of trace(P) and log(det(P))."""

    matrices = [np.asarray(value, dtype=float) for value in covariance_by_node.values()]
    if not matrices:
        raise ValueError("At least one covariance is required.")
    traces = []
    logdets = []
    for matrix in matrices:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Covariances must be square matrices.")
        sign, logdet = np.linalg.slogdet(matrix)
        if sign <= 0.0 or not np.isfinite(logdet):
            raise ValueError("Covariances must be positive definite.")
        traces.append(float(np.trace(matrix)))
        logdets.append(float(logdet))
    return float(np.mean(traces)), float(np.mean(logdets))


def topology_rollout_metrics_from_history(
    *, history, truth_by_node, transmitted_message_count=0,
    topology_change_count=0, replay_count=0, resynchronization_count=0,
    start_index: int = 0, stop_index: int | None = None,
) -> TopologyRolloutMetrics:
    """Adapt a production network-filter history to counterfactual metrics."""

    node_ids = tuple(history.node_ids)
    if set(node_ids) != set(truth_by_node):
        raise ValueError("History nodes and truth nodes must match.")
    sample_count = len(history.active_state_history_by_node[node_ids[0]])
    stop = sample_count if stop_index is None else int(stop_index)
    start = int(start_index)
    if not 0 <= start < stop <= sample_count:
        raise ValueError("Metric window must be a nonempty valid history slice.")
    covariance_trace = []
    covariance_logdet = []
    position_errors = []
    position_rmse_by_node = []
    covariance_trace_by_node = []
    nees_by_node = []
    nees = []
    for node in node_ids:
        node_covariance_trace = []
        state = np.asarray(
            history.active_state_history_by_node[node], dtype=float
        )[start:stop]
        truth = np.asarray(truth_by_node[node], dtype=float)[start:stop]
        covariance = np.asarray(
            history.active_covariance_history_by_node[node], dtype=float
        )[start:stop]
        if state.shape != truth.shape or covariance.shape != (
            state.shape[0], state.shape[1], state.shape[1]
        ):
            raise ValueError("History state, covariance, and truth shapes must align.")
        error = state[:, :3] - truth[:, :3]
        position_errors.append(error)
        position_rmse_by_node.append(compute_rmse(error))
        node_nees = compute_nees_history(state, truth, covariance)
        nees.extend(node_nees)
        nees_by_node.append(float(np.mean(node_nees)))
        for matrix in covariance:
            trace, logdet = covariance_summary({node: matrix})
            covariance_trace.append(trace)
            node_covariance_trace.append(trace)
            covariance_logdet.append(logdet)
        covariance_trace_by_node.append(float(np.mean(node_covariance_trace)))
    nees_array = np.asarray(nees, dtype=float)
    nis_summaries = []
    nis_history_by_node = getattr(history, "nis_history_by_node", {})
    for node in node_ids:
        by_modality = {}
        for epoch in nis_history_by_node.get(node, ())[start:stop]:
            for information_id, value in epoch.items():
                if ":absolute:" in information_id:
                    continue
                modality = modality_from_information_id(information_id)
                by_modality.setdefault(modality, []).append(float(value))
        for modality, values in sorted(by_modality.items()):
            array = np.asarray(values, dtype=float)
            lower, upper = nis_interval(modality)
            nis_summaries.append(NodeModalityNisSummary(
                node_id=node,
                modality=modality,
                sample_count=len(array),
                mean_nis=float(np.mean(array)),
                coverage_95=float(np.mean(
                    (array >= lower) & (array <= upper)
                )),
                upper_violation_rate=float(np.mean(array > upper)),
            ))
    return TopologyRolloutMetrics(
        mean_covariance_trace=float(np.mean(covariance_trace)),
        mean_covariance_logdet=float(np.mean(covariance_logdet)),
        position_rmse=compute_rmse(np.vstack(position_errors)),
        mean_nees=float(np.mean(nees_array)),
        nees_95_coverage=interval_coverage(nees_array, NEES_95_DOF6),
        worst_node_position_rmse=float(max(position_rmse_by_node)),
        transmitted_message_count=int(transmitted_message_count),
        topology_change_count=int(topology_change_count),
        replay_count=int(replay_count),
        resynchronization_count=int(resynchronization_count),
        position_rmse_by_node=tuple(
            (node, float(value))
            for node, value in zip(node_ids, position_rmse_by_node)
        ),
        mean_covariance_trace_by_node=tuple(
            (node, float(value))
            for node, value in zip(node_ids, covariance_trace_by_node)
        ),
        mean_nees_by_node=tuple(
            (node, float(value))
            for node, value in zip(node_ids, nees_by_node)
        ),
        nis_by_node_and_modality=tuple(nis_summaries),
    )


def _validate_metrics(metrics: TopologyRolloutMetrics) -> None:
    scalar_values = (
        metrics.mean_covariance_trace,
        metrics.mean_covariance_logdet,
        metrics.position_rmse,
        metrics.mean_nees,
        metrics.nees_95_coverage,
        metrics.worst_node_position_rmse,
    )
    if not all(np.isfinite(value) for value in scalar_values):
        raise ValueError("Rollout metrics must be finite.")
    if not 0.0 <= metrics.nees_95_coverage <= 1.0:
        raise ValueError("nees_95_coverage must be in [0, 1].")
    counts = (
        metrics.transmitted_message_count,
        metrics.topology_change_count,
        metrics.replay_count,
        metrics.resynchronization_count,
    )
    if any(value < 0 for value in counts):
        raise ValueError("Rollout resource counts cannot be negative.")
