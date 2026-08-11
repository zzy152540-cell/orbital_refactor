from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from cooperative.network_schmidt_runner import run_network_schmidt_filter
from cooperative.topology import chain_topology, fully_connected_topology
from cooperative.topology_policy import (
    GraphObservation,
    GraphMeasurementFeature,
    TopologyAction,
    UndirectedEdge,
    build_graph_observation,
    normalized_undirected_edge,
)
from experiments.edge_marginal_information import (
    TopologyRolloutMetrics,
    topology_rollout_metrics_from_history,
)
from experiments.v14_exact_transport_scale_scan import (
    build_exact_transport_case,
)


@dataclass(frozen=True)
class TopologyCounterfactualAction:
    """A legal topology intervention applied after one decision epoch."""

    kind: str
    topology: TopologyAction
    added_edges: tuple[UndirectedEdge, ...] = ()
    removed_edges: tuple[UndirectedEdge, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"keep", "add", "swap", "remove"}:
            raise ValueError("Unsupported topology counterfactual action kind.")
        active = set(self.topology.active_edges)
        if set(self.added_edges) - active:
            raise ValueError("Added edges must be active in the action topology.")
        if set(self.removed_edges) & active:
            raise ValueError("Removed edges cannot remain active.")


@dataclass(frozen=True)
class CounterfactualRollout:
    action: TopologyCounterfactualAction
    metrics: TopologyRolloutMetrics
    decision_state_by_node: tuple[tuple[str, tuple[float, ...]], ...]
    decision_covariance_diagonal_by_node: tuple[
        tuple[str, tuple[float, ...]], ...
    ]
    recent_nis_by_edge: tuple[
        tuple[UndirectedEdge, tuple[tuple[str, float], ...]], ...
    ]
    nis_sample_count_by_edge: tuple[
        tuple[UndirectedEdge, tuple[tuple[str, int], ...]], ...
    ]
    consecutive_anomaly_count_by_edge: tuple[
        tuple[UndirectedEdge, tuple[tuple[str, int], ...]], ...
    ]
    observation_age_by_edge: tuple[tuple[UndirectedEdge, float], ...]


@dataclass(frozen=True)
class ShortHorizonCounterfactualResult:
    node_count: int
    seed: int
    decision_epoch: int
    horizon_epochs: int
    decision_observation: GraphObservation
    rollouts: tuple[CounterfactualRollout, ...]
    future_seed: int | None = None


def run_short_horizon_topology_counterfactual(
    *, node_count: int = 3, seed: int = 0, decision_epoch: int = 1,
    horizon_epochs: int = 3, dt: float = 2.0,
    relative_modalities: tuple[str, ...] = ("RANGE",),
    future_relative_update_order: tuple[str, ...] | None = None,
    future_seed: int | None = None,
    truth_initial_state_by_node: dict[str, np.ndarray] | None = None,
    inactive_edges_after_decision: tuple[UndirectedEdge, ...] = (),
    absolute_navigation_dropout_nodes_after_decision: tuple[str, ...] = (),
    disturbance_start_epoch: int | None = None,
) -> ShortHorizonCounterfactualResult:
    """Evaluate legal topology actions after a shared deterministic prefix."""

    if node_count not in {3, 5}:
        raise ValueError("The controlled experiment supports 3 or 5 nodes.")
    if decision_epoch < 0 or horizon_epochs <= 0 or dt <= 0.0:
        raise ValueError("Decision epoch, horizon, and dt are invalid.")
    if future_relative_update_order is not None and (
        len(set(future_relative_update_order))
        != len(future_relative_update_order)
        or set(future_relative_update_order) != set(relative_modalities)
    ):
        raise ValueError(
            "future_relative_update_order must be a permutation of "
            "relative_modalities."
        )
    disturbance_epoch = (
        decision_epoch
        if disturbance_start_epoch is None else int(disturbance_start_epoch)
    )
    if not 0 <= disturbance_epoch <= decision_epoch:
        raise ValueError(
            "disturbance_start_epoch must be between zero and decision_epoch."
        )
    node_ids = tuple(f"sat_{index + 1:02d}" for index in range(node_count))
    dropout_nodes = tuple(
        str(node) for node in absolute_navigation_dropout_nodes_after_decision
    )
    if len(set(dropout_nodes)) != len(dropout_nodes):
        raise ValueError("Absolute-navigation dropout nodes must be unique.")
    if set(dropout_nodes) - set(node_ids):
        raise ValueError("Absolute-navigation dropout references an unknown node.")
    candidate_topology = fully_connected_topology(node_ids)
    baseline_topology = chain_topology(node_ids)
    candidate_edges = _topology_edges(candidate_topology)
    baseline_edges = _topology_edges(baseline_topology)
    inactive_edges = tuple(
        normalized_undirected_edge(*edge)
        for edge in inactive_edges_after_decision
    )
    if len(set(inactive_edges)) != len(inactive_edges):
        raise ValueError("Inactive edges after decision must be unique.")
    if set(inactive_edges) - set(candidate_edges):
        raise ValueError("Inactive schedule references a non-candidate edge.")
    actions = build_counterfactual_actions(
        node_ids=node_ids,
        candidate_edges=candidate_edges,
        baseline_edges=baseline_edges,
    )
    duration = float((decision_epoch + horizon_epochs) * dt)
    decision_time = float(decision_epoch * dt)
    disturbance_start_time = float(disturbance_epoch * dt)
    case = build_exact_transport_case(
        seed=seed, duration=duration, dt=dt,
        range_sigma=2.0, range_rate_sigma=0.05,
        az_el_sigma=np.deg2rad(0.05), optical_sigma=1e-3,
        absolute_sigma=3.0, process_noise_acceleration=1e-8,
        packet_loss=0.0, delay=0.0, acknowledge_messages=True,
        node_count=node_count,
        topology_type="short_horizon_counterfactual_complete",
        topology_override=candidate_topology,
        relative_modalities=relative_modalities,
        truth_initial_state_by_node=truth_initial_state_by_node,
        topology_inactive_windows_by_undirected_edge={
            edge: ((disturbance_start_time, duration),)
            for edge in inactive_edges
        },
        absolute_navigation_dropout_windows_by_node={
            node: ((disturbance_start_time, duration),)
            for node in dropout_nodes
        },
        future_noise_seed=future_seed,
        future_noise_start_index=(
            None if future_seed is None else decision_epoch + 1
        ),
    )
    decision_time = float(case["timestamps"][decision_epoch])
    evaluation_start = decision_epoch + 1
    evaluation_stop = evaluation_start + horizon_epochs
    rollouts = tuple(
        _run_action(
            case=case,
            node_ids=node_ids,
            baseline_edges=baseline_edges,
            action=action,
            decision_time=decision_time,
            decision_epoch=decision_epoch,
            evaluation_start=evaluation_start,
            evaluation_stop=evaluation_stop,
            future_relative_update_order=future_relative_update_order,
        )
        for action in actions
    )
    reference = rollouts[0]
    if any(
        rollout.decision_state_by_node != reference.decision_state_by_node
        or rollout.decision_covariance_diagonal_by_node
        != reference.decision_covariance_diagonal_by_node
        or rollout.recent_nis_by_edge != reference.recent_nis_by_edge
        or rollout.nis_sample_count_by_edge
        != reference.nis_sample_count_by_edge
        or rollout.consecutive_anomaly_count_by_edge
        != reference.consecutive_anomaly_count_by_edge
        or rollout.observation_age_by_edge
        != reference.observation_age_by_edge
        for rollout in rollouts[1:]
    ):
        raise RuntimeError("Counterfactual rollouts do not share one prefix.")
    truth_at_decision = {
        node: case["truth"][node][decision_epoch] for node in node_ids
    }
    return ShortHorizonCounterfactualResult(
        node_count=node_count,
        seed=seed,
        decision_epoch=decision_epoch,
        horizon_epochs=horizon_epochs,
        decision_observation=build_graph_observation(
            timestamp=decision_time,
            state_by_node={
                node: np.asarray(state)
                for node, state in reference.decision_state_by_node
            },
            covariance_by_node={
                node: np.diag(diagonal)
                for node, diagonal
                in reference.decision_covariance_diagonal_by_node
            },
            estimator_metrics_by_node={
                node: {
                    "absolute_navigation_available": float(
                        node not in set(dropout_nodes)
                    )
                }
                for node in node_ids
            },
            candidate_distance_by_edge={
                edge: float(np.linalg.norm(
                    truth_at_decision[edge[0]][:3]
                    - truth_at_decision[edge[1]][:3]
                ))
                for edge in candidate_edges
            },
            measurement_modalities_by_edge={
                edge: relative_modalities for edge in candidate_edges
            },
            communication_available_by_edge={
                edge: edge not in set(inactive_edges)
                for edge in candidate_edges
            },
            nis_by_modality_by_edge={
                edge: dict(values)
                for edge, values in reference.recent_nis_by_edge
            },
            nis_sample_count_by_modality_by_edge={
                edge: dict(values)
                for edge, values in reference.nis_sample_count_by_edge
            },
            consecutive_anomaly_count_by_modality_by_edge={
                edge: dict(values)
                for edge, values
                in reference.consecutive_anomaly_count_by_edge
            },
            observation_age_by_edge=dict(
                reference.observation_age_by_edge
            ),
            previous_active_edges=baseline_edges,
            estimation_dependency_edges=baseline_edges,
            measurements=_decision_measurement_features(
                case["observations"], decision_time
            ),
        ),
        rollouts=rollouts,
        future_seed=future_seed,
    )


def build_counterfactual_actions(
    *, node_ids: tuple[str, ...],
    candidate_edges: tuple[UndirectedEdge, ...],
    baseline_edges: tuple[UndirectedEdge, ...],
) -> tuple[TopologyCounterfactualAction, ...]:
    """Enumerate keep, add, connected swap, and diagnostic remove actions."""

    candidates = set(candidate_edges)
    baseline = set(baseline_edges)
    if baseline - candidates or not _is_connected(node_ids, baseline):
        raise ValueError("Baseline must be a connected candidate topology.")
    values = [
        TopologyCounterfactualAction(
            "keep", TopologyAction("counterfactual_keep", tuple(sorted(baseline)))
        )
    ]
    for added in sorted(candidates - baseline):
        values.append(TopologyCounterfactualAction(
            "add",
            TopologyAction(
                "counterfactual_add", tuple(sorted(baseline | {added}))
            ),
            added_edges=(added,),
        ))
        for removed in sorted(baseline):
            swapped = (baseline | {added}) - {removed}
            if _is_connected(node_ids, swapped):
                values.append(TopologyCounterfactualAction(
                    "swap",
                    TopologyAction(
                        "counterfactual_swap", tuple(sorted(swapped))
                    ),
                    added_edges=(added,),
                    removed_edges=(removed,),
                ))
    for removed in sorted(baseline):
        values.append(TopologyCounterfactualAction(
            "remove",
            TopologyAction(
                "counterfactual_remove",
                tuple(sorted(baseline - {removed})),
            ),
            removed_edges=(removed,),
        ))
    return tuple(values)


def _order_future_relative_observations(
    observations, *, decision_time, update_order,
):
    """Reorder modalities within each future directed-link observation set."""

    if update_order is None:
        return list(observations)
    rank = {modality: index for index, modality in enumerate(update_order)}
    ordered = []
    index = 0
    while index < len(observations):
        observation = observations[index]
        if float(observation.timestamp) <= float(decision_time):
            ordered.append(observation)
            index += 1
            continue
        key = (
            float(observation.timestamp),
            observation.observer_id,
            observation.target_id,
        )
        stop = index + 1
        while stop < len(observations):
            candidate = observations[stop]
            candidate_key = (
                float(candidate.timestamp),
                candidate.observer_id,
                candidate.target_id,
            )
            if candidate_key != key:
                break
            stop += 1
        ordered.extend(sorted(
            observations[index:stop],
            key=lambda item: rank[item.modality],
        ))
        index = stop
    return ordered


def _run_action(
    *, case, node_ids, baseline_edges, action, decision_time,
    decision_epoch, evaluation_start, evaluation_stop,
    future_relative_update_order,
) -> CounterfactualRollout:
    baseline = set(baseline_edges)
    selected = set(action.topology.active_edges)
    active_neighbors_by_timestamp = {}
    topology_version_by_timestamp = {}
    for timestamp in case["timestamps"]:
        timestamp = float(timestamp)
        logical_edges = baseline if timestamp <= decision_time else selected
        physical_edges = {
            normalized_undirected_edge(node, neighbor)
            for node, neighbors in case[
                "active_neighbors_by_timestamp"
            ][timestamp].items()
            for neighbor in neighbors
        }
        edges = logical_edges & physical_edges
        physical_version = int(
            case["topology_version_by_timestamp"][timestamp]
        )
        version = (
            2 * physical_version
            + int(timestamp > decision_time and selected != baseline)
        )
        active_neighbors_by_timestamp[timestamp] = _adjacency(node_ids, edges)
        topology_version_by_timestamp[timestamp] = version

    observations = [
        observation for observation in case["observations"]
        if normalized_undirected_edge(
            observation.observer_id, observation.target_id
        ) in (
            baseline if observation.timestamp <= decision_time else selected
        )
    ]
    observations = _order_future_relative_observations(
        observations,
        decision_time=decision_time,
        update_order=future_relative_update_order,
    )
    state_messages = {}
    for receiver, messages in case["state_messages"].items():
        branch_messages = []
        for message in messages:
            edges = (
                baseline if message.timestamp <= decision_time else selected
            )
            edge = normalized_undirected_edge(receiver, message.source_node_id)
            if edge not in edges:
                continue
            physical_version = int(
                case["topology_version_by_timestamp"][
                    float(message.timestamp)
                ]
            )
            version = (
                2 * physical_version
                + int(message.timestamp > decision_time and selected != baseline)
            )
            branch_messages.append(replace(
                message,
                metadata={**message.metadata, "topology_version": version},
            ))
        state_messages[receiver] = branch_messages
    history = run_network_schmidt_filter(
        timestamps=case["timestamps"],
        initial_state_by_node=case["initial_states"],
        initial_covariance_by_node=case["initial_covariances"],
        topology=case["topology"],
        observation_messages=observations,
        absolute_position_observations=case["absolute_observations"],
        observation_usage="observer_only",
        process_noise_acceleration=1e-8,
        consider_refresh_mode="exact_transport_event_replay",
        state_messages_by_receiver=state_messages,
        replay_history_window=10.0,
        expected_lineage_by_link=case["lineages"],
        topology_version_by_timestamp=topology_version_by_timestamp,
        active_neighbors_by_timestamp=active_neighbors_by_timestamp,
        relative_observation_order=future_relative_update_order,
        relative_observation_order_start_time=decision_time,
    )
    transmitted_in_window = sum(
        evaluation_start <= _timestamp_index(case["timestamps"], message.timestamp)
        < evaluation_stop
        for messages in state_messages.values()
        for message in messages
    )
    metrics = topology_rollout_metrics_from_history(
        history=history,
        truth_by_node=case["truth"],
        transmitted_message_count=transmitted_in_window,
        topology_change_count=int(selected != baseline),
        replay_count=sum(
            value.replay_count
            for value in history.replay_performance_by_node.values()
        ),
        start_index=evaluation_start,
        stop_index=evaluation_stop,
    )
    recent_nis, sample_count, anomaly_count, observation_age = (
        _historical_edge_features(
        history=history,
        observations=case["observations"],
        candidate_edges=_topology_edges(case["topology"]),
        decision_epoch=decision_epoch,
        )
    )
    return CounterfactualRollout(
        action=action,
        metrics=metrics,
        decision_state_by_node=tuple(
            (node, tuple(
                float(value)
                for value in history.active_state_history_by_node[
                    node
                ][decision_epoch]
            ))
            for node in node_ids
        ),
        decision_covariance_diagonal_by_node=tuple(
            (node, tuple(
                float(value)
                for value in np.diag(
                    history.active_covariance_history_by_node[
                        node
                    ][decision_epoch]
                )
            ))
            for node in node_ids
        ),
        recent_nis_by_edge=recent_nis,
        nis_sample_count_by_edge=sample_count,
        consecutive_anomaly_count_by_edge=anomaly_count,
        observation_age_by_edge=observation_age,
    )


def _historical_edge_features(
    *, history, observations, candidate_edges, decision_epoch,
    window_epochs: int = 3,
):
    observation_by_id = {
        observation.information_id: observation
        for observation in observations
    }
    recent: dict[
        UndirectedEdge, dict[str, list[float]]
    ] = {edge: {} for edge in candidate_edges}
    last_epoch: dict[UndirectedEdge, int] = {}
    latest_anomaly: dict[UndirectedEdge, dict[str, tuple[int, int]]] = {}
    start = max(0, decision_epoch - window_epochs + 1)
    for node in history.node_ids:
        for index in range(decision_epoch + 1):
            for information_id, nis in (
                history.nis_history_by_node[node][index].items()
            ):
                observation = observation_by_id.get(information_id)
                if observation is None:
                    continue
                edge = normalized_undirected_edge(
                    observation.observer_id, observation.target_id
                )
                last_epoch[edge] = max(index, last_epoch.get(edge, -1))
                if index >= start:
                    recent[edge].setdefault(
                        observation.modality, []
                    ).append(float(nis))
                diagnostics = history.integrity_history_by_node[
                    node
                ][index].get(information_id)
                if diagnostics is not None:
                    previous = latest_anomaly.setdefault(edge, {}).get(
                        observation.modality
                    )
                    if previous is None or index >= previous[0]:
                        latest_anomaly[edge][observation.modality] = (
                            index, diagnostics.consecutive_anomaly_count
                        )
    return (
        tuple(sorted(
            (
                edge,
                tuple(sorted(
                    (modality, float(np.mean(values)))
                    for modality, values in by_modality.items()
                )),
            )
            for edge, by_modality in recent.items()
        )),
        tuple(sorted(
            (
                edge,
                tuple(sorted(
                    (modality, len(values))
                    for modality, values in by_modality.items()
                )),
            )
            for edge, by_modality in recent.items()
        )),
        tuple(sorted(
            (
                edge,
                tuple(sorted(
                    (modality, value[1])
                    for modality, value in latest_anomaly.get(edge, {}).items()
                )),
            )
            for edge in candidate_edges
        )),
        tuple(sorted(
            (
                edge,
                float(
                    decision_epoch - last_epoch[edge]
                    if edge in last_epoch else decision_epoch + 1
                ),
            )
            for edge in candidate_edges
        )),
    )


def _decision_measurement_features(observations, timestamp):
    values = []
    for observation in observations:
        if not np.isclose(observation.timestamp, timestamp):
            continue
        quaternion = observation.metadata.get("quaternion_i2b_wxyz")
        values.append(GraphMeasurementFeature(
            observer_id=observation.observer_id,
            target_id=observation.target_id,
            modality=observation.modality,
            frame=observation.frame,
            covariance=tuple(tuple(float(value) for value in row)
                             for row in np.asarray(observation.covariance)),
            quaternion_i2b_wxyz=(
                None if quaternion is None
                else tuple(float(value) for value in quaternion)
            ),
        ))
    return tuple(values)


def _timestamp_index(timestamps, timestamp: float) -> int:
    matches = np.flatnonzero(np.isclose(timestamps, float(timestamp)))
    if matches.size != 1:
        raise ValueError("Message timestamp is outside the rollout grid.")
    return int(matches[0])


def _topology_edges(topology) -> tuple[UndirectedEdge, ...]:
    return tuple(sorted({
        normalized_undirected_edge(node, neighbor)
        for node in topology.node_ids
        for neighbor in topology.neighbors(node)
    }))


def _adjacency(node_ids, edges):
    values = {node: [] for node in node_ids}
    for left, right in sorted(edges):
        values[left].append(right)
        values[right].append(left)
    return {node: tuple(neighbors) for node, neighbors in values.items()}


def _is_connected(node_ids, edges) -> bool:
    adjacency = _adjacency(node_ids, edges)
    visited = set()
    pending = [node_ids[0]]
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        pending.extend(adjacency[node])
    return len(visited) == len(node_ids)
