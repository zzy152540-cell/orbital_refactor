from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cooperative.network_schmidt_runner import run_network_schmidt_filter
from cooperative.topology import (
    NetworkTopology,
    chain_topology,
    fully_connected_topology,
)
from cooperative.topology_policy import (
    GraphObservation,
    TopologyAction,
    build_graph_observation,
    normalized_undirected_edge,
)
from experiments.edge_marginal_information import (
    EdgeMarginalInformation,
    evaluate_candidate_edge_marginals,
    topology_rollout_metrics_from_history,
)
from experiments.v14_exact_transport_scale_scan import build_exact_transport_case
from orbital_core.measurement_semantics import PHYSICAL_SENSOR_MODALITIES


@dataclass(frozen=True)
class SmallFleetEdgeMarginalResult:
    node_count: int
    seed: int
    decision_observation: GraphObservation
    baseline_action: TopologyAction
    edge_marginals: tuple[EdgeMarginalInformation, ...]


def run_small_fleet_edge_marginal_experiment(
    *, node_count: int, seed: int = 0, duration: float = 10.0,
    dt: float = 2.0,
    relative_modalities: tuple[str, ...] = PHYSICAL_SENSOR_MODALITIES,
) -> SmallFleetEdgeMarginalResult:
    """Run fixed-noise with/without-edge rollouts for a 3- or 5-node fleet."""

    if node_count not in {3, 5}:
        raise ValueError("The controlled marginal experiment supports 3 or 5 nodes.")
    node_ids = tuple(f"sat_{index + 1:02d}" for index in range(node_count))
    candidate_topology = fully_connected_topology(node_ids)
    baseline_topology = chain_topology(node_ids)
    candidate_edges = _topology_edges(candidate_topology)
    baseline_edges = _topology_edges(baseline_topology)
    baseline_action = TopologyAction("chain_baseline", baseline_edges)
    case = build_exact_transport_case(
        seed=seed, duration=duration, dt=dt,
        range_sigma=2.0, range_rate_sigma=0.05,
        az_el_sigma=np.deg2rad(0.05), optical_sigma=1e-3,
        absolute_sigma=3.0, process_noise_acceleration=1e-8,
        packet_loss=0.0, delay=0.0, acknowledge_messages=True,
        node_count=node_count, topology_type="edge_marginal_candidate_complete",
        topology_override=candidate_topology,
        relative_modalities=relative_modalities,
    )
    initial_truth = {node: case["truth"][node][0] for node in node_ids}
    decision_observation = build_graph_observation(
        timestamp=float(case["timestamps"][0]),
        state_by_node=case["initial_states"],
        covariance_by_node=case["initial_covariances"],
        candidate_distance_by_edge={
            edge: float(np.linalg.norm(
                initial_truth[edge[0]][:3] - initial_truth[edge[1]][:3]
            ))
            for edge in candidate_edges
        },
        measurement_modalities_by_edge={
            edge: relative_modalities for edge in candidate_edges
        },
        previous_active_edges=baseline_edges,
        estimation_dependency_edges=baseline_edges,
    )

    def evaluate_action(action: TopologyAction):
        topology = _topology_from_action(node_ids, action)
        active_edges = set(action.active_edges)
        observations = [
            observation for observation in case["observations"]
            if normalized_undirected_edge(
                observation.observer_id, observation.target_id
            ) in active_edges
        ]
        state_messages = {
            receiver: [
                message for message in case["state_messages"][receiver]
                if normalized_undirected_edge(
                    receiver, message.source_node_id
                ) in active_edges
            ]
            for receiver in node_ids
        }
        lineages = {
            edge: lineage for edge, lineage in case["lineages"].items()
            if normalized_undirected_edge(*edge) in active_edges
        }
        history = run_network_schmidt_filter(
            timestamps=case["timestamps"],
            initial_state_by_node=case["initial_states"],
            initial_covariance_by_node=case["initial_covariances"],
            topology=topology,
            observation_messages=observations,
            absolute_position_observations=case["absolute_observations"],
            observation_usage="observer_only",
            process_noise_acceleration=1e-8,
            consider_refresh_mode="exact_transport_event_replay",
            state_messages_by_receiver=state_messages,
            replay_history_window=10.0,
            expected_lineage_by_link=lineages,
        )
        replay_count = sum(
            value.replay_count
            for value in history.replay_performance_by_node.values()
        )
        return topology_rollout_metrics_from_history(
            history=history,
            truth_by_node=case["truth"],
            transmitted_message_count=sum(
                len(messages) for messages in state_messages.values()
            ),
            topology_change_count=len(active_edges ^ set(baseline_edges)),
            replay_count=replay_count,
        )

    return SmallFleetEdgeMarginalResult(
        node_count=node_count,
        seed=seed,
        decision_observation=decision_observation,
        baseline_action=baseline_action,
        edge_marginals=evaluate_candidate_edge_marginals(
            observation=decision_observation,
            baseline_action=baseline_action,
            evaluate_action=evaluate_action,
        ),
    )


def _topology_edges(topology: NetworkTopology):
    return tuple(sorted({
        normalized_undirected_edge(node, neighbor)
        for node in topology.node_ids
        for neighbor in topology.neighbors(node)
    }))


def _topology_from_action(
    node_ids: tuple[str, ...], action: TopologyAction,
) -> NetworkTopology:
    adjacency = {node: [] for node in node_ids}
    for left, right in action.active_edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    return NetworkTopology(adjacency)
