from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from cooperative.topology import NetworkTopology, fully_connected_topology
from cooperative.topology_policy import (
    GraphEdgeFeature, GraphNodeFeature, GraphObservation,
    LowChurnConnectedTreePolicy, TopologyAction, TopologyPolicy,
    build_graph_observation,
)
from cooperative.network_schmidt_runner import run_network_schmidt_filter
from cooperative.network_schmidt_orchestrator import NetworkSchmidtOrchestrator
from experiments.network_filter_metrics import network_history_metrics
from experiments.walker_filter_setup import (
    build_walker_filter_case,
    union_topology_from_epoch_records,
)
from experiments.walker_graph_dataset import (
    GraphTransition,
    WalkerGraphDataset,
    build_pre_walker_graph_observation,
    build_walker_graph_outcome,
)
from experiments.v14_online_topology_resynchronization import (
    _items_by_timestamp, _metrics as _online_metrics,
    _source_updates_from_messages,
)
from orbital_core.constants import R_EARTH
from scenarios.measurement_visibility import (
    generate_inter_satellite_observation_opportunities,
    VisibilityConfig,
)
from scenarios.walker_scenario import (
    WalkerDeltaConfig,
    WalkerDeltaScenario,
    generate_walker_delta_scenario,
)


@dataclass(frozen=True)
class DynamicTopologyEpoch:
    timestamp: float
    version: int
    candidate_undirected_edge_count: int
    active_undirected_edges: tuple[tuple[str, str], ...]
    added_edges: tuple[tuple[str, str], ...]
    removed_edges: tuple[tuple[str, str], ...]
    maximum_node_degree: int


@dataclass(frozen=True)
class WalkerDynamicTopologyPlan:
    scenario: WalkerDeltaScenario
    epoch_records: tuple[DynamicTopologyEpoch, ...]
    topology_by_timestamp: dict[float, NetworkTopology]
    graph_observation_by_timestamp: dict[float, GraphObservation]
    topology_change_count: int
    added_edge_count: int
    removed_edge_count: int
    minimum_candidate_edge_count: int
    maximum_candidate_edge_count: int
    minimum_active_edge_count: int
    maximum_active_edge_count: int
    maximum_selected_node_degree: int


@dataclass(frozen=True)
class WalkerDynamicFilterResult:
    duration: float
    topology_change_count: int
    configured_union_edge_count: int
    maximum_configured_node_degree: int
    mean_position_rmse: float
    mean_velocity_rmse: float
    mean_nees: float
    mean_nees_95_coverage: float
    mean_nis_by_modality: dict[str, float]
    message_acceptance_rate: float
    message_rejection_count: int
    rejection_counts: dict[str, int]
    psd_failure_count: int
    minimum_joint_eigenvalue: float
    run_seconds: float
    replay_count: int
    maximum_replay_seconds: float
    maximum_remote_event_count: int
    maximum_observation_count: int


@dataclass(frozen=True)
class WalkerOnlineDynamicFilterResult:
    duration: float
    topology_change_count: int
    configured_union_edge_count: int
    resynchronization_count: int
    resynchronized_links: tuple[tuple[str, str, str], ...]
    rejected_message_count: int
    rejection_counts_by_reason: dict[str, int]
    stale_topology_message_count: int
    protocol_rejected_message_count: int
    transmitted_message_count: int
    dropped_message_count: int
    position_rmse: float
    mean_nees: float
    nees_95_coverage: float
    psd_failure_count: int
    minimum_joint_eigenvalue: float
    final_lineage_by_directed_link: dict[tuple[str, str], str]
    maximum_checkpoint_count: int
    maximum_pinned_checkpoint_count: int
    maximum_retained_journal_count: int
    maximum_local_dimension: int
    packet_loss_rate: float
    communication_delay: float
    graph_dataset: WalkerGraphDataset


def build_v14_walker_dynamic_topology_plan(
    *, duration: float = 600.0, dt: float = 2.0,
    maximum_range: float = 7000e3, maximum_degree: int = 3,
    topology_policy: TopologyPolicy | None = None,
    walker_config: WalkerDeltaConfig | None = None,
) -> WalkerDynamicTopologyPlan:
    """Select a low-churn connected topology from Walker LOS edges."""

    if duration <= 0.0 or dt <= 0.0:
        raise ValueError("duration and dt must be positive.")
    if maximum_degree < 2:
        raise ValueError("maximum_degree must be at least two.")
    policy = topology_policy or LowChurnConnectedTreePolicy(maximum_degree)
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt)
    config = walker_config or WalkerDeltaConfig(
        total_satellites=20, plane_count=5, phasing=3,
        semi_major_axis=R_EARTH + 700e3, eccentricity=0.0,
        inclination=np.deg2rad(53.0),
    )
    scenario = generate_walker_delta_scenario(
        timestamps=timestamps,
        config=config,
    )
    opportunities = generate_inter_satellite_observation_opportunities(
        timestamps=timestamps,
        truth_state_history_by_node=scenario.truth_state_history_by_node,
        candidate_topology=fully_connected_topology(scenario.node_ids),
        visibility_by_modality={
            "LINK": VisibilityConfig(maximum_range=maximum_range)
        },
    )
    candidates_by_timestamp: dict[float, dict[tuple[str, str], float]] = {
        float(timestamp): {} for timestamp in timestamps
    }
    for item in opportunities:
        if not item.visibility.visible:
            continue
        edge = tuple(sorted((item.observer_id, item.target_id)))
        candidates_by_timestamp[item.timestamp][edge] = item.visibility.range

    previous: set[tuple[str, str]] = set()
    version = 0
    records = []
    topology_by_timestamp = {}
    graph_observation_by_timestamp = {}
    for index, timestamp in enumerate(timestamps):
        candidate_ranges = candidates_by_timestamp[float(timestamp)]
        graph_observation = build_graph_observation(
            timestamp=float(timestamp),
            state_by_node={
                node: scenario.truth_state_history_by_node[node][index]
                for node in scenario.node_ids
            },
            candidate_distance_by_edge=candidate_ranges,
            previous_active_edges=tuple(sorted(previous)),
        )
        selected = set(policy.select(graph_observation).active_edges)
        added = selected - previous
        removed = previous - selected
        if records and (added or removed):
            version += 1
        adjacency = {node: [] for node in scenario.node_ids}
        for left, right in sorted(selected):
            adjacency[left].append(right)
            adjacency[right].append(left)
        topology = NetworkTopology(adjacency)
        degrees = [len(topology.neighbors(node)) for node in scenario.node_ids]
        record = DynamicTopologyEpoch(
            timestamp=float(timestamp), version=version,
            candidate_undirected_edge_count=len(candidate_ranges),
            active_undirected_edges=tuple(sorted(selected)),
            added_edges=tuple(sorted(added)), removed_edges=tuple(sorted(removed)),
            maximum_node_degree=max(degrees),
        )
        records.append(record)
        topology_by_timestamp[float(timestamp)] = topology
        graph_observation_by_timestamp[float(timestamp)] = graph_observation
        previous = selected
    return WalkerDynamicTopologyPlan(
        scenario=scenario, epoch_records=tuple(records),
        topology_by_timestamp=topology_by_timestamp,
        graph_observation_by_timestamp=graph_observation_by_timestamp,
        topology_change_count=version,
        added_edge_count=sum(len(record.added_edges) for record in records[1:]),
        removed_edge_count=sum(len(record.removed_edges) for record in records[1:]),
        minimum_candidate_edge_count=min(
            record.candidate_undirected_edge_count for record in records
        ),
        maximum_candidate_edge_count=max(
            record.candidate_undirected_edge_count for record in records
        ),
        minimum_active_edge_count=min(
            len(record.active_undirected_edges) for record in records
        ),
        maximum_active_edge_count=max(
            len(record.active_undirected_edges) for record in records
        ),
        maximum_selected_node_degree=max(
            record.maximum_node_degree for record in records
        ),
    )


def run_v14_walker_dynamic_filter_smoke(
    *, duration: float = 120.0, dt: float = 2.0, seed: int = 0,
    maximum_range: float = 7000e3, maximum_degree: int = 3,
) -> WalkerDynamicFilterResult:
    """Run the current offline exact-replay path on a selected topology plan."""

    plan = build_v14_walker_dynamic_topology_plan(
        duration=duration, dt=dt, maximum_range=maximum_range,
        maximum_degree=maximum_degree,
    )
    union_topology, union_edges = union_topology_from_epoch_records(
        plan.scenario.node_ids, plan.epoch_records
    )
    inactive_windows = _inactive_windows(plan, union_edges, dt=dt)
    case = build_walker_filter_case(
        seed=seed, duration=duration, dt=dt, maximum_range=maximum_range,
        topology=union_topology,
        truth_history_by_node=plan.scenario.truth_state_history_by_node,
        topology_type="walker_dynamic_union",
        topology_inactive_windows_by_undirected_edge=inactive_windows,
    )
    started = perf_counter()
    history = run_network_schmidt_filter(
        timestamps=case["timestamps"],
        initial_state_by_node=case["initial_states"],
        initial_covariance_by_node=case["initial_covariances"],
        topology=case["topology"],
        observation_messages=case["observations"],
        absolute_position_observations=case["absolute_observations"],
        observation_usage="observer_only",
        process_noise_acceleration=1e-8,
        consider_refresh_mode="exact_transport_event_replay",
        state_messages_by_receiver=case["state_messages"],
        replay_history_window=10.0,
        expected_lineage_by_link=case["lineages"],
        topology_version_by_timestamp=case["topology_version_by_timestamp"],
        active_neighbors_by_timestamp=case["active_neighbors_by_timestamp"],
    )
    metrics = network_history_metrics(
        history, case["truth"], len(case["transmitted_messages"]),
        perf_counter() - started,
    )
    accepted = metrics[7]
    rejected = metrics[9]
    performance = metrics[13]
    return WalkerDynamicFilterResult(
        duration=duration, topology_change_count=plan.topology_change_count,
        configured_union_edge_count=len(union_edges),
        maximum_configured_node_degree=max(
            len(union_topology.neighbors(node)) for node in union_topology.node_ids
        ),
        mean_position_rmse=metrics[0], mean_velocity_rmse=metrics[1],
        mean_nees=metrics[2], mean_nees_95_coverage=metrics[3],
        mean_nis_by_modality=metrics[14],
        message_acceptance_rate=(
            accepted / (accepted + rejected) if accepted + rejected else 0.0
        ),
        message_rejection_count=rejected, rejection_counts=metrics[11],
        psd_failure_count=metrics[10], minimum_joint_eigenvalue=metrics[6],
        run_seconds=metrics[12], replay_count=performance["replay_count"],
        maximum_replay_seconds=performance["maximum_replay_seconds"],
        maximum_remote_event_count=performance["maximum_remote_event_count"],
        maximum_observation_count=performance["maximum_observation_count"],
    )


def run_v14_walker_online_dynamic_filter_smoke(
    *, duration: float = 60.0, dt: float = 2.0, seed: int = 0,
    maximum_range: float = 7000e3, maximum_degree: int = 3,
    max_pinned_age: float = 20.0,
    packet_loss_rate: float = 0.0, communication_delay: float = 0.0,
    topology_policy: TopologyPolicy | None = None,
) -> WalkerOnlineDynamicFilterResult:
    """Run the Walker plan through lifecycle-aware online orchestration."""

    plan = build_v14_walker_dynamic_topology_plan(
        duration=duration, dt=dt, maximum_range=maximum_range,
        maximum_degree=maximum_degree,
    )
    online_policy = topology_policy or LowChurnConnectedTreePolicy(
        maximum_degree
    )
    if topology_policy is None:
        union_topology, union_edges = union_topology_from_epoch_records(
            plan.scenario.node_ids, plan.epoch_records
        )
    else:
        union_edges = {
            edge.nodes
            for observation in plan.graph_observation_by_timestamp.values()
            for edge in observation.candidate_edges
        }
        union_adjacency = {node: [] for node in plan.scenario.node_ids}
        for left, right in sorted(union_edges):
            union_adjacency[left].append(right)
            union_adjacency[right].append(left)
        union_topology = NetworkTopology(union_adjacency)
    case = build_walker_filter_case(
        seed=seed, duration=duration, dt=dt, maximum_range=maximum_range,
        topology=union_topology,
        truth_history_by_node=plan.scenario.truth_state_history_by_node,
        topology_type="walker_dynamic_union",
    )
    source_updates = _source_updates_from_messages(
        case["transmitted_messages"], union_topology.node_ids,
    )
    observations_by_time = _items_by_timestamp(case["observations"])
    absolute_by_time = _items_by_timestamp(case["absolute_observations"])
    orchestrator = NetworkSchmidtOrchestrator(
        initial_state_by_node=case["initial_states"],
        initial_covariance_by_node=case["initial_covariances"],
        topology=union_topology, initial_timestamp=0.0,
        process_noise_acceleration=1e-8, history_window=10.0,
        max_pinned_age=max_pinned_age, packet_loss_rate=packet_loss_rate,
        communication_delay=communication_delay, random_seed=20270805 + seed,
        resynchronize_on_resume=True,
    )
    states = {
        node: np.zeros((len(case["timestamps"]), 6))
        for node in union_topology.node_ids
    }
    covariances = {
        node: np.zeros((len(case["timestamps"]), 6, 6))
        for node in union_topology.node_ids
    }
    rejected = stale = protocol_rejected = transmitted = dropped = 0
    rejection_counts = {}
    resynchronized = []
    minimum_eigenvalue = float("inf")
    psd_failures = 0
    pre_graph_observations = []
    topology_actions = []
    graph_outcomes = []
    previous_online_edges: set[tuple[str, str]] = set()
    online_topology_change_count = 0
    online_topology_version = 1
    for index, (timestamp, record) in enumerate(zip(
        case["timestamps"], plan.epoch_records
    )):
        timestamp = float(timestamp)
        relative_observations = observations_by_time.get(timestamp, ())
        prior_covariance_by_node = {
            node: session.state.active_covariance
            for node, session in orchestrator.sessions.items()
        }
        pre_graph_observations.append(build_pre_walker_graph_observation(
            timestamp=timestamp,
            plan_observation=plan.graph_observation_by_timestamp[timestamp],
            state_by_node={
                node: session.state.active_state
                for node, session in orchestrator.sessions.items()
            },
            covariance_by_node=prior_covariance_by_node,
            relative_observations=relative_observations,
            packet_loss_rate=packet_loss_rate,
            communication_delay=communication_delay,
            previous_active_edges=tuple(sorted(previous_online_edges)),
            estimation_dependency_edges={
                tuple(sorted((node, neighbor)))
                for node, session in orchestrator.sessions.items()
                for neighbor in session.state.neighbor_ids
            },
        ))
        action = online_policy.select(pre_graph_observations[-1])
        selected_edges = set(action.active_edges)
        unavailable = selected_edges - set(union_edges)
        if unavailable:
            raise ValueError(
                "Online policy selected edges outside the configured union: "
                f"{tuple(sorted(unavailable))}"
            )
        if index and selected_edges != previous_online_edges:
            online_topology_change_count += 1
            online_topology_version += 1
        topology_actions.append(action)
        adjacency = {node: [] for node in union_topology.node_ids}
        for left, right in sorted(selected_edges):
            adjacency[left].append(right)
            adjacency[right].append(left)
        topology = NetworkTopology(adjacency)
        active = {
            node: topology.neighbors(node) for node in topology.node_ids
        }
        result = orchestrator.step(
            timestamp, topology_version=online_topology_version,
            active_neighbors_by_node=active,
            source_update_by_node={
                node: source_updates[(node, timestamp)]
                for node in union_topology.node_ids
            },
            observations=relative_observations,
            absolute_observations=absolute_by_time.get(timestamp, ()),
        )
        graph_outcomes.append(build_walker_graph_outcome(
            timestamp=timestamp,
            step_result=result,
            relative_observations=relative_observations,
            prior_covariance_by_node=prior_covariance_by_node,
            action=topology_actions[-1],
            previous_active_edges=(
                pre_graph_observations[-1].previous_active_edges
            ),
        ))
        previous_online_edges = selected_edges
        rejected += result.rejected_message_count
        stale += result.stale_topology_message_count
        protocol_rejected += result.protocol_rejected_message_count
        transmitted += result.transmitted_message_count
        dropped += result.dropped_message_count
        resynchronized.extend(result.resynchronized_links)
        for reason, count in result.rejection_counts_by_reason.items():
            rejection_counts[reason] = rejection_counts.get(reason, 0) + count
        for node, step_result in result.result_by_node.items():
            states[node][index] = step_result.state.active_state
            covariances[node][index] = step_result.state.active_covariance
            eigenvalue = float(np.linalg.eigvalsh(
                step_result.state.joint_covariance
            ).min())
            minimum_eigenvalue = min(minimum_eigenvalue, eigenvalue)
            psd_failures += int(eigenvalue < -1e-8)
    metrics = _online_metrics(states, covariances, case["truth"])
    performances = [
        session.coordinator.performance
        for session in orchestrator.sessions.values()
    ]
    return WalkerOnlineDynamicFilterResult(
        duration=duration,
        topology_change_count=online_topology_change_count,
        configured_union_edge_count=len(union_edges),
        resynchronization_count=len(resynchronized),
        resynchronized_links=tuple(resynchronized),
        rejected_message_count=rejected,
        rejection_counts_by_reason=rejection_counts,
        stale_topology_message_count=stale,
        protocol_rejected_message_count=protocol_rejected,
        transmitted_message_count=transmitted, dropped_message_count=dropped,
        position_rmse=metrics["position_rmse"], mean_nees=metrics["nees"],
        nees_95_coverage=metrics["coverage"],
        psd_failure_count=psd_failures,
        minimum_joint_eigenvalue=minimum_eigenvalue,
        final_lineage_by_directed_link={
            (receiver, source): lifecycle.lineage_id
            for receiver, session in orchestrator.sessions.items()
            for source, lifecycle in session.link_by_neighbor.items()
        },
        maximum_checkpoint_count=max(
            value.maximum_checkpoint_count for value in performances
        ),
        maximum_pinned_checkpoint_count=max(
            value.maximum_pinned_checkpoint_count for value in performances
        ),
        maximum_retained_journal_count=max(
            value.maximum_retained_journal_count for value in performances
        ),
        maximum_local_dimension=max(
            session.state.dimension for session in orchestrator.sessions.values()
        ),
        packet_loss_rate=packet_loss_rate,
        communication_delay=communication_delay,
        graph_dataset=WalkerGraphDataset(
            feature_version="v14.2-causal",
            transitions=tuple(
                GraphTransition(
                    pre_observation=pre_graph_observations[index],
                    action=topology_actions[index],
                    outcome=graph_outcomes[index],
                    next_pre_observation=(
                        pre_graph_observations[index + 1]
                        if index + 1 < len(pre_graph_observations)
                        else None
                    ),
                )
                for index in range(len(pre_graph_observations))
            ),
        ),
    )


def _select_connected_edges(
    node_ids, candidate_ranges, *, previous_edges, maximum_degree,
):
    observation = GraphObservation(
        0.0,
        tuple(GraphNodeFeature(node, ()) for node in node_ids),
        tuple(GraphEdgeFeature(edge, float(distance))
              for edge, distance in sorted(candidate_ranges.items())),
        tuple(sorted(previous_edges)),
    )
    return set(LowChurnConnectedTreePolicy(maximum_degree).select(observation).active_edges)


def _inactive_windows(plan, union_edges, *, dt):
    windows = {}
    for edge in sorted(union_edges):
        inactive = [
            record.timestamp for record in plan.epoch_records
            if edge not in record.active_undirected_edges
        ]
        groups = []
        for timestamp in inactive:
            if not groups or not np.isclose(timestamp, groups[-1][1] + dt):
                groups.append([timestamp, timestamp])
            else:
                groups[-1][1] = timestamp
        if groups:
            windows[edge] = tuple((start, end) for start, end in groups)
    return windows
