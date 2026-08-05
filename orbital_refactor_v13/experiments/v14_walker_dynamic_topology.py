from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from cooperative.topology import NetworkTopology, fully_connected_topology
from cooperative.network_schmidt_runner import run_network_schmidt_filter
from cooperative.network_schmidt_orchestrator import NetworkSchmidtOrchestrator
from experiments.v14_exact_transport_scale_scan import _build_case, _metrics
from experiments.v14_online_topology_resynchronization import (
    _items_by_timestamp, _metrics as _online_metrics,
    _source_updates_from_messages,
)
from experiments.v14_walker_geometry_audit import _component_sizes
from orbital_core.constants import R_EARTH
from scenarios.measurement_visibility import (
    VisibilityConfig,
    generate_inter_satellite_observation_opportunities,
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


def build_v14_walker_dynamic_topology_plan(
    *, duration: float = 600.0, dt: float = 2.0,
    maximum_range: float = 7000e3, maximum_degree: int = 3,
) -> WalkerDynamicTopologyPlan:
    """Select a low-churn connected topology from Walker 20/5/3 LOS edges."""

    if duration <= 0.0 or dt <= 0.0:
        raise ValueError("duration and dt must be positive.")
    if maximum_degree < 2:
        raise ValueError("maximum_degree must be at least two.")
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt)
    scenario = generate_walker_delta_scenario(
        timestamps=timestamps,
        config=WalkerDeltaConfig(
            total_satellites=20, plane_count=5, phasing=3,
            semi_major_axis=R_EARTH + 700e3, eccentricity=0.0,
            inclination=np.deg2rad(53.0),
        ),
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
    for timestamp in timestamps:
        candidate_ranges = candidates_by_timestamp[float(timestamp)]
        selected = _select_connected_edges(
            scenario.node_ids, candidate_ranges,
            previous_edges=previous, maximum_degree=maximum_degree,
        )
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
        previous = selected
    return WalkerDynamicTopologyPlan(
        scenario=scenario, epoch_records=tuple(records),
        topology_by_timestamp=topology_by_timestamp,
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
    union_edges = set().union(*(
        set(record.active_undirected_edges) for record in plan.epoch_records
    ))
    adjacency = {node: [] for node in plan.scenario.node_ids}
    for left, right in sorted(union_edges):
        adjacency[left].append(right)
        adjacency[right].append(left)
    union_topology = NetworkTopology(adjacency)
    inactive_windows = _inactive_windows(plan, union_edges, dt=dt)
    initial_truth = {
        node: history[0]
        for node, history in plan.scenario.truth_state_history_by_node.items()
    }
    modalities = ("RADAR", "INFRARED", "OPTICAL")
    case = _build_case(
        seed=seed, duration=duration, dt=dt,
        range_sigma=2.0, range_rate_sigma=0.05,
        az_el_sigma=np.deg2rad(0.05), optical_sigma=1e-3,
        absolute_sigma=3.0, process_noise_acceleration=1e-8,
        packet_loss=0.0, delay=0.0, acknowledge_messages=True,
        node_count=20, topology_type="walker_dynamic_union",
        topology_override=union_topology,
        truth_initial_state_by_node=initial_truth,
        visibility_by_modality={
            modality: VisibilityConfig(maximum_range=maximum_range)
            for modality in modalities
        },
        topology_inactive_windows_by_undirected_edge=inactive_windows,
        relative_modalities=modalities,
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
    metrics = _metrics(
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
) -> WalkerOnlineDynamicFilterResult:
    """Run the Walker plan through lifecycle-aware online orchestration."""

    plan = build_v14_walker_dynamic_topology_plan(
        duration=duration, dt=dt, maximum_range=maximum_range,
        maximum_degree=maximum_degree,
    )
    union_edges = set().union(*(
        set(record.active_undirected_edges) for record in plan.epoch_records
    ))
    adjacency = {node: [] for node in plan.scenario.node_ids}
    for left, right in sorted(union_edges):
        adjacency[left].append(right)
        adjacency[right].append(left)
    union_topology = NetworkTopology(adjacency)
    initial_truth = {
        node: history[0]
        for node, history in plan.scenario.truth_state_history_by_node.items()
    }
    modalities = ("RADAR", "INFRARED", "OPTICAL")
    case = _build_case(
        seed=seed, duration=duration, dt=dt,
        range_sigma=2.0, range_rate_sigma=0.05,
        az_el_sigma=np.deg2rad(0.05), optical_sigma=1e-3,
        absolute_sigma=3.0, process_noise_acceleration=1e-8,
        packet_loss=0.0, delay=0.0, acknowledge_messages=True,
        node_count=20, topology_type="walker_dynamic_union",
        topology_override=union_topology,
        truth_initial_state_by_node=initial_truth,
        visibility_by_modality={
            modality: VisibilityConfig(maximum_range=maximum_range)
            for modality in modalities
        },
        relative_modalities=modalities,
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
    for index, (timestamp, record) in enumerate(zip(
        case["timestamps"], plan.epoch_records
    )):
        timestamp = float(timestamp)
        topology = plan.topology_by_timestamp[timestamp]
        active = {
            node: topology.neighbors(node) for node in topology.node_ids
        }
        result = orchestrator.step(
            timestamp, topology_version=record.version + 1,
            active_neighbors_by_node=active,
            source_update_by_node={
                node: source_updates[(node, timestamp)]
                for node in union_topology.node_ids
            },
            observations=observations_by_time.get(timestamp, ()),
            absolute_observations=absolute_by_time.get(timestamp, ()),
        )
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
        duration=duration, topology_change_count=plan.topology_change_count,
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
    )


def _select_connected_edges(
    node_ids, candidate_ranges, *, previous_edges, maximum_degree,
):
    nodes = tuple(node_ids)
    if _component_sizes(nodes, candidate_ranges) != (len(nodes),):
        raise ValueError("Candidate visibility graph is disconnected.")
    parent = {node: node for node in nodes}
    degree = {node: 0 for node in nodes}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    selected = set()
    ranked = sorted(
        candidate_ranges,
        key=lambda edge: (
            edge not in previous_edges, candidate_ranges[edge], edge,
        ),
    )
    for left, right in ranked:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            continue
        if degree[left] >= maximum_degree or degree[right] >= maximum_degree:
            continue
        selected.add((left, right))
        degree[left] += 1
        degree[right] += 1
        parent[right_root] = left_root
        if len(selected) == len(nodes) - 1:
            break
    if len(selected) != len(nodes) - 1:
        raise RuntimeError(
            "Greedy degree-constrained selector could not form a spanning tree."
        )
    return selected


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
