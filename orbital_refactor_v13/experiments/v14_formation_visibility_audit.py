from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from cooperative.topology import (
    chain_topology, fully_connected_topology, two_hop_chain_topology,
)
from cooperative.network_schmidt_runner import run_network_schmidt_filter
from experiments.v14_exact_transport_scale_scan import _build_case, _metrics
from orbital_core.constants import R_EARTH
from orbital_core.metrics import compute_nees_history, compute_rmse
from scenarios.fleet_scenario import (
    FleetScenario,
    centered_along_track_offsets,
    generate_differential_orbit_fleet_scenario,
)
from scenarios.measurement_visibility import (
    VisibilityConfig,
    VisibilityOpportunitySummary,
    generate_inter_satellite_observation_opportunities,
    summarize_observation_opportunities,
)


@dataclass(frozen=True)
class FormationVisibilityAuditResult:
    node_count: int
    duration: float
    dt: float
    along_track_spacing: float
    semi_major_axis_step: float
    topology_type: str
    maximum_range: float
    scenario: FleetScenario
    all_pairs_summary: VisibilityOpportunitySummary
    chain_summary: VisibilityOpportunitySummary
    initially_visible_undirected_edges: tuple[tuple[str, str], ...]
    connected_at_every_epoch: bool


@dataclass(frozen=True)
class PhysicalFleetFilterBaselineResult:
    node_count: int
    run_count: int
    mean_position_rmse: float
    mean_velocity_rmse: float
    mean_nees: float
    mean_nees_95_coverage: float
    mean_position_rmse_by_node: dict[str, float]
    mean_nees_by_node: dict[str, float]
    mean_nis_by_modality: dict[str, float]
    mean_nis_95_coverage_by_modality: dict[str, float]
    observation_count_by_modality_per_run: dict[str, int]
    message_acceptance_rate: float
    message_rejection_count: int
    psd_failure_count: int
    minimum_joint_eigenvalue: float
    mean_run_seconds: float
    replay_count: int
    maximum_replay_seconds: float
    maximum_remote_event_count: int
    maximum_observation_count: int
    maximum_checkpoint_count: int
    connected_at_every_epoch: bool
    visibility_summary: VisibilityOpportunitySummary


@dataclass(frozen=True)
class PhysicalVisibilityTransitionResult:
    loss: PhysicalFleetFilterBaselineResult
    recovery: PhysicalFleetFilterBaselineResult


@dataclass(frozen=True)
class StaggeredVisibilityScanResult:
    baseline: PhysicalFleetFilterBaselineResult
    transition_timestamps: tuple[float, ...]
    minimum_visible_directed_edges: int
    maximum_visible_directed_edges: int


@dataclass(frozen=True)
class PhysicalTopologyComparisonResult:
    nearest_neighbor_chain: PhysicalFleetFilterBaselineResult
    two_hop_chain: PhysicalFleetFilterBaselineResult


def run_v14_formation_visibility_audit(
    *, node_count: int, duration: float = 600.0, dt: float = 10.0,
    along_track_spacing: float = 750.0, semi_major_axis_step: float = 0.0,
    semi_major_axis_offsets: tuple[float, ...] | None = None,
    maximum_range: float = 5000.0, topology_type: str = "chain",
) -> FormationVisibilityAuditResult:
    """Audit a physical along-track fleet before coupling it to the filter.

    All-pairs opportunities describe the geometry-derived observation graph.
    The chain result separately checks the sparse topology intended for the
    first filter integration.  No communication or estimator state is changed.
    """

    if node_count < 2:
        raise ValueError("node_count must be at least two.")
    if duration <= 0.0 or dt <= 0.0:
        raise ValueError("duration and dt must be positive.")
    if maximum_range <= 0.0:
        raise ValueError("maximum_range must be positive.")
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt)
    topology_builders = {
        "chain": chain_topology,
        "two_hop_chain": two_hop_chain_topology,
    }
    if topology_type not in topology_builders:
        raise ValueError("Unsupported audit topology_type.")
    orbital_radius = R_EARTH + 700e3
    scenario = generate_differential_orbit_fleet_scenario(
        timestamps=timestamps,
        base_semi_major_axis=orbital_radius,
        eccentricity=0.001,
        inclination=np.deg2rad(23.0),
        raan=0.0,
        argument_of_perigee=0.0,
        base_true_anomaly=0.0,
        offset_by_node=centered_along_track_offsets(
            node_count=node_count, orbital_radius=orbital_radius,
            spacing=along_track_spacing,
            semi_major_axis_step=semi_major_axis_step,
            semi_major_axis_offsets=semi_major_axis_offsets,
        ),
    )
    visibility = {
        modality: VisibilityConfig(maximum_range=maximum_range)
        for modality in ("RADAR", "INFRARED", "OPTICAL")
    }
    all_pairs = generate_inter_satellite_observation_opportunities(
        timestamps=timestamps,
        truth_state_history_by_node=scenario.truth_state_history_by_node,
        candidate_topology=fully_connected_topology(scenario.node_ids),
        visibility_by_modality=visibility,
    )
    chain = generate_inter_satellite_observation_opportunities(
        timestamps=timestamps,
        truth_state_history_by_node=scenario.truth_state_history_by_node,
        candidate_topology=topology_builders[topology_type](scenario.node_ids),
        visibility_by_modality=visibility,
    )
    all_pairs_summary = summarize_observation_opportunities(all_pairs)
    chain_summary = summarize_observation_opportunities(chain)
    initial_edges = tuple(sorted({
        tuple(sorted((item.observer_id, item.target_id)))
        for item in all_pairs
        if item.timestamp == timestamps[0] and item.modality == "RADAR"
        and item.visibility.visible
    }))
    connected = all(
        _is_connected(scenario.node_ids, {
            tuple(sorted((item.observer_id, item.target_id)))
            for item in chain
            if item.timestamp == timestamp and item.modality == "RADAR"
            and item.visibility.visible
        })
        for timestamp in timestamps
    )
    return FormationVisibilityAuditResult(
        node_count=node_count, duration=duration, dt=dt,
        along_track_spacing=along_track_spacing,
        semi_major_axis_step=semi_major_axis_step,
        topology_type=topology_type,
        maximum_range=maximum_range,
        scenario=scenario, all_pairs_summary=all_pairs_summary,
        chain_summary=chain_summary,
        initially_visible_undirected_edges=initial_edges,
        connected_at_every_epoch=connected,
    )


def run_v14_physical_fleet_filter_baseline(
    *, node_count: int = 5, seeds: int = 10, duration: float = 120.0,
    dt: float = 2.0, along_track_spacing: float = 750.0,
    semi_major_axis_step: float = 0.0,
    semi_major_axis_offsets: tuple[float, ...] | None = None,
    maximum_range: float = 5000.0, topology_type: str = "chain",
) -> PhysicalFleetFilterBaselineResult:
    """Run the exact-replay filter on the audited physical chain formation."""

    if seeds < 1:
        raise ValueError("seeds must be at least one.")
    audit = run_v14_formation_visibility_audit(
        node_count=node_count, duration=duration, dt=dt,
        along_track_spacing=along_track_spacing,
        semi_major_axis_step=semi_major_axis_step,
        semi_major_axis_offsets=semi_major_axis_offsets,
        maximum_range=maximum_range,
        topology_type=topology_type,
    )
    initial_truth = {
        node: history[0]
        for node, history in audit.scenario.truth_state_history_by_node.items()
    }
    modalities = ("RADAR", "INFRARED", "OPTICAL")
    visibility = {
        modality: VisibilityConfig(maximum_range=maximum_range)
        for modality in modalities
    }
    values = []
    position_rmse_by_node: dict[str, list[float]] = {}
    nees_by_node: dict[str, list[float]] = {}
    observation_counts = None
    for seed in range(seeds):
        case = _build_case(
            seed=seed, duration=duration, dt=dt,
            range_sigma=2.0, range_rate_sigma=0.05,
            az_el_sigma=np.deg2rad(0.05), optical_sigma=1e-3,
            absolute_sigma=3.0, process_noise_acceleration=1e-8,
            packet_loss=0.0, delay=0.0, acknowledge_messages=True,
            node_count=node_count, topology_type=topology_type,
            truth_initial_state_by_node=initial_truth,
            visibility_by_modality=visibility,
            relative_modalities=modalities,
        )
        counts = {
            modality: sum(
                observation.modality == modality
                for observation in case["observations"]
            )
            for modality in modalities
        }
        if observation_counts is None:
            observation_counts = counts
        elif counts != observation_counts:
            raise RuntimeError("Physical visibility counts changed between seeds.")
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
        )
        values.append(_metrics(
            history, case["truth"], len(case["transmitted_messages"]),
            perf_counter() - started,
        ))
        for node in history.node_ids:
            error = history.active_state_history_by_node[node] - case["truth"][node]
            position_rmse_by_node.setdefault(node, []).append(
                compute_rmse(error[:, :3])
            )
            nees_by_node.setdefault(node, []).append(float(np.mean(
                compute_nees_history(
                    history.active_state_history_by_node[node],
                    case["truth"][node],
                    history.active_covariance_history_by_node[node],
                )
            )))
    accepted = sum(value[7] for value in values)
    rejected = sum(value[9] for value in values)
    return PhysicalFleetFilterBaselineResult(
        node_count=node_count, run_count=seeds,
        mean_position_rmse=float(np.mean([value[0] for value in values])),
        mean_velocity_rmse=float(np.mean([value[1] for value in values])),
        mean_nees=float(np.mean([value[2] for value in values])),
        mean_nees_95_coverage=float(np.mean([value[3] for value in values])),
        mean_position_rmse_by_node={
            node: float(np.mean(node_values))
            for node, node_values in position_rmse_by_node.items()
        },
        mean_nees_by_node={
            node: float(np.mean(node_values))
            for node, node_values in nees_by_node.items()
        },
        mean_nis_by_modality=_mean_dict([value[14] for value in values]),
        mean_nis_95_coverage_by_modality=_mean_dict(
            [value[15] for value in values]
        ),
        observation_count_by_modality_per_run=observation_counts or {},
        message_acceptance_rate=(
            accepted / (accepted + rejected) if accepted + rejected else 0.0
        ),
        message_rejection_count=rejected,
        psd_failure_count=sum(value[10] for value in values),
        minimum_joint_eigenvalue=min(value[6] for value in values),
        mean_run_seconds=float(np.mean([value[12] for value in values])),
        replay_count=sum(value[13]["replay_count"] for value in values),
        maximum_replay_seconds=max(
            value[13]["maximum_replay_seconds"] for value in values
        ),
        maximum_remote_event_count=max(
            value[13]["maximum_remote_event_count"] for value in values
        ),
        maximum_observation_count=max(
            value[13]["maximum_observation_count"] for value in values
        ),
        maximum_checkpoint_count=max(
            value[13]["maximum_checkpoint_count"] for value in values
        ),
        connected_at_every_epoch=audit.connected_at_every_epoch,
        visibility_summary=audit.chain_summary,
    )


def run_v14_physical_visibility_transition_experiment(
    *, node_count: int = 5, seeds: int = 10, duration: float = 120.0,
    dt: float = 2.0, maximum_range: float = 5000.0,
) -> PhysicalVisibilityTransitionResult:
    """Compare natural range loss and recovery with communication held ideal."""

    common = dict(
        node_count=node_count, seeds=seeds, duration=duration, dt=dt,
        maximum_range=maximum_range,
    )
    return PhysicalVisibilityTransitionResult(
        loss=run_v14_physical_fleet_filter_baseline(
            **common, along_track_spacing=4400.0,
            semi_major_axis_step=-2000.0,
        ),
        recovery=run_v14_physical_fleet_filter_baseline(
            **common, along_track_spacing=4800.0,
            semi_major_axis_step=2000.0,
        ),
    )


def run_v14_ten_satellite_staggered_visibility_scan(
    *, seeds: int = 3, duration: float = 120.0, dt: float = 2.0,
    maximum_range: float = 5000.0,
) -> StaggeredVisibilityScanResult:
    """Run a connected two-hop fleet with staggered redundant-edge losses."""

    increments = (-400.0, -500.0, -600.0, -700.0, -450.0,
                  -550.0, -650.0, -750.0, -800.0)
    offsets = [0.0]
    for increment in increments:
        offsets.append(offsets[-1] + increment)
    center = float(np.mean(offsets))
    centered_offsets = tuple(value - center for value in offsets)
    baseline = run_v14_physical_fleet_filter_baseline(
        node_count=10, seeds=seeds, duration=duration, dt=dt,
        along_track_spacing=2300.0,
        semi_major_axis_offsets=centered_offsets,
        maximum_range=maximum_range, topology_type="two_hop_chain",
    )
    edge_counts = baseline.visibility_summary.visible_directed_edge_count_by_timestamp
    ordered = sorted(edge_counts.items())
    transitions = tuple(
        timestamp
        for (_, previous), (timestamp, current) in zip(ordered, ordered[1:])
        if current != previous
    )
    return StaggeredVisibilityScanResult(
        baseline=baseline, transition_timestamps=transitions,
        minimum_visible_directed_edges=min(edge_counts.values()),
        maximum_visible_directed_edges=max(edge_counts.values()),
    )


def run_v14_ten_satellite_topology_comparison(
    *, seeds: int = 3, duration: float = 120.0, dt: float = 2.0,
) -> PhysicalTopologyComparisonResult:
    """Isolate nearest-neighbor versus two-hop topology on identical truth."""

    increments = (-400.0, -500.0, -600.0, -700.0, -450.0,
                  -550.0, -650.0, -750.0, -800.0)
    offsets = [0.0]
    for increment in increments:
        offsets.append(offsets[-1] + increment)
    center = float(np.mean(offsets))
    common = dict(
        node_count=10, seeds=seeds, duration=duration, dt=dt,
        along_track_spacing=2300.0,
        semi_major_axis_offsets=tuple(value - center for value in offsets),
        maximum_range=1e9,
    )
    return PhysicalTopologyComparisonResult(
        nearest_neighbor_chain=run_v14_physical_fleet_filter_baseline(
            **common, topology_type="chain",
        ),
        two_hop_chain=run_v14_physical_fleet_filter_baseline(
            **common, topology_type="two_hop_chain",
        ),
    )


def _is_connected(node_ids, edges) -> bool:
    adjacency = {node: set() for node in node_ids}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    reached = {node_ids[0]}
    pending = [node_ids[0]]
    while pending:
        node = pending.pop()
        for neighbor in adjacency[node] - reached:
            reached.add(neighbor)
            pending.append(neighbor)
    return len(reached) == len(node_ids)


def _mean_dict(values):
    keys = sorted({key for value in values for key in value})
    return {
        key: float(np.mean([value[key] for value in values if key in value]))
        for key in keys
    }
