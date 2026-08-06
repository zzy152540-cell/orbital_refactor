from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cooperative.topology import NetworkTopology, fully_connected_topology
from orbital_core.constants import R_EARTH
from scenarios.measurement_visibility import (
    VisibilityConfig,
    VisibilityOpportunitySummary,
    generate_inter_satellite_observation_opportunities,
    summarize_observation_opportunities,
)
from scenarios.walker_scenario import (
    WalkerDeltaConfig,
    WalkerDeltaScenario,
    generate_walker_delta_scenario,
)


@dataclass(frozen=True)
class WalkerGeometryAuditResult:
    scenario: WalkerDeltaScenario
    visibility_summary: VisibilityOpportunitySummary
    connected_at_every_epoch: bool
    minimum_visible_directed_edges: int
    maximum_visible_directed_edges: int
    minimum_initial_pair_range: float
    maximum_initial_pair_range: float
    persistent_undirected_edge_count: int
    persistent_component_sizes: tuple[int, ...]
    minimum_persistent_node_degree: int
    maximum_persistent_node_degree: int
    maximum_persistent_edge_range: float | None
    persistent_topology: NetworkTopology
    maximum_instantaneous_component_count: int
    minimum_largest_instantaneous_component: int


@dataclass(frozen=True)
class WalkerStaticTopologyScanResult:
    result_by_plane_and_phasing: dict[tuple[int, int], WalkerGeometryAuditResult]
    persistent_connected_candidates: tuple[tuple[int, int], ...]
    instantaneously_connected_candidates: tuple[tuple[int, int], ...]


def run_v14_walker_geometry_audit(
    *, total_satellites: int = 20, plane_count: int = 5,
    phasing: int = 1, altitude: float = 700e3,
    inclination: float = np.deg2rad(53.0), duration: float = 1800.0,
    dt: float = 30.0, maximum_range: float | None = None,
) -> WalkerGeometryAuditResult:
    """Generate a Walker constellation and audit all-pairs line of sight."""

    if altitude <= 0.0:
        raise ValueError("altitude must be positive.")
    if duration <= 0.0 or dt <= 0.0:
        raise ValueError("duration and dt must be positive.")
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt)
    scenario = generate_walker_delta_scenario(
        timestamps=timestamps,
        config=WalkerDeltaConfig(
            total_satellites=total_satellites, plane_count=plane_count,
            phasing=phasing, semi_major_axis=R_EARTH + altitude,
            eccentricity=0.0, inclination=inclination,
        ),
    )
    topology = fully_connected_topology(scenario.node_ids)
    opportunities = generate_inter_satellite_observation_opportunities(
        timestamps=timestamps,
        truth_state_history_by_node=scenario.truth_state_history_by_node,
        candidate_topology=topology,
        visibility_by_modality={
            "LINE_OF_SIGHT": VisibilityConfig(maximum_range=maximum_range)
        },
    )
    summary = summarize_observation_opportunities(opportunities)
    edges_by_timestamp = {
        float(timestamp): {
            tuple(sorted((item.observer_id, item.target_id)))
            for item in opportunities
            if item.timestamp == timestamp and item.visibility.visible
        }
        for timestamp in timestamps
    }
    component_sizes_by_timestamp = {
        timestamp: _component_sizes(scenario.node_ids, edges)
        for timestamp, edges in edges_by_timestamp.items()
    }
    connected = all(
        len(sizes) == 1 for sizes in component_sizes_by_timestamp.values()
    )
    persistent_edges = set.intersection(*(
        set(edges) for edges in edges_by_timestamp.values()
    ))
    persistent_component_sizes = _component_sizes(
        scenario.node_ids, persistent_edges,
    )
    persistent_degrees = {
        node: sum(node in edge for edge in persistent_edges)
        for node in scenario.node_ids
    }
    persistent_ranges = [
        item.visibility.range
        for item in opportunities
        if tuple(sorted((item.observer_id, item.target_id))) in persistent_edges
    ]
    persistent_adjacency = {node: [] for node in scenario.node_ids}
    for left, right in sorted(persistent_edges):
        persistent_adjacency[left].append(right)
        persistent_adjacency[right].append(left)
    initial_positions = np.vstack([
        scenario.truth_state_history_by_node[node][0, :3]
        for node in scenario.node_ids
    ])
    ranges = [
        float(np.linalg.norm(initial_positions[left] - initial_positions[right]))
        for left in range(total_satellites)
        for right in range(left + 1, total_satellites)
    ]
    edge_counts = summary.visible_directed_edge_count_by_timestamp
    return WalkerGeometryAuditResult(
        scenario=scenario, visibility_summary=summary,
        connected_at_every_epoch=connected,
        minimum_visible_directed_edges=min(edge_counts.values()),
        maximum_visible_directed_edges=max(edge_counts.values()),
        minimum_initial_pair_range=min(ranges),
        maximum_initial_pair_range=max(ranges),
        persistent_undirected_edge_count=len(persistent_edges),
        persistent_component_sizes=persistent_component_sizes,
        minimum_persistent_node_degree=min(persistent_degrees.values()),
        maximum_persistent_node_degree=max(persistent_degrees.values()),
        maximum_persistent_edge_range=(
            max(persistent_ranges) if persistent_ranges else None
        ),
        persistent_topology=NetworkTopology(persistent_adjacency),
        maximum_instantaneous_component_count=max(
            len(sizes) for sizes in component_sizes_by_timestamp.values()
        ),
        minimum_largest_instantaneous_component=min(
            sizes[0] for sizes in component_sizes_by_timestamp.values()
        ),
    )


def run_v14_walker_static_topology_scan(
    *, plane_counts: tuple[int, ...] = (4, 5, 10),
    phasing_values_by_plane: dict[int, tuple[int, ...]] | None = None,
    altitude: float = 700e3, inclination: float = np.deg2rad(53.0),
    duration: float = 1800.0, dt: float = 30.0,
    maximum_range: float | None = None,
) -> WalkerStaticTopologyScanResult:
    """Scan Walker plane/phasing choices for a physically valid static graph."""

    if not plane_counts or len(set(plane_counts)) != len(plane_counts):
        raise ValueError("plane_counts must be nonempty and unique.")
    results = {}
    for plane_count in plane_counts:
        if 20 % plane_count != 0:
            raise ValueError("Every plane_count must divide 20.")
        phasing_values = (
            tuple(range(plane_count))
            if phasing_values_by_plane is None
            else phasing_values_by_plane.get(plane_count, ())
        )
        if not phasing_values:
            raise ValueError("Every plane_count requires at least one phasing value.")
        if len(set(phasing_values)) != len(phasing_values):
            raise ValueError("Phasing values must be unique per plane count.")
        for phasing in phasing_values:
            results[(plane_count, phasing)] = run_v14_walker_geometry_audit(
                total_satellites=20, plane_count=plane_count, phasing=phasing,
                altitude=altitude, inclination=inclination,
                duration=duration, dt=dt, maximum_range=maximum_range,
            )
    persistent = tuple(sorted(
        key for key, result in results.items()
        if result.persistent_component_sizes == (20,)
    ))
    instantaneous = tuple(sorted(
        key for key, result in results.items()
        if result.connected_at_every_epoch
    ))
    return WalkerStaticTopologyScanResult(
        result_by_plane_and_phasing=results,
        persistent_connected_candidates=persistent,
        instantaneously_connected_candidates=instantaneous,
    )


def _component_sizes(node_ids, undirected_edges) -> tuple[int, ...]:
    adjacency = {node: set() for node in node_ids}
    for left, right in undirected_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    unseen = set(node_ids)
    sizes = []
    while unseen:
        root = unseen.pop()
        component = {root}
        pending = [root]
        while pending:
            node = pending.pop()
            for neighbor in adjacency[node] & unseen:
                unseen.remove(neighbor)
                component.add(neighbor)
                pending.append(neighbor)
        sizes.append(len(component))
    return tuple(sorted(sizes, reverse=True))
