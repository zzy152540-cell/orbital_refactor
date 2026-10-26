"""Audit physical communication topology for the frozen 15-satellite fleet.

This is supplemental evidence only.  It does not alter the fixed-ring
acceptance configuration or feed any result into the acceptance summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cooperative.topology import NetworkTopology, fully_connected_topology
from cooperative.topology_policy import LowChurnConnectedTreePolicy, build_graph_observation
from orbital_core.constants import R_EARTH
from scenarios.bilateral_multitarget import generate_bilateral_walker_truth
from scenarios.walker_scenario import WalkerDeltaConfig
from scenarios.measurement_visibility import (
    VisibilityConfig,
    generate_inter_satellite_observation_opportunities,
)


def run_prescan(
    *, duration: float, dt: float, maximum_ranges_km,
    friendly_plane_count: int = 3, friendly_phasing: int = 1,
) -> dict:
    friendly_config = WalkerDeltaConfig(
        total_satellites=15,
        plane_count=int(friendly_plane_count),
        phasing=int(friendly_phasing),
        semi_major_axis=R_EARTH + 700e3,
        eccentricity=0.0,
        inclination=np.deg2rad(53.0),
        node_prefix="friendly",
    )
    truth = generate_bilateral_walker_truth(
        duration_seconds=duration,
        sample_interval_seconds=dt,
        friendly_config=friendly_config,
    )
    node_ids = truth.manifest.friendly_observer_ids
    ring_edges = _edges(truth.manifest.friendly_topology)
    scans = [
        _scan_range(truth, maximum_range_km=value, ring_edges=ring_edges)
        for value in maximum_ranges_km
    ]
    return {
        "role": "supplemental_bilateral_dynamic_topology_prescan",
        "affectsAcceptanceConfiguration": False,
        "durationSeconds": float(duration),
        "sampleIntervalSeconds": float(dt),
        "friendlySatelliteCount": len(node_ids),
        "friendlyWalker": {
            "planeCount": int(friendly_plane_count),
            "phasing": int(friendly_phasing),
        },
        "fixedRing": {
            "undirectedEdgeCount": len(ring_edges),
            "nodeDegree": 2,
            "logicallyConnected": True,
        },
        "dynamicPolicy": {
            "name": "LowChurnConnectedTreePolicy",
            "maximumDegree": 3,
            "requiresConnectedCandidateGraph": True,
            "earthOccultation": True,
        },
        "rangeScans": scans,
    }


def run_geometry_scan(*, duration: float, dt: float, maximum_range_km: float):
    """Compare all valid 15-satellite Walker plane/phasing combinations."""

    rows = []
    for plane_count in (1, 3, 5, 15):
        for phasing in range(plane_count):
            config = WalkerDeltaConfig(
                total_satellites=15,
                plane_count=plane_count,
                phasing=phasing,
                semi_major_axis=R_EARTH + 700e3,
                eccentricity=0.0,
                inclination=np.deg2rad(53.0),
                node_prefix="friendly",
            )
            truth = generate_bilateral_walker_truth(
                duration_seconds=duration,
                sample_interval_seconds=dt,
                friendly_config=config,
            )
            scan = _scan_range(
                truth,
                maximum_range_km=maximum_range_km,
                ring_edges=_edges(truth.manifest.friendly_topology),
            )
            rows.append({
                "planeCount": plane_count,
                "phasing": phasing,
                **scan,
            })
    return sorted(
        rows,
        key=lambda item: (
            -item["connectedEpochFraction"],
            item["componentCountMax"],
            -item["candidateEdgeCountMin"],
            item["planeCount"],
            item["phasing"],
        ),
    )


def build_dynamic_topology_schedule(
    truth, *, maximum_range_km: float = 7000.0, maximum_degree: int = 3,
    inactive_node_ids_by_timestamp=None,
):
    """Build the supplemental low-churn physical topology for every epoch.

    This helper deliberately does not mutate ``truth.manifest``.  Callers must
    pass the returned schedule explicitly into an estimator run, which keeps
    the frozen fixed-ring acceptance path unchanged.
    """

    node_ids = truth.manifest.friendly_observer_ids
    opportunities = generate_inter_satellite_observation_opportunities(
        timestamps=truth.timestamps,
        truth_state_history_by_node=truth.friendly_truth_state_history_by_node,
        candidate_topology=fully_connected_topology(node_ids),
        visibility_by_modality={
            "LINK": VisibilityConfig(
                maximum_range=float(maximum_range_km) * 1000.0,
            )
        },
    )
    candidates_by_time = {float(value): {} for value in truth.timestamps}
    for item in opportunities:
        if item.visibility.visible:
            candidates_by_time[item.timestamp][tuple(sorted((
                item.observer_id, item.target_id,
            )))] = float(item.visibility.range)

    inactive_node_ids_by_timestamp = inactive_node_ids_by_timestamp or {}
    policy = LowChurnConnectedTreePolicy(maximum_degree=int(maximum_degree))
    previous_edges = set()
    schedule = {}
    change_count = 0
    active_component_counts = []
    largest_active_component_sizes = []
    for index, timestamp_value in enumerate(truth.timestamps):
        timestamp = float(timestamp_value)
        inactive = frozenset(
            inactive_node_ids_by_timestamp.get(timestamp, ())
        )
        unknown = inactive - set(node_ids)
        if unknown:
            raise ValueError(
                f"Inactive topology nodes are unknown: {sorted(unknown)}"
            )
        active_node_ids = tuple(
            node_id for node_id in node_ids if node_id not in inactive
        )
        candidates = {
            edge: distance
            for edge, distance in candidates_by_time[timestamp].items()
            if not (set(edge) & inactive)
        }
        components = _components(active_node_ids, candidates)
        active_component_counts.append(len(components))
        largest_active_component_sizes.append(max(map(len, components)))
        if len(components) != 1 and not inactive:
            raise ValueError(
                "Dynamic topology candidate graph is disconnected at "
                f"t={timestamp:.3f}s."
            )
        previous_active_edges = {
            edge for edge in previous_edges
            if not (set(edge) & inactive)
        }
        selected = set()
        for component in components:
            if len(component) == 1:
                continue
            component_nodes = set(component)
            component_candidates = {
                edge: distance for edge, distance in candidates.items()
                if set(edge) <= component_nodes
            }
            observation = build_graph_observation(
                timestamp=timestamp,
                state_by_node={
                    node_id: truth.friendly_truth_state_history_by_node[
                        node_id
                    ][index]
                    for node_id in component
                },
                candidate_distance_by_edge=component_candidates,
                previous_active_edges=tuple(sorted(
                    edge for edge in previous_active_edges
                    if set(edge) <= component_nodes
                )),
            )
            selected.update(policy.select(observation).active_edges)
        if previous_edges and selected != previous_edges:
            change_count += 1
        previous_edges = selected
        adjacency = {node_id: [] for node_id in node_ids}
        for left, right in sorted(selected):
            adjacency[left].append(right)
            adjacency[right].append(left)
        schedule[timestamp] = NetworkTopology(adjacency)
    return schedule, {
        "maximumRangeKm": float(maximum_range_km),
        "maximumDegree": int(maximum_degree),
        "epochCount": len(schedule),
        "edgeCountMin": min(
            len(_edges(topology)) for topology in schedule.values()
        ),
        "edgeCountMax": max(
            len(_edges(topology)) for topology in schedule.values()
        ),
        "topologyChangeCount": change_count,
        "containsFailureAwareReselection": bool(
            inactive_node_ids_by_timestamp
        ),
        "activeComponentCountMin": min(active_component_counts),
        "activeComponentCountMax": max(active_component_counts),
        "smallestLargestActiveComponentSize": min(
            largest_active_component_sizes
        ),
    }


def _scan_range(truth, *, maximum_range_km, ring_edges):
    maximum_range_m = (
        None if maximum_range_km is None else float(maximum_range_km) * 1000.0
    )
    node_ids = truth.manifest.friendly_observer_ids
    opportunities = generate_inter_satellite_observation_opportunities(
        timestamps=truth.timestamps,
        truth_state_history_by_node=truth.friendly_truth_state_history_by_node,
        candidate_topology=fully_connected_topology(node_ids),
        visibility_by_modality={
            "LINK": VisibilityConfig(maximum_range=maximum_range_m)
        },
    )
    candidates_by_time = {float(value): {} for value in truth.timestamps}
    for item in opportunities:
        if item.visibility.visible:
            candidates_by_time[item.timestamp][tuple(sorted((
                item.observer_id, item.target_id,
            )))] = float(item.visibility.range)

    candidate_counts = []
    component_counts = []
    largest_components = []
    maximum_candidate_degree = 0
    ring_visible_counts = []
    connected_epochs = 0
    selected_edge_counts = []
    topology_changes = 0
    previous_edges = set()
    policy = LowChurnConnectedTreePolicy(maximum_degree=3)
    for index, timestamp_value in enumerate(truth.timestamps):
        timestamp = float(timestamp_value)
        candidates = candidates_by_time[timestamp]
        candidate_counts.append(len(candidates))
        components = _components(node_ids, candidates)
        component_counts.append(len(components))
        largest_components.append(max(map(len, components)))
        degrees = {node_id: 0 for node_id in node_ids}
        for left, right in candidates:
            degrees[left] += 1
            degrees[right] += 1
        maximum_candidate_degree = max(
            maximum_candidate_degree, max(degrees.values())
        )
        ring_visible_counts.append(len(set(candidates) & ring_edges))
        if len(components) != 1:
            continue
        connected_epochs += 1
        observation = build_graph_observation(
            timestamp=timestamp,
            state_by_node={
                node_id: truth.friendly_truth_state_history_by_node[node_id][index]
                for node_id in node_ids
            },
            candidate_distance_by_edge=candidates,
            previous_active_edges=tuple(sorted(previous_edges)),
        )
        selected = set(policy.select(observation).active_edges)
        selected_edge_counts.append(len(selected))
        if previous_edges and selected != previous_edges:
            topology_changes += 1
        previous_edges = selected

    epoch_count = len(truth.timestamps)
    return {
        "maximumRangeKm": maximum_range_km,
        "epochCount": epoch_count,
        "candidateEdgeCountMin": min(candidate_counts),
        "candidateEdgeCountMax": max(candidate_counts),
        "maximumCandidateNodeDegree": maximum_candidate_degree,
        "componentCountMin": min(component_counts),
        "componentCountMax": max(component_counts),
        "smallestLargestComponentSize": min(largest_components),
        "connectedEpochCount": connected_epochs,
        "connectedEpochFraction": connected_epochs / epoch_count,
        "ringVisibleEdgeCountMin": min(ring_visible_counts),
        "ringVisibleEdgeCountMax": max(ring_visible_counts),
        "ringEdgeVisibilityFraction": (
            float(sum(ring_visible_counts)) / (epoch_count * len(ring_edges))
        ),
        "selectedTreeEdgeCountMin": (
            min(selected_edge_counts) if selected_edge_counts else None
        ),
        "selectedTreeEdgeCountMax": (
            max(selected_edge_counts) if selected_edge_counts else None
        ),
        "selectedTopologyChangeCountOnConnectedEpochs": topology_changes,
    }


def _edges(topology):
    return {
        tuple(sorted((node_id, neighbor_id)))
        for node_id in topology.node_ids
        for neighbor_id in topology.neighbors(node_id)
    }


def _components(node_ids, edges):
    adjacency = {node_id: set() for node_id in node_ids}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    remaining = set(node_ids)
    values = []
    while remaining:
        frontier = {next(iter(remaining))}
        component = set()
        while frontier:
            node_id = frontier.pop()
            if node_id in component:
                continue
            component.add(node_id)
            frontier.update(adjacency[node_id] - component)
        remaining.difference_update(component)
        values.append(tuple(sorted(component)))
    return tuple(sorted(values, key=lambda value: (-len(value), value)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--friendly-planes", type=int, default=3)
    parser.add_argument("--friendly-phasing", type=int, default=1)
    parser.add_argument(
        "--maximum-ranges-km", type=float, nargs="+",
        default=(3000.0, 5000.0, 7000.0, 10000.0),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "results/external_acceptance/"
            "bilateral_dynamic_topology_prescan_120s.json"
        ),
    )
    parser.add_argument(
        "--scan-geometries", action="store_true",
        help="Also scan every valid 15-satellite plane/phasing combination.",
    )
    args = parser.parse_args()
    result = run_prescan(
        duration=args.duration,
        dt=args.dt,
        maximum_ranges_km=tuple(args.maximum_ranges_km),
        friendly_plane_count=args.friendly_planes,
        friendly_phasing=args.friendly_phasing,
    )
    if args.scan_geometries:
        result["geometryScans"] = run_geometry_scan(
            duration=args.duration,
            dt=args.dt,
            maximum_range_km=max(args.maximum_ranges_km),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
