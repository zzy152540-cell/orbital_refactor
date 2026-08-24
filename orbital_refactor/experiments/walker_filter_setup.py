from __future__ import annotations

import numpy as np

from cooperative.topology import NetworkTopology
from experiments.v14_exact_transport_scale_scan import build_exact_transport_case
from scenarios.measurement_visibility import VisibilityConfig

WALKER_FILTER_MODALITIES = ("RADAR", "INFRARED", "OPTICAL")


def union_topology_from_epoch_records(node_ids, epoch_records):
    """Build the configured topology containing every selected Walker edge."""

    union_edges = set().union(*(
        set(record.active_undirected_edges) for record in epoch_records
    ))
    adjacency = {node: [] for node in node_ids}
    for left, right in sorted(union_edges):
        adjacency[left].append(right)
        adjacency[right].append(left)
    return NetworkTopology(adjacency), union_edges


def build_walker_filter_case(
    *, seed, duration, dt, maximum_range, topology, truth_history_by_node,
    topology_type, topology_inactive_windows_by_undirected_edge=None,
    absolute_navigation_dropout_windows_by_node=None,
):
    """Build a physical-modality filter case for a Walker constellation."""

    initial_truth = {
        node: history[0] for node, history in truth_history_by_node.items()
    }
    return build_exact_transport_case(
        seed=seed, duration=duration, dt=dt,
        range_sigma=2.0, range_rate_sigma=0.05,
        az_el_sigma=np.deg2rad(0.05), optical_sigma=1e-3,
        absolute_sigma=3.0, process_noise_acceleration=1e-8,
        packet_loss=0.0, delay=0.0, acknowledge_messages=True,
        node_count=len(truth_history_by_node), topology_type=topology_type,
        topology_override=topology,
        truth_initial_state_by_node=initial_truth,
        visibility_by_modality={
            modality: VisibilityConfig(maximum_range=maximum_range)
            for modality in WALKER_FILTER_MODALITIES
        },
        topology_inactive_windows_by_undirected_edge=(
            topology_inactive_windows_by_undirected_edge
        ),
        absolute_navigation_dropout_windows_by_node=(
            absolute_navigation_dropout_windows_by_node
        ),
        relative_modalities=WALKER_FILTER_MODALITIES,
    )
