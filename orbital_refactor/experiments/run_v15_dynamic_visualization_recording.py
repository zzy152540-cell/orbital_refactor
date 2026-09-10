"""Generate a 120 s Walker-20 replay with phased navigation/topology faults."""

from __future__ import annotations

import argparse
from pathlib import Path

from cooperative.network_schmidt_runner import run_network_schmidt_filter
from experiments.v14_walker_geometry_audit import run_v14_walker_geometry_audit
from experiments.walker_filter_setup import build_walker_filter_case
from experiments.v15_dynamic_visualization import phased_visualization_overlays
from visualization.cann_history_adapter import build_cann_visualization_history
from visualization.network_history_adapter import network_history_visualization_frames
from visualization.recording import VisualizationRecordingWriter


def generate_dynamic_walker_recording(output: str | Path, *, duration=120.0,
                                      dt=2.0, seed=0, include_cann=True):
    audit = run_v14_walker_geometry_audit(
        total_satellites=20, plane_count=10, phasing=1,
        duration=1800.0, dt=30.0, maximum_range=6000e3,
    )
    topology = audit.persistent_topology
    node_ids = topology.node_ids
    dropout_nodes = tuple(node_ids[:2])
    persistent_edges = sorted({
        tuple(sorted((node, neighbor)))
        for node in node_ids for neighbor in topology.neighbors(node)
    })
    inactive_edge = persistent_edges[0]
    case = build_walker_filter_case(
        seed=seed, duration=duration, dt=dt, maximum_range=6000e3,
        topology=topology,
        truth_history_by_node=audit.scenario.truth_state_history_by_node,
        topology_type="walker_persistent",
        absolute_navigation_dropout_windows_by_node={
            node: ((30.0, 58.0),) for node in dropout_nodes
        },
        topology_inactive_windows_by_undirected_edge={
            inactive_edge: ((60.0, 88.0),)
        },
    )
    history = run_network_schmidt_filter(
        timestamps=case["timestamps"], initial_state_by_node=case["initial_states"],
        initial_covariance_by_node=case["initial_covariances"],
        topology=case["topology"], observation_messages=case["observations"],
        absolute_position_observations=case["absolute_observations"],
        observation_usage="observer_only", process_noise_acceleration=1e-8,
        consider_refresh_mode="exact_transport_event_replay",
        state_messages_by_receiver=case["state_messages"],
        replay_history_window=10.0, expected_lineage_by_link=case["lineages"],
        topology_version_by_timestamp=case["topology_version_by_timestamp"],
        active_neighbors_by_timestamp=case["active_neighbors_by_timestamp"],
    )
    cann = None
    if include_cann:
        cann = build_cann_visualization_history(
            history=history, initial_state_by_node=case["initial_states"],
            absolute_position_observations=case["absolute_observations"],
        )
    edges, events = phased_visualization_overlays(
        case=case, dropout_nodes=dropout_nodes, inactive_edge=inactive_edge,
        dropout_window=(30.0, 58.0), topology_window=(60.0, 88.0),
    )
    frames = network_history_visualization_frames(
        history=history, truth_history_by_node=case["truth"],
        topology=case["topology"], observation_messages=case["observations"],
        absolute_position_observations=case["absolute_observations"],
        navigation_by_epoch=None if cann is None else cann.navigation_by_epoch,
        cann_by_epoch=None if cann is None else cann.cann_by_epoch,
        edges_by_epoch=edges, events_by_epoch=events,
        scenario_id="walker-20-dynamic-120s", run_id=f"seed-{seed}",
    )
    target = Path(output)
    with VisualizationRecordingWriter(target) as writer:
        for frame in frames:
            writer.append(frame)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(
        "results/visualization_recordings/walker20_120s_dynamic_cann"
    ))
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-cann", action="store_true")
    args = parser.parse_args()
    print(generate_dynamic_walker_recording(
        args.output, duration=args.duration, dt=args.dt, seed=args.seed,
        include_cann=not args.no_cann,
    ))


if __name__ == "__main__":
    main()
