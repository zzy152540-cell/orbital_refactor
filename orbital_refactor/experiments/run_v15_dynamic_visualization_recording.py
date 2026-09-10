"""Generate a 120 s Walker-20 replay with phased navigation/topology faults."""

from __future__ import annotations

import argparse
from pathlib import Path

from cooperative.message_transport import CommunicationWindow
from cooperative.network_orchestrator_history import network_history_from_orchestrator
from cooperative.network_schmidt_orchestrator import NetworkSchmidtOrchestrator
from experiments.v14_walker_geometry_audit import run_v14_walker_geometry_audit
from experiments.v14_online_topology_resynchronization import (
    _items_by_timestamp, _source_updates_from_messages,
)
from experiments.walker_filter_setup import build_walker_filter_case
from experiments.v15_dynamic_visualization import (
    COMMUNICATION_DEGRADATION_PROFILES, online_visualization_overlays,
)
from visualization.cann_history_adapter import build_cann_visualization_history
from visualization.network_history_adapter import network_history_visualization_frames
from visualization.recording import VisualizationRecordingWriter


def generate_dynamic_walker_recording(output: str | Path, *, duration=120.0,
                                      dt=2.0, seed=0, include_cann=True,
                                      communication_profile="mild"):
    if communication_profile not in COMMUNICATION_DEGRADATION_PROFILES:
        raise ValueError(f"Unknown communication profile: {communication_profile}")
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
            inactive_edge: ((76.0, 88.0),)
        },
    )
    profile = COMMUNICATION_DEGRADATION_PROFILES[communication_profile]
    communication_schedule = {
        (receiver, source): tuple(
            CommunicationWindow(start, end, packet_loss_rate=loss, delay=delay)
            for start, end, loss, delay in profile
        )
        for receiver in node_ids for source in topology.neighbors(receiver)
    }
    orchestrator = NetworkSchmidtOrchestrator(
        initial_state_by_node=case["initial_states"],
        initial_covariance_by_node=case["initial_covariances"],
        topology=case["topology"],
        initial_timestamp=float(case["timestamps"][0]),
        process_noise_acceleration=1e-8, history_window=10.0,
        max_pinned_age=10.0, communication_schedule_by_link=communication_schedule,
        random_seed=20260910 + seed, resynchronize_on_resume=True,
    )
    source_updates = _source_updates_from_messages(
        case["transmitted_messages"], node_ids,
    )
    observations = _items_by_timestamp(case["observations"])
    absolute = _items_by_timestamp(case["absolute_observations"])
    for timestamp in case["timestamps"]:
        timestamp = float(timestamp)
        orchestrator.step(
            timestamp,
            topology_version=case["topology_version_by_timestamp"][timestamp],
            active_neighbors_by_node=case["active_neighbors_by_timestamp"][timestamp],
            source_update_by_node={
                node: source_updates[(node, timestamp)] for node in node_ids
            },
            observations=observations.get(timestamp, ()),
            absolute_observations=absolute.get(timestamp, ()),
        )
    history = network_history_from_orchestrator(orchestrator)
    cann = None
    if include_cann:
        cann = build_cann_visualization_history(
            history=history, initial_state_by_node=case["initial_states"],
            absolute_position_observations=case["absolute_observations"],
        )
    edges, events, metadata = online_visualization_overlays(
        case=case, steps=orchestrator.history_snapshot(),
        dropout_nodes=dropout_nodes, dropout_window=(30.0, 58.0),
    )
    frames = network_history_visualization_frames(
        history=history, truth_history_by_node=case["truth"],
        topology=case["topology"], observation_messages=case["observations"],
        absolute_position_observations=case["absolute_observations"],
        navigation_by_epoch=None if cann is None else cann.navigation_by_epoch,
        cann_by_epoch=None if cann is None else cann.cann_by_epoch,
        edges_by_epoch=edges, events_by_epoch=events,
        metadata_by_epoch=metadata,
        scenario_id="walker-20-dynamic-120s",
        run_id=f"seed-{seed}-{communication_profile}",
    )
    target = Path(output)
    with VisualizationRecordingWriter(target) as writer:
        for frame in frames:
            writer.append(frame)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(
        "results/visualization_recordings/walker20_120s_online_mild_cann"
    ))
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-cann", action="store_true")
    parser.add_argument(
        "--communication-profile", choices=tuple(COMMUNICATION_DEGRADATION_PROFILES),
        default="mild",
    )
    args = parser.parse_args()
    print(generate_dynamic_walker_recording(
        args.output, duration=args.duration, dt=args.dt, seed=args.seed,
        include_cann=not args.no_cann,
        communication_profile=args.communication_profile,
    ))


if __name__ == "__main__":
    main()
