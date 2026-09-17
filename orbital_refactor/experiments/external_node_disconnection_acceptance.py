"""R10 paired Walker-20 complete node-disconnection acceptance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path

import numpy as np

from adapters.multimodal_sensor_simulator import MeasurementSource
from cooperative.network_schmidt_runner import run_network_schmidt_filter
from cooperative.topology import NetworkTopology
from experiments.v14_walker_geometry_audit import run_v14_walker_geometry_audit
from experiments.walker_filter_setup import build_walker_filter_case
from experiments.walker_raw_sensor_comparison import (
    replace_multimodal_messages_with_shared_frontend,
)
from orbital_core.dynamics import accel_two_body_j2
from orbital_core.metrics import compute_rmse


@dataclass(frozen=True)
class NodeDisconnectionRecord:
    seed: int
    disconnected_nodes: tuple[str, ...]
    normal_position_rmse_m: float
    disconnected_position_rmse_m: float
    position_rmse_increase_percent: float
    normal_velocity_rmse_mps: float
    disconnected_velocity_rmse_mps: float
    velocity_rmse_increase_percent: float
    normal_acceleration_rmse_mps2: float
    disconnected_acceleration_rmse_mps2: float
    acceleration_rmse_increase_percent: float
    normal_affected_position_rmse_m: float
    disconnected_affected_position_rmse_m: float
    affected_position_rmse_increase_percent: float
    normal_remaining_position_rmse_m: float
    disconnected_remaining_position_rmse_m: float
    remaining_position_rmse_increase_percent: float
    normal_position_error_p95_m: float
    disconnected_position_error_p95_m: float
    normal_worst_node_position_rmse_m: float
    disconnected_worst_node_position_rmse_m: float
    protocol_rejection_count: int


@dataclass(frozen=True)
class NodeDisconnectionReport:
    walker_definition: tuple[int, int, int]
    duration_seconds: float
    dt_seconds: float
    measurement_source: str
    run_count: int
    disconnected_node_count: int
    selection_pattern: str
    selection_seed: int | None
    threshold_percent: float
    formal_minimum_run_count: int
    formal_minimum_duration_seconds: float
    paired_mean_position_rmse_increase_percent: float
    paired_increase_95_half_width_percent: float
    worst_paired_position_rmse_increase_percent: float
    mean_affected_position_rmse_increase_percent: float
    mean_remaining_position_rmse_increase_percent: float
    threshold_met: bool
    formal_sample_size_met: bool
    formal_duration_met: bool
    passed: bool
    records: tuple[NodeDisconnectionRecord, ...]


def run_external_node_disconnection_acceptance(
    *, seeds=(0, 1, 2, 3, 4), duration=120.0, dt=2.0,
    maximum_range=6000e3, threshold_percent=15.0,
    selection_pattern="dispersed", selection_seed=0, disconnected_nodes=None,
    formal_minimum_run_count=20, formal_minimum_duration_seconds=1200.0,
    measurement_source_config=None,
):
    seeds = tuple(map(int, seeds))
    if not seeds:
        raise ValueError("seeds must be nonempty.")
    audit = run_v14_walker_geometry_audit(
        total_satellites=20, plane_count=10, phasing=1,
        duration=1800.0, dt=30.0, maximum_range=maximum_range,
    )
    topology = audit.persistent_topology
    selected = (
        select_disconnected_nodes(
            topology, selection_pattern, selection_seed=selection_seed
        )
        if disconnected_nodes is None else tuple(map(str, disconnected_nodes))
    )
    if len(selected) != 4 or len(set(selected)) != 4:
        raise ValueError("R10 requires exactly four unique disconnected nodes.")
    if not set(selected).issubset(topology.node_ids):
        raise ValueError("Disconnected nodes must belong to the topology.")
    isolated_topology = isolate_nodes(topology, selected)

    records = []
    for seed in seeds:
        case = build_walker_filter_case(
            seed=seed, duration=duration, dt=dt, maximum_range=maximum_range,
            topology=topology,
            truth_history_by_node=audit.scenario.truth_state_history_by_node,
            topology_type="walker_persistent",
        )
        _apply_measurement_source(case, measurement_source_config, seed=seed)
        normal = _run(case, topology)
        observations, messages, lineages = isolate_case_inputs(case, selected)
        degraded = _run(
            case, isolated_topology, observations=observations,
            state_messages=messages, lineages=lineages,
        )
        base = _metrics(normal, case["truth"], selected)
        loss = _metrics(degraded, case["truth"], selected)
        records.append(NodeDisconnectionRecord(
            seed=seed, disconnected_nodes=selected,
            normal_position_rmse_m=base["position"],
            disconnected_position_rmse_m=loss["position"],
            position_rmse_increase_percent=_increase(base["position"], loss["position"]),
            normal_velocity_rmse_mps=base["velocity"],
            disconnected_velocity_rmse_mps=loss["velocity"],
            velocity_rmse_increase_percent=_increase(base["velocity"], loss["velocity"]),
            normal_acceleration_rmse_mps2=base["acceleration"],
            disconnected_acceleration_rmse_mps2=loss["acceleration"],
            acceleration_rmse_increase_percent=_increase(base["acceleration"], loss["acceleration"]),
            normal_affected_position_rmse_m=base["affected"],
            disconnected_affected_position_rmse_m=loss["affected"],
            affected_position_rmse_increase_percent=_increase(base["affected"], loss["affected"]),
            normal_remaining_position_rmse_m=base["remaining"],
            disconnected_remaining_position_rmse_m=loss["remaining"],
            remaining_position_rmse_increase_percent=_increase(base["remaining"], loss["remaining"]),
            normal_position_error_p95_m=base["p95"],
            disconnected_position_error_p95_m=loss["p95"],
            normal_worst_node_position_rmse_m=base["worst_node"],
            disconnected_worst_node_position_rmse_m=loss["worst_node"],
            protocol_rejection_count=sum(
                degraded.refresh_diagnostics.get(name, 0) for name in (
                    "stale_rejected", "lineage_rejected", "history_rejected",
                    "event_bundle_rejected",
                )
            ),
        ))
    increases = np.array([item.position_rmse_increase_percent for item in records])
    half_width = (
        1.96 * np.std(increases, ddof=1) / np.sqrt(increases.size)
        if increases.size > 1 else np.nan
    )
    threshold_met = bool(np.mean(increases) <= threshold_percent)
    sample_met = len(records) >= formal_minimum_run_count
    duration_met = duration >= formal_minimum_duration_seconds
    protocol_ok = all(item.protocol_rejection_count == 0 for item in records)
    return NodeDisconnectionReport(
        walker_definition=(20, 10, 1), duration_seconds=float(duration),
        dt_seconds=float(dt),
        measurement_source=(
            "analytic" if measurement_source_config is None
            else measurement_source_config.source.value
        ),
        run_count=len(records), disconnected_node_count=4,
        selection_pattern=selection_pattern,
        selection_seed=(
            int(selection_seed) if selection_pattern == "random" else None
        ),
        threshold_percent=threshold_percent,
        formal_minimum_run_count=formal_minimum_run_count,
        formal_minimum_duration_seconds=formal_minimum_duration_seconds,
        paired_mean_position_rmse_increase_percent=float(np.mean(increases)),
        paired_increase_95_half_width_percent=float(half_width),
        worst_paired_position_rmse_increase_percent=float(np.max(increases)),
        mean_affected_position_rmse_increase_percent=float(np.mean([
            item.affected_position_rmse_increase_percent for item in records
        ])),
        mean_remaining_position_rmse_increase_percent=float(np.mean([
            item.remaining_position_rmse_increase_percent for item in records
        ])),
        threshold_met=threshold_met, formal_sample_size_met=sample_met,
        formal_duration_met=duration_met,
        passed=bool(threshold_met and sample_met and duration_met and protocol_ok),
        records=tuple(records),
    )


def _apply_measurement_source(case, source_config, *, seed):
    if source_config is None or source_config.source is MeasurementSource.ANALYTIC:
        return
    case["observations"] = tuple(
        replace_multimodal_messages_with_shared_frontend(
            case["observations"], timestamps=case["timestamps"],
            truth_state_history_by_node=case["truth"],
            config=source_config.sensors, random_seed=3_000_000 + int(seed),
            covariance_calibration_by_modality=(
                source_config.covariance_calibration_by_modality
            ),
        )
    )


def save_external_node_disconnection_report(report, output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = output / "summary.json", output / "records.csv"
    json_path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=NodeDisconnectionRecord.__dataclass_fields__)
        writer.writeheader()
        for item in report.records:
            row = asdict(item)
            row["disconnected_nodes"] = ";".join(item.disconnected_nodes)
            writer.writerow(row)
    return json_path, csv_path


def select_disconnected_nodes(topology, pattern, *, selection_seed=0):
    nodes = tuple(sorted(topology.node_ids))
    if len(nodes) != 20:
        raise ValueError("R10 requires a 20-node topology.")
    if pattern == "dispersed":
        return tuple(nodes[index] for index in (0, 5, 10, 15))
    if pattern == "adjacent":
        selected = [nodes[0]]
        while len(selected) < 4:
            candidates = sorted({
                neighbor for node in selected for neighbor in topology.neighbors(node)
                if neighbor not in selected
            })
            if not candidates:
                raise ValueError("Could not select four connected nodes.")
            selected.append(candidates[0])
        return tuple(selected)
    if pattern == "random":
        rng = np.random.default_rng(int(selection_seed))
        indices = np.sort(rng.choice(len(nodes), size=4, replace=False))
        return tuple(nodes[index] for index in indices)
    raise ValueError(
        "selection_pattern must be 'dispersed', 'adjacent', or 'random'."
    )


def isolate_nodes(topology, selected):
    selected = set(selected)
    return NetworkTopology({
        node: tuple(
            neighbor for neighbor in topology.neighbors(node)
            if node not in selected and neighbor not in selected
        ) for node in topology.node_ids
    })


def isolate_case_inputs(case, selected):
    selected = set(selected)
    observations = tuple(
        item for item in case["observations"]
        if item.observer_id not in selected and item.target_id not in selected
    )
    messages = {
        receiver: tuple(
            item for item in values
            if receiver not in selected and item.source_node_id not in selected
            and item.target_node_id not in selected
        ) for receiver, values in case["state_messages"].items()
    }
    lineages = {
        edge: value for edge, value in case["lineages"].items()
        if edge[0] not in selected and edge[1] not in selected
    }
    return observations, messages, lineages


def _run(case, topology, *, observations=None, state_messages=None, lineages=None):
    return run_network_schmidt_filter(
        timestamps=case["timestamps"], initial_state_by_node=case["initial_states"],
        initial_covariance_by_node=case["initial_covariances"], topology=topology,
        observation_messages=case["observations"] if observations is None else observations,
        absolute_position_observations=case["absolute_observations"],
        observation_usage="observer_only", process_noise_acceleration=1e-8,
        consider_refresh_mode="exact_transport_event_replay",
        state_messages_by_receiver=case["state_messages"] if state_messages is None else state_messages,
        replay_history_window=10.0,
        expected_lineage_by_link=case["lineages"] if lineages is None else lineages,
    )


def _metrics(history, truth, affected_nodes):
    nodes = tuple(history.node_ids)
    affected = tuple(affected_nodes)
    remaining = tuple(node for node in nodes if node not in set(affected))
    position = {
        node: history.active_state_history_by_node[node][:, :3] - truth[node][:, :3]
        for node in nodes
    }
    velocity = np.vstack([
        history.active_state_history_by_node[node][:, 3:] - truth[node][:, 3:]
        for node in nodes
    ])
    acceleration = np.vstack([np.array([
        accel_two_body_j2(value[:3]) - accel_two_body_j2(reference[:3])
        for value, reference in zip(history.active_state_history_by_node[node], truth[node])
    ]) for node in nodes])
    pooled = np.vstack(list(position.values()))
    by_node = {node: compute_rmse(value) for node, value in position.items()}
    return {
        "position": compute_rmse(pooled), "velocity": compute_rmse(velocity),
        "acceleration": compute_rmse(acceleration),
        "affected": compute_rmse(np.vstack([position[node] for node in affected])),
        "remaining": compute_rmse(np.vstack([position[node] for node in remaining])),
        "p95": float(np.percentile(np.linalg.norm(pooled, axis=1), 95)),
        "worst_node": max(by_node.values()),
    }


def _increase(normal, degraded):
    if normal <= 0:
        raise ValueError("Reference RMSE must be positive.")
    return float(100 * (degraded - normal) / normal)
