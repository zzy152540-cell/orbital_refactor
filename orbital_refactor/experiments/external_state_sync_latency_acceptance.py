"""P11 wall-clock state synchronization latency acceptance runner."""

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path

import numpy as np

from cooperative.network_schmidt_orchestrator import NetworkSchmidtOrchestrator
from experiments.online_filter_inputs import (
    items_by_timestamp,
    source_updates_from_messages,
)
from experiments.v14_walker_geometry_audit import run_v14_walker_geometry_audit
from experiments.walker_filter_setup import build_walker_filter_case


@dataclass(frozen=True)
class StateSyncLatencyRecord:
    seed: int
    scenario: str
    category: str
    receiver_id: str
    source_id: str
    accepted: bool
    reason: str
    message_timestamp: float
    application_timestamp: float
    simulated_link_delay_seconds: float
    creation_to_applied_ms: float
    receive_to_applied_ms: float
    endpoint_state_max_abs_error: float | None
    endpoint_covariance_max_abs_error: float | None
    endpoint_covariance_relative_fro_error: float | None


@dataclass(frozen=True)
class StateSyncLatencyGroup:
    category: str
    sample_count: int
    mean_creation_to_applied_ms: float
    p95_creation_to_applied_ms: float
    maximum_creation_to_applied_ms: float
    mean_receive_to_applied_ms: float
    mean_simulated_link_delay_seconds: float
    threshold_met: bool | None


@dataclass(frozen=True)
class StateSyncLatencyReport:
    walker_definition: tuple[int, int, int]
    duration_seconds: float
    dt_seconds: float
    threshold_ms: float
    formal_category: str
    timing_clock: str
    threshold_met: bool
    formal_sample_size_met: bool
    formal_duration_met: bool
    protocol_clean: bool
    rejected_message_count_by_scenario: dict[str, int]
    rejection_counts_by_reason: dict[str, int]
    passed: bool
    records: tuple[StateSyncLatencyRecord, ...]
    groups: tuple[StateSyncLatencyGroup, ...]


def run_external_state_sync_latency_acceptance(
    *, seeds=(0, 1, 2, 3, 4), duration=4.0, dt=0.2,
    maximum_range=6000e3, threshold_ms=150.0,
    formal_minimum_seed_count=20,
    formal_minimum_duration_seconds=1200.0,
) -> StateSyncLatencyReport:
    seeds = tuple(map(int, seeds))
    if not seeds or duration <= 0.0 or dt <= 0.0:
        raise ValueError("seeds, duration and dt must be valid and positive.")
    audit = run_v14_walker_geometry_audit(
        total_satellites=20, plane_count=10, phasing=1,
        duration=1800.0, dt=30.0, maximum_range=maximum_range,
    )
    records = []
    rejected_by_scenario = {}
    rejection_counts = {}
    for seed in seeds:
        case = build_walker_filter_case(
            seed=seed, duration=duration, dt=dt,
            maximum_range=maximum_range,
            topology=audit.persistent_topology,
            truth_history_by_node=audit.scenario.truth_state_history_by_node,
            topology_type="walker_persistent",
        )
        for scenario, delay, exercise_resync in (
            ("ordinary_zero_delay", 0.0, False),
            ("delayed_history_replay", 2.0 * dt, False),
            ("link_resume_resynchronization", 0.0, True),
        ):
            scenario_records, scenario_rejections = _run_scenario(
                seed=seed, scenario=scenario, delay=delay,
                exercise_resync=exercise_resync, case=case,
                history_window=max(10.0, 4.0 * dt),
            )
            records.extend(scenario_records)
            rejected_by_scenario[scenario] = (
                rejected_by_scenario.get(scenario, 0)
                + sum(scenario_rejections.values())
            )
            for reason, count in scenario_rejections.items():
                rejection_counts[reason] = (
                    rejection_counts.get(reason, 0) + count
                )
    categories = (
        "ordinary_update", "historical_replay",
        "explicit_resynchronization", "resynchronization_background",
    )
    groups = tuple(
        _summarize(category, records, threshold_ms)
        for category in categories
        if any(item.category == category for item in records)
    )
    ordinary = next(item for item in groups if item.category == "ordinary_update")
    threshold_met = bool(ordinary.threshold_met)
    sample_met = len(seeds) >= int(formal_minimum_seed_count)
    duration_met = duration >= float(formal_minimum_duration_seconds)
    protocol_clean = rejected_by_scenario.get("ordinary_zero_delay", 0) == 0
    return StateSyncLatencyReport(
        walker_definition=(20, 10, 1), duration_seconds=float(duration),
        dt_seconds=float(dt), threshold_ms=float(threshold_ms),
        formal_category="ordinary_update",
        timing_clock="time.perf_counter_monotonic_wall_clock",
        threshold_met=threshold_met, formal_sample_size_met=sample_met,
        formal_duration_met=duration_met, protocol_clean=protocol_clean,
        rejected_message_count_by_scenario=rejected_by_scenario,
        rejection_counts_by_reason=rejection_counts,
        passed=threshold_met and sample_met and duration_met and protocol_clean,
        records=tuple(records), groups=groups,
    )


def _run_scenario(*, seed, scenario, delay, exercise_resync, case, history_window):
    topology = case["topology"]
    nodes = tuple(topology.node_ids)
    source_updates = source_updates_from_messages(
        case["transmitted_messages"], nodes
    )
    observations = items_by_timestamp(case["observations"])
    absolute = items_by_timestamp(case["absolute_observations"])
    orchestrator = NetworkSchmidtOrchestrator(
        initial_state_by_node=case["initial_states"],
        initial_covariance_by_node=case["initial_covariances"],
        topology=topology, initial_timestamp=0.0,
        process_noise_acceleration=1e-8, history_window=history_window,
        communication_delay=delay, random_seed=20290000 + seed,
        resynchronize_on_resume=exercise_resync,
        record_wall_clock_latency=True,
    )
    full_neighbors = {
        node: tuple(topology.neighbors(node)) for node in nodes
    }
    selected_edge = (nodes[0], full_neighbors[nodes[0]][0])
    records = []
    rejection_counts = {}
    for timestamp in case["timestamps"]:
        timestamp = float(timestamp)
        active, version = _active_neighbors(
            full_neighbors, selected_edge=selected_edge,
            timestamp=timestamp, duration=float(case["timestamps"][-1]),
            exercise_resync=exercise_resync,
        )
        result = orchestrator.step(
            timestamp, topology_version=version,
            active_neighbors_by_node=active,
            source_update_by_node={
                node: source_updates[(node, timestamp)] for node in nodes
            },
            observations=observations.get(timestamp, ()),
            absolute_observations=absolute.get(timestamp, ()),
        )
        for reason, count in result.rejection_counts_by_reason.items():
            rejection_counts[reason] = rejection_counts.get(reason, 0) + count
        for item in result.message_diagnostic_records:
            is_resync = bool(item.get("resynchronization_message", False))
            is_historical = float(item["message_timestamp"]) < timestamp
            category = (
                "explicit_resynchronization" if is_resync
                else "historical_replay" if is_historical
                else "resynchronization_background"
                if scenario == "link_resume_resynchronization"
                else "ordinary_update"
            )
            records.append(StateSyncLatencyRecord(
                seed=seed, scenario=scenario, category=category,
                receiver_id=str(item["receiver_id"]),
                source_id=str(item["source_id"]),
                accepted=bool(item["accepted"]),
                reason=str(item["reason"]),
                message_timestamp=float(item["message_timestamp"]),
                application_timestamp=timestamp,
                simulated_link_delay_seconds=float(
                    item["simulated_link_delay_seconds"]
                ),
                creation_to_applied_ms=float(item["creation_to_applied_ms"]),
                receive_to_applied_ms=float(item["receive_to_applied_ms"]),
                endpoint_state_max_abs_error=item.get(
                    "endpoint_state_max_abs_error"
                ),
                endpoint_covariance_max_abs_error=item.get(
                    "endpoint_covariance_max_abs_error"
                ),
                endpoint_covariance_relative_fro_error=item.get(
                    "endpoint_covariance_relative_fro_error"
                ),
            ))
    return records, rejection_counts


def _active_neighbors(
    full_neighbors, *, selected_edge, timestamp, duration, exercise_resync,
):
    if not exercise_resync or duration <= 0.0:
        return full_neighbors, 0
    first, second = selected_edge
    inactive = duration / 3.0 <= timestamp < 2.0 * duration / 3.0
    if not inactive:
        return full_neighbors, 2 if timestamp >= 2.0 * duration / 3.0 else 0
    active = {
        node: tuple(
            neighbor for neighbor in neighbors
            if {node, neighbor} != {first, second}
        )
        for node, neighbors in full_neighbors.items()
    }
    return active, 1


def _summarize(category, records, threshold):
    selected = tuple(
        item for item in records
        if item.category == category and item.accepted
    )
    created = np.asarray([item.creation_to_applied_ms for item in selected])
    received = np.asarray([item.receive_to_applied_ms for item in selected])
    simulated = np.asarray([
        item.simulated_link_delay_seconds for item in selected
    ])
    return StateSyncLatencyGroup(
        category=category, sample_count=len(selected),
        mean_creation_to_applied_ms=float(np.mean(created)),
        p95_creation_to_applied_ms=float(np.percentile(created, 95)),
        maximum_creation_to_applied_ms=float(np.max(created)),
        mean_receive_to_applied_ms=float(np.mean(received)),
        mean_simulated_link_delay_seconds=float(np.mean(simulated)),
        threshold_met=(
            bool(np.mean(created) <= threshold)
            if category == "ordinary_update" else None
        ),
    )


def save_external_state_sync_latency_report(report, output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary = asdict(report)
    summary.pop("records")
    summary["records_file"] = "communication_latency.csv"
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output / "communication_latency.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=tuple(asdict(report.records[0]))
        )
        writer.writeheader()
        writer.writerows(asdict(item) for item in report.records)
    return output
