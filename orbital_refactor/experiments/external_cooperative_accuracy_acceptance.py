"""P09 paired Walker-20 independent/cooperative accuracy acceptance test."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from cooperative.network_schmidt_runner import run_network_schmidt_filter
from experiments.network_filter_metrics import modality_from_information_id
from experiments.v14_walker_geometry_audit import run_v14_walker_geometry_audit
from experiments.walker_filter_setup import build_walker_filter_case
from orbital_core.metrics import compute_rmse


@dataclass(frozen=True)
class CooperativeAccuracyRecord:
    seed: int
    independent_position_rmse_m: float
    cooperative_position_rmse_m: float
    cooperative_improvement_percent: float
    independent_worst_node_position_rmse_m: float
    cooperative_worst_node_position_rmse_m: float
    independent_run_seconds: float
    cooperative_run_seconds: float


@dataclass(frozen=True)
class CooperativeNodeRecord:
    seed: int
    node_id: str
    independent_position_rmse_m: float
    cooperative_position_rmse_m: float
    cooperative_improvement_percent: float
    topology_degree: int
    radar_observation_count: int
    infrared_observation_count: int
    optical_observation_count: int
    mean_relative_nis: float
    mean_radar_nis: float
    mean_infrared_nis: float
    mean_optical_nis: float
    mean_neighbor_range_m: float
    mean_neighbor_los_angle_deg: float
    independent_radial_rmse_m: float
    independent_along_track_rmse_m: float
    independent_cross_track_rmse_m: float
    cooperative_radial_rmse_m: float
    cooperative_along_track_rmse_m: float
    cooperative_cross_track_rmse_m: float
    initial_segment_improvement_percent: float
    steady_segment_improvement_percent: float


@dataclass(frozen=True)
class CooperativeAccuracyReport:
    walker_definition: tuple[int, int, int]
    duration_seconds: float
    dt_seconds: float
    run_count: int
    threshold_percent: float
    formal_minimum_run_count: int
    formal_minimum_duration_seconds: float
    mean_independent_position_rmse_m: float
    mean_cooperative_position_rmse_m: float
    paired_mean_improvement_percent: float
    paired_improvement_95_half_width_percent: float
    worst_paired_improvement_percent: float
    threshold_met: bool
    formal_sample_size_met: bool
    formal_duration_met: bool
    passed: bool
    records: tuple[CooperativeAccuracyRecord, ...]
    node_records: tuple[CooperativeNodeRecord, ...]


def run_external_cooperative_accuracy_acceptance(
    *, seeds=(0, 1, 2, 3, 4), duration=120.0, dt=2.0,
    maximum_range=6000e3, threshold_percent=5.0,
    formal_minimum_run_count=20, formal_minimum_duration_seconds=1200.0,
):
    """Compare identical Walker cases with and without cooperative inputs.

    The independent arm retains each node's initial condition, dynamics and
    absolute-navigation observations, but receives neither inter-satellite
    observations nor neighbor state messages.  The cooperative arm additionally
    enables the production three-modal Schmidt/replay chain.
    """

    seeds = tuple(int(value) for value in seeds)
    if not seeds:
        raise ValueError("seeds must be nonempty.")
    audit = run_v14_walker_geometry_audit(
        total_satellites=20, plane_count=10, phasing=1,
        duration=1800.0, dt=30.0, maximum_range=maximum_range,
    )
    if audit.persistent_component_sizes != (20,):
        raise ValueError("Walker 20/10/1 persistent topology is not connected.")

    records = []
    node_records = []
    for seed in seeds:
        case = build_walker_filter_case(
            seed=seed, duration=duration, dt=dt, maximum_range=maximum_range,
            topology=audit.persistent_topology,
            truth_history_by_node=audit.scenario.truth_state_history_by_node,
            topology_type="walker_persistent",
        )
        independent_started = perf_counter()
        independent = run_network_schmidt_filter(
            timestamps=case["timestamps"],
            initial_state_by_node=case["initial_states"],
            initial_covariance_by_node=case["initial_covariances"],
            topology=case["topology"], observation_messages=(),
            absolute_position_observations=case["absolute_observations"],
            process_noise_acceleration=1e-8,
            consider_refresh_mode="propagate_only",
        )
        independent_seconds = perf_counter() - independent_started

        cooperative_started = perf_counter()
        cooperative = run_network_schmidt_filter(
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
        cooperative_seconds = perf_counter() - cooperative_started
        independent_rmse, independent_by_node = _fleet_position_metrics(
            independent, case["truth"]
        )
        cooperative_rmse, cooperative_by_node = _fleet_position_metrics(
            cooperative, case["truth"]
        )
        observation_counts = _observation_counts(case["observations"])
        mean_nis = _mean_relative_nis_by_node(cooperative)
        geometry = _neighbor_geometry(case["truth"], case["topology"])
        for node_id in independent.node_ids:
            independent_node_rmse = independent_by_node[node_id]
            cooperative_node_rmse = cooperative_by_node[node_id]
            independent_rtn = _rtn_position_rmse(
                independent.active_state_history_by_node[node_id],
                case["truth"][node_id],
            )
            cooperative_rtn = _rtn_position_rmse(
                cooperative.active_state_history_by_node[node_id],
                case["truth"][node_id],
            )
            split = max(2, len(case["timestamps"]) // 3)
            initial_improvement = _segment_improvement(
                independent.active_state_history_by_node[node_id],
                cooperative.active_state_history_by_node[node_id],
                case["truth"][node_id], slice(0, split),
            )
            steady_improvement = _segment_improvement(
                independent.active_state_history_by_node[node_id],
                cooperative.active_state_history_by_node[node_id],
                case["truth"][node_id], slice(split, None),
            )
            node_records.append(CooperativeNodeRecord(
                seed=seed, node_id=node_id,
                independent_position_rmse_m=independent_node_rmse,
                cooperative_position_rmse_m=cooperative_node_rmse,
                cooperative_improvement_percent=float(
                    100.0 * (independent_node_rmse - cooperative_node_rmse)
                    / independent_node_rmse
                ),
                topology_degree=len(case["topology"].neighbors(node_id)),
                radar_observation_count=observation_counts.get(node_id, {}).get(
                    "RADAR", 0
                ),
                infrared_observation_count=(
                    observation_counts.get(node_id, {}).get("INFRARED", 0)
                ),
                optical_observation_count=observation_counts.get(node_id, {}).get(
                    "OPTICAL", 0
                ),
                mean_relative_nis=mean_nis[node_id]["ALL"],
                mean_radar_nis=mean_nis[node_id]["RADAR"],
                mean_infrared_nis=mean_nis[node_id]["INFRARED"],
                mean_optical_nis=mean_nis[node_id]["OPTICAL"],
                mean_neighbor_range_m=geometry[node_id][0],
                mean_neighbor_los_angle_deg=geometry[node_id][1],
                independent_radial_rmse_m=independent_rtn[0],
                independent_along_track_rmse_m=independent_rtn[1],
                independent_cross_track_rmse_m=independent_rtn[2],
                cooperative_radial_rmse_m=cooperative_rtn[0],
                cooperative_along_track_rmse_m=cooperative_rtn[1],
                cooperative_cross_track_rmse_m=cooperative_rtn[2],
                initial_segment_improvement_percent=initial_improvement,
                steady_segment_improvement_percent=steady_improvement,
            ))
        records.append(CooperativeAccuracyRecord(
            seed=seed,
            independent_position_rmse_m=independent_rmse,
            cooperative_position_rmse_m=cooperative_rmse,
            cooperative_improvement_percent=float(
                100.0 * (independent_rmse - cooperative_rmse)
                / independent_rmse
            ),
            independent_worst_node_position_rmse_m=max(independent_by_node.values()),
            cooperative_worst_node_position_rmse_m=max(cooperative_by_node.values()),
            independent_run_seconds=independent_seconds,
            cooperative_run_seconds=cooperative_seconds,
        ))

    improvements = np.asarray([
        record.cooperative_improvement_percent for record in records
    ])
    half_width = (
        1.96 * float(np.std(improvements, ddof=1)) / np.sqrt(improvements.size)
        if improvements.size > 1 else float("nan")
    )
    threshold_met = bool(float(np.mean(improvements)) >= threshold_percent)
    sample_met = len(records) >= formal_minimum_run_count
    duration_met = duration >= formal_minimum_duration_seconds
    return CooperativeAccuracyReport(
        walker_definition=(20, 10, 1), duration_seconds=float(duration),
        dt_seconds=float(dt), run_count=len(records),
        threshold_percent=float(threshold_percent),
        formal_minimum_run_count=int(formal_minimum_run_count),
        formal_minimum_duration_seconds=float(formal_minimum_duration_seconds),
        mean_independent_position_rmse_m=float(np.mean([
            record.independent_position_rmse_m for record in records
        ])),
        mean_cooperative_position_rmse_m=float(np.mean([
            record.cooperative_position_rmse_m for record in records
        ])),
        paired_mean_improvement_percent=float(np.mean(improvements)),
        paired_improvement_95_half_width_percent=half_width,
        worst_paired_improvement_percent=float(np.min(improvements)),
        threshold_met=threshold_met, formal_sample_size_met=sample_met,
        formal_duration_met=bool(duration_met),
        passed=bool(threshold_met and sample_met and duration_met),
        records=tuple(records),
        node_records=tuple(node_records),
    )


def save_external_cooperative_accuracy_report(report, output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = output / "summary.json", output / "records.csv"
    node_csv_path = output / "node_records.csv"
    json_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=tuple(CooperativeAccuracyRecord.__dataclass_fields__)
        )
        writer.writeheader()
        for record in report.records:
            writer.writerow(asdict(record))
    with node_csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=tuple(CooperativeNodeRecord.__dataclass_fields__)
        )
        writer.writeheader()
        for record in report.node_records:
            writer.writerow(asdict(record))
    return json_path, csv_path, node_csv_path


def _fleet_position_metrics(history, truth):
    by_node = {
        node: compute_rmse(
            history.active_state_history_by_node[node][:, :3]
            - truth[node][:, :3]
        )
        for node in history.node_ids
    }
    pooled = np.vstack([
        history.active_state_history_by_node[node][:, :3] - truth[node][:, :3]
        for node in history.node_ids
    ])
    return compute_rmse(pooled), by_node


def _observation_counts(observations):
    counts = {}
    for observation in observations:
        node_counts = counts.setdefault(
            observation.observer_id,
            {"RADAR": 0, "INFRARED": 0, "OPTICAL": 0},
        )
        if observation.valid_flag and observation.modality in node_counts:
            node_counts[observation.modality] += 1
    return counts


def _mean_relative_nis_by_node(history):
    result = {}
    for node_id in history.node_ids:
        by_modality = {"RADAR": [], "INFRARED": [], "OPTICAL": []}
        for epoch in history.nis_history_by_node[node_id]:
            for information_id, value in epoch.items():
                if ":absolute:" in information_id:
                    continue
                modality = modality_from_information_id(information_id)
                if modality in by_modality:
                    by_modality[modality].append(float(value))
        all_values = [value for values in by_modality.values() for value in values]
        result[node_id] = {
            modality: float(np.mean(values)) if values else float("nan")
            for modality, values in by_modality.items()
        }
        result[node_id]["ALL"] = (
            float(np.mean(all_values)) if all_values else float("nan")
        )
    return result


def _rtn_position_rmse(estimate, truth):
    components = []
    for estimated_state, truth_state in zip(estimate, truth):
        radial = truth_state[:3] / np.linalg.norm(truth_state[:3])
        normal = np.cross(truth_state[:3], truth_state[3:])
        normal /= np.linalg.norm(normal)
        transverse = np.cross(normal, radial)
        components.append(
            np.column_stack((radial, transverse, normal)).T
            @ (estimated_state[:3] - truth_state[:3])
        )
    values = np.asarray(components)
    return tuple(float(np.sqrt(np.mean(values[:, index] ** 2))) for index in range(3))


def _neighbor_geometry(truth, topology):
    result = {}
    for node_id in topology.node_ids:
        neighbors = topology.neighbors(node_id)
        ranges, angles = [], []
        for index in range(len(truth[node_id])):
            origin = truth[node_id][index, :3]
            offsets = [truth[neighbor][index, :3] - origin for neighbor in neighbors]
            ranges.extend(float(np.linalg.norm(offset)) for offset in offsets)
            if len(offsets) >= 2:
                for left in range(len(offsets)):
                    for right in range(left + 1, len(offsets)):
                        cosine = np.dot(offsets[left], offsets[right]) / (
                            np.linalg.norm(offsets[left]) * np.linalg.norm(offsets[right])
                        )
                        angles.append(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))
        result[node_id] = (
            float(np.mean(ranges)),
            float(np.mean(angles)) if angles else float("nan"),
        )
    return result


def _segment_improvement(independent, cooperative, truth, selection):
    independent_rmse = compute_rmse(
        independent[selection, :3] - truth[selection, :3]
    )
    cooperative_rmse = compute_rmse(
        cooperative[selection, :3] - truth[selection, :3]
    )
    return float(100.0 * (independent_rmse - cooperative_rmse) / independent_rmse)
