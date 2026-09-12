"""Targeted edge/modality ablation for P09 local cooperative regressions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path

import numpy as np

from cooperative.network_schmidt_runner import run_network_schmidt_filter
from experiments.v14_walker_geometry_audit import run_v14_walker_geometry_audit
from experiments.walker_filter_setup import build_walker_filter_case
from orbital_core.metrics import compute_rmse


@dataclass(frozen=True)
class CooperativeNodeAblationRecord:
    seed: int
    node_id: str
    variant: str
    retained_neighbors: tuple[str, ...]
    retained_modalities: tuple[str, ...]
    position_rmse_m: float
    steady_position_rmse_m: float
    improvement_over_independent_percent: float
    steady_improvement_over_independent_percent: float


@dataclass(frozen=True)
class CooperativeNodeAblationReport:
    duration_seconds: float
    dt_seconds: float
    node_id: str
    run_count: int
    records: tuple[CooperativeNodeAblationRecord, ...]
    mean_improvement_by_variant: dict[str, float]
    mean_steady_improvement_by_variant: dict[str, float]


def run_external_cooperative_node_ablation(
    *, node_id="sat_p03_s01", seeds=(0, 1, 2, 3, 4),
    duration=120.0, dt=2.0, maximum_range=6000e3,
):
    seeds = tuple(int(value) for value in seeds)
    if not seeds:
        raise ValueError("seeds must be nonempty.")
    audit = run_v14_walker_geometry_audit(
        total_satellites=20, plane_count=10, phasing=1,
        duration=1800.0, dt=30.0, maximum_range=maximum_range,
    )
    if node_id not in audit.persistent_topology.node_ids:
        raise ValueError(f"Unknown Walker node: {node_id}")
    neighbors = audit.persistent_topology.neighbors(node_id)
    variants = {
        "full": (neighbors, ("RADAR", "INFRARED", "OPTICAL")),
        **{
            f"without_{neighbor}": (
                tuple(value for value in neighbors if value != neighbor),
                ("RADAR", "INFRARED", "OPTICAL"),
            )
            for neighbor in neighbors
        },
        "radar_only": (neighbors, ("RADAR",)),
        "infrared_only": (neighbors, ("INFRARED",)),
        "optical_only": (neighbors, ("OPTICAL",)),
    }
    records = []
    for seed in seeds:
        case = build_walker_filter_case(
            seed=seed, duration=duration, dt=dt, maximum_range=maximum_range,
            topology=audit.persistent_topology,
            truth_history_by_node=audit.scenario.truth_state_history_by_node,
            topology_type="walker_persistent",
        )
        independent = _run(case, observations=(), state_messages=None,
                           replay=False)
        truth = case["truth"][node_id]
        split = max(2, len(case["timestamps"]) // 3)
        independent_rmse = _position_rmse(independent, truth, node_id)
        independent_steady = _position_rmse(
            independent, truth, node_id, slice(split, None)
        )
        for name, (retained_neighbors, retained_modalities) in variants.items():
            observations = [
                item for item in case["observations"]
                if item.observer_id != node_id or (
                    item.target_id in retained_neighbors
                    and item.modality in retained_modalities
                )
            ]
            state_messages = dict(case["state_messages"])
            state_messages[node_id] = [
                item for item in state_messages[node_id]
                if item.source_node_id in retained_neighbors
            ]
            history = _run(
                case, observations=observations,
                state_messages=state_messages, replay=True,
            )
            rmse = _position_rmse(history, truth, node_id)
            steady = _position_rmse(
                history, truth, node_id, slice(split, None)
            )
            records.append(CooperativeNodeAblationRecord(
                seed=seed, node_id=node_id, variant=name,
                retained_neighbors=tuple(retained_neighbors),
                retained_modalities=tuple(retained_modalities),
                position_rmse_m=rmse, steady_position_rmse_m=steady,
                improvement_over_independent_percent=float(
                    100.0 * (independent_rmse - rmse) / independent_rmse
                ),
                steady_improvement_over_independent_percent=float(
                    100.0 * (independent_steady - steady) / independent_steady
                ),
            ))
    return CooperativeNodeAblationReport(
        duration_seconds=float(duration), dt_seconds=float(dt),
        node_id=node_id, run_count=len(seeds), records=tuple(records),
        mean_improvement_by_variant=_means(records, "improvement_over_independent_percent"),
        mean_steady_improvement_by_variant=_means(
            records, "steady_improvement_over_independent_percent"
        ),
    )


def save_external_cooperative_node_ablation(report, output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = output / "summary.json", output / "records.csv"
    json_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=tuple(CooperativeNodeAblationRecord.__dataclass_fields__)
        )
        writer.writeheader()
        for record in report.records:
            row = asdict(record)
            row["retained_neighbors"] = "|".join(record.retained_neighbors)
            row["retained_modalities"] = "|".join(record.retained_modalities)
            writer.writerow(row)
    return json_path, csv_path


def _run(case, *, observations, state_messages, replay):
    return run_network_schmidt_filter(
        timestamps=case["timestamps"],
        initial_state_by_node=case["initial_states"],
        initial_covariance_by_node=case["initial_covariances"],
        topology=case["topology"], observation_messages=observations,
        absolute_position_observations=case["absolute_observations"],
        observation_usage="observer_only", process_noise_acceleration=1e-8,
        consider_refresh_mode=(
            "exact_transport_event_replay" if replay else "propagate_only"
        ),
        state_messages_by_receiver=state_messages,
        replay_history_window=10.0 if replay else None,
        expected_lineage_by_link=case["lineages"] if replay else None,
    )


def _position_rmse(history, truth, node_id, selection=slice(None)):
    error = history.active_state_history_by_node[node_id][selection, :3] - (
        truth[selection, :3]
    )
    return compute_rmse(error)


def _means(records, field):
    variants = sorted({record.variant for record in records})
    return {
        variant: float(np.mean([
            getattr(record, field) for record in records
            if record.variant == variant
        ]))
        for variant in variants
    }
