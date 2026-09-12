"""Read-only predicted-information audit for P09 relative updates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path

import numpy as np

from cooperative.network_schmidt_runner import run_network_schmidt_filter
from experiments.relative_measurement_error_projection import (
    relative_update_truth_decomposition,
)
from experiments.v14_walker_geometry_audit import run_v14_walker_geometry_audit
from experiments.walker_filter_setup import build_walker_filter_case


@dataclass(frozen=True)
class InformationGainAuditRecord:
    seed: int
    node_id: str
    neighbor_id: str
    timestamp: float
    modality: str
    predicted_position_covariance_reduction: float
    predicted_position_covariance_reduction_fraction: float
    position_injection_risk: float
    neighbor_to_measurement_covariance_trace_ratio: float
    actual_position_error_change_m: float
    actual_position_error_worsened: bool


@dataclass(frozen=True)
class InformationGainAuditGroup:
    neighbor_id: str
    sample_count: int
    mean_predicted_reduction_fraction: float
    actual_worsening_fraction: float
    mean_actual_position_error_change_m: float


@dataclass(frozen=True)
class InformationGainAuditReport:
    node_id: str
    modality: str
    run_count: int
    sample_count: int
    predicted_reduction_to_actual_change_correlation: float
    injection_risk_to_actual_change_correlation: float
    low_gain_quartile_worsening_fraction: float
    high_gain_quartile_worsening_fraction: float
    records: tuple[InformationGainAuditRecord, ...]
    groups: tuple[InformationGainAuditGroup, ...]


def run_external_cooperative_information_gain_audit(
    *, node_id="sat_p03_s01", modality="RADAR", seeds=(0, 1, 2, 3, 4),
    duration=120.0, dt=2.0, maximum_range=6000e3,
):
    seeds = tuple(int(value) for value in seeds)
    if not seeds:
        raise ValueError("seeds must be nonempty.")
    audit = run_v14_walker_geometry_audit(
        total_satellites=20, plane_count=10, phasing=1,
        duration=1800.0, dt=30.0, maximum_range=maximum_range,
    )
    records = []
    for seed in seeds:
        case = build_walker_filter_case(
            seed=seed, duration=duration, dt=dt, maximum_range=maximum_range,
            topology=audit.persistent_topology,
            truth_history_by_node=audit.scenario.truth_state_history_by_node,
            topology_type="walker_persistent",
        )
        history = run_network_schmidt_filter(
            timestamps=case["timestamps"],
            initial_state_by_node=case["initial_states"],
            initial_covariance_by_node=case["initial_covariances"],
            topology=case["topology"],
            observation_messages=case["observations"],
            absolute_position_observations=case["absolute_observations"],
            observation_usage="observer_only", process_noise_acceleration=1e-8,
            consider_refresh_mode="exact_transport_event_replay",
            state_messages_by_receiver=case["state_messages"],
            replay_history_window=10.0,
            expected_lineage_by_link=case["lineages"],
        )
        decompositions = relative_update_truth_decomposition(
            history=history, truth_by_node=case["truth"]
        )
        for value in decompositions:
            if value.node_id != node_id or value.modalities != (modality,):
                continue
            change = (
                value.active_position_error_norm_after
                - value.active_position_error_norm_before
            )
            records.append(InformationGainAuditRecord(
                seed=seed, node_id=node_id, neighbor_id=value.neighbor_id,
                timestamp=value.timestamp, modality=modality,
                predicted_position_covariance_reduction=(
                    value.predicted_position_covariance_reduction
                ),
                predicted_position_covariance_reduction_fraction=(
                    value.predicted_position_covariance_reduction_fraction
                ),
                position_injection_risk=value.position_injection_risk,
                neighbor_to_measurement_covariance_trace_ratio=(
                    value.neighbor_to_measurement_covariance_trace_ratio
                ),
                actual_position_error_change_m=change,
                actual_position_error_worsened=bool(change > 0.0),
            ))
    if not records:
        raise RuntimeError("No matching relative updates were recorded.")
    predicted = np.asarray([
        value.predicted_position_covariance_reduction_fraction
        for value in records
    ])
    injected = np.asarray([value.position_injection_risk for value in records])
    actual = np.asarray([value.actual_position_error_change_m for value in records])
    lower, upper = np.quantile(predicted, (0.25, 0.75))
    groups = tuple(_group(records, neighbor) for neighbor in sorted({
        value.neighbor_id for value in records
    }))
    return InformationGainAuditReport(
        node_id=node_id, modality=modality, run_count=len(seeds),
        sample_count=len(records),
        predicted_reduction_to_actual_change_correlation=_correlation(
            predicted, actual
        ),
        injection_risk_to_actual_change_correlation=_correlation(injected, actual),
        low_gain_quartile_worsening_fraction=float(np.mean(
            actual[predicted <= lower] > 0.0
        )),
        high_gain_quartile_worsening_fraction=float(np.mean(
            actual[predicted >= upper] > 0.0
        )),
        records=tuple(records), groups=groups,
    )


def save_external_cooperative_information_gain_audit(report, output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = output / "summary.json", output / "records.csv"
    json_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=tuple(InformationGainAuditRecord.__dataclass_fields__)
        )
        writer.writeheader()
        for record in report.records:
            writer.writerow(asdict(record))
    return json_path, csv_path


def _group(records, neighbor_id):
    selected = [value for value in records if value.neighbor_id == neighbor_id]
    changes = np.asarray([value.actual_position_error_change_m for value in selected])
    return InformationGainAuditGroup(
        neighbor_id=neighbor_id, sample_count=len(selected),
        mean_predicted_reduction_fraction=float(np.mean([
            value.predicted_position_covariance_reduction_fraction
            for value in selected
        ])),
        actual_worsening_fraction=float(np.mean(changes > 0.0)),
        mean_actual_position_error_change_m=float(np.mean(changes)),
    )


def _correlation(left, right):
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])
