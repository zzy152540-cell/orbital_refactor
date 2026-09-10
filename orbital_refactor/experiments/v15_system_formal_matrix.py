from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.v14_online_topology_resynchronization import (
    run_v14_online_topology_resynchronization_experiment,
)


@dataclass(frozen=True)
class FormalMatrixRun:
    condition: str
    node_count: int
    seed: int
    passed: bool
    runtime_seconds: float
    position_rmse_m: float
    mean_nees: float
    nees_95_coverage: float
    minimum_covariance_eigenvalue: float
    resynchronization_count: int
    protocol_rejection_count: int
    dropped_message_count: int
    absolute_observation_count: int
    suppressed_absolute_observation_count: int
    target_node_position_rmse_m: float
    target_pre_dropout_rmse_m: float | None
    target_dropout_rmse_m: float | None
    target_post_recovery_rmse_m: float | None
    navigation_dropout_node_count: int
    affected_nodes_position_rmse_m: float | None
    affected_nodes_dropout_rmse_m: float | None
    affected_nodes_post_recovery_rmse_m: float | None
    failure_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class FormalMatrixGroup:
    condition: str
    node_count: int
    run_count: int
    pass_count: int
    mean_position_rmse_m: float
    position_rmse_95_half_width_m: float
    mean_nees: float
    mean_nees_95_coverage: float
    minimum_covariance_eigenvalue: float
    mean_runtime_seconds: float
    mean_target_node_position_rmse_m: float
    mean_target_dropout_rmse_m: float | None
    mean_target_post_recovery_rmse_m: float | None
    mean_affected_nodes_position_rmse_m: float | None
    mean_affected_nodes_dropout_rmse_m: float | None
    mean_affected_nodes_post_recovery_rmse_m: float | None


@dataclass(frozen=True)
class FormalMatrixReport:
    passed: bool
    duration_seconds: float
    dt_seconds: float
    runs: tuple[FormalMatrixRun, ...]
    groups: tuple[FormalMatrixGroup, ...]


def run_v15_system_formal_matrix(
    *, node_counts=(5, 10, 20), seed_values=range(3),
    conditions=(
        "normal", "link_outage_recovery", "packet_loss", "communication_delay",
        "absolute_navigation_dropout",
        "multi_node_navigation_dropout",
        "navigation_communication_degradation",
    ),
    duration=20.0, dt=2.0,
):
    nodes = tuple(int(value) for value in node_counts)
    seeds = tuple(int(value) for value in seed_values)
    cases = tuple(conditions)
    if not nodes or any(value < 3 for value in nodes):
        raise ValueError("node_counts must contain values of at least three.")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seed_values must be nonempty and unique.")
    unknown = set(cases) - {
        "normal", "link_outage_recovery", "packet_loss", "communication_delay",
        "absolute_navigation_dropout",
        "multi_node_navigation_dropout",
        "multi_node_navigation_reference",
        "navigation_communication_degradation",
    }
    if not cases or unknown:
        raise ValueError("Unsupported formal-matrix condition.")
    runs = tuple(
        _run_case(condition, node_count, seed, duration=duration, dt=dt)
        for condition in cases
        for node_count in nodes
        for seed in seeds
    )
    groups = tuple(
        _summarize(condition, node_count, runs)
        for condition in cases for node_count in nodes
    )
    return FormalMatrixReport(
        passed=all(run.passed for run in runs),
        duration_seconds=float(duration), dt_seconds=float(dt),
        runs=runs, groups=groups,
    )


def save_formal_matrix_report(report, *, json_path, csv_path):
    json_target = Path(json_path)
    csv_target = Path(csv_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with csv_target.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = tuple(FormalMatrixRun.__dataclass_fields__)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in report.runs:
            row = asdict(run)
            row["failure_reasons"] = "; ".join(run.failure_reasons)
            writer.writerow(row)
    return json_target, csv_target


def _run_case(condition, node_count, seed, *, duration, dt):
    inactive = (
        None if condition == "link_outage_recovery" else {}
    )
    communication_degraded = (
        condition == "navigation_communication_degradation"
    )
    packet_loss_rate = (
        0.20 if condition == "packet_loss" or communication_degraded else 0.0
    )
    communication_delay = (
        dt if condition == "communication_delay" or communication_degraded
        else 0.0
    )
    max_pinned_age = (
        max(dt, 0.2 * duration)
        if condition == "link_outage_recovery"
        else max(6.0 * dt, 0.2 * duration)
        if communication_degraded
        else max(4.0 * dt, 0.2 * duration)
    )
    start = max(dt, 0.25 * duration)
    end = min(duration - dt, 0.75 * duration)
    dropout_nodes = (
        ("sat_01", "sat_02")
        if condition in {
            "multi_node_navigation_dropout",
            "navigation_communication_degradation",
        }
        else ("sat_01",)
        if condition == "absolute_navigation_dropout"
        else ()
    )
    evaluation_nodes = (
        ("sat_01", "sat_02")
        if condition in {
            "multi_node_navigation_dropout",
            "multi_node_navigation_reference",
            "navigation_communication_degradation",
        }
        else dropout_nodes or ("sat_01",)
    )
    navigation_dropouts = (
        {node: ((start, end),) for node in dropout_nodes}
        if dropout_nodes else None
    )
    metric_phase_windows = {
        node: ((start, end),)
        for node in evaluation_nodes
    }
    started = perf_counter()
    result = run_v14_online_topology_resynchronization_experiment(
        seed_values=(seed,), duration=duration, dt=dt,
        inactive_window=(start, end),
        topology_inactive_windows_by_undirected_edge=inactive,
        max_pinned_age=max_pinned_age,
        topology_type="ring", node_count=node_count,
        packet_loss_rate=packet_loss_rate,
        communication_delay=communication_delay,
        absolute_navigation_dropout_windows_by_node=navigation_dropouts,
        metric_phase_windows_by_node=metric_phase_windows,
    )
    failures = []
    finite_values = (
        result.mean_position_rmse, result.mean_nees,
        result.mean_nees_95_coverage,
        result.minimum_joint_covariance_eigenvalue,
    )
    if not np.all(np.isfinite(finite_values)):
        failures.append("non-finite estimation metric")
    if result.minimum_joint_covariance_eigenvalue < -1e-8:
        failures.append("joint covariance lost positive semidefiniteness")
    if result.protocol_rejected_message_count:
        failures.append("protocol rejected a message")
    if condition == "link_outage_recovery" and not result.resynchronization_count:
        failures.append("link restoration did not resynchronize")
    if condition == "normal" and result.resynchronization_count:
        failures.append("normal condition unexpectedly resynchronized")
    if (
        condition in {"packet_loss", "navigation_communication_degradation"}
        and not result.dropped_message_count
    ):
        failures.append("packet-loss condition dropped no messages")
    if (
        condition == "communication_delay"
        and result.protocol_rejected_message_count
        and not result.resynchronization_count
    ):
        failures.append("delayed protocol gap was not resynchronized")
    epoch_count = int(round(duration / dt)) + 1
    expected_absolute_count = node_count * epoch_count
    suppressed_absolute_count = (
        expected_absolute_count - result.absolute_observation_count
    )
    if (
        condition in {
            "absolute_navigation_dropout", "multi_node_navigation_dropout",
            "navigation_communication_degradation",
        }
        and suppressed_absolute_count <= 0
    ):
        failures.append("absolute-navigation dropout suppressed no observations")
    return FormalMatrixRun(
        condition=condition, node_count=node_count, seed=seed,
        passed=not failures, runtime_seconds=perf_counter() - started,
        position_rmse_m=result.mean_position_rmse,
        mean_nees=result.mean_nees,
        nees_95_coverage=result.mean_nees_95_coverage,
        minimum_covariance_eigenvalue=(
            result.minimum_joint_covariance_eigenvalue
        ),
        resynchronization_count=result.resynchronization_count,
        protocol_rejection_count=result.protocol_rejected_message_count,
        dropped_message_count=result.dropped_message_count,
        absolute_observation_count=result.absolute_observation_count,
        suppressed_absolute_observation_count=suppressed_absolute_count,
        target_node_position_rmse_m=(
            result.mean_position_rmse_by_node["sat_01"]
        ),
        target_pre_dropout_rmse_m=_phase_metric(
            result, "pre_dropout"
        ),
        target_dropout_rmse_m=_phase_metric(result, "dropout"),
        target_post_recovery_rmse_m=_phase_metric(
            result, "post_recovery"
        ),
        navigation_dropout_node_count=len(dropout_nodes),
        affected_nodes_position_rmse_m=_node_mean(
            result.mean_position_rmse_by_node, evaluation_nodes
        ),
        affected_nodes_dropout_rmse_m=_phase_node_mean(
            result, evaluation_nodes, "dropout"
        ),
        affected_nodes_post_recovery_rmse_m=_phase_node_mean(
            result, evaluation_nodes, "post_recovery"
        ),
        failure_reasons=tuple(failures),
    )


def _summarize(condition, node_count, runs):
    selected = tuple(
        run for run in runs
        if run.condition == condition and run.node_count == node_count
    )
    rmse = np.asarray([run.position_rmse_m for run in selected])
    half_width = (
        1.96 * float(np.std(rmse, ddof=1)) / np.sqrt(len(rmse))
        if len(rmse) > 1 else 0.0
    )
    return FormalMatrixGroup(
        condition=condition, node_count=node_count, run_count=len(selected),
        pass_count=sum(run.passed for run in selected),
        mean_position_rmse_m=float(np.mean(rmse)),
        position_rmse_95_half_width_m=half_width,
        mean_nees=float(np.mean([run.mean_nees for run in selected])),
        mean_nees_95_coverage=float(np.mean([
            run.nees_95_coverage for run in selected
        ])),
        minimum_covariance_eigenvalue=min(
            run.minimum_covariance_eigenvalue for run in selected
        ),
        mean_runtime_seconds=float(np.mean([
            run.runtime_seconds for run in selected
        ])),
        mean_target_node_position_rmse_m=float(np.mean([
            run.target_node_position_rmse_m for run in selected
        ])),
        mean_target_dropout_rmse_m=_optional_mean(
            run.target_dropout_rmse_m for run in selected
        ),
        mean_target_post_recovery_rmse_m=_optional_mean(
            run.target_post_recovery_rmse_m for run in selected
        ),
        mean_affected_nodes_position_rmse_m=_optional_mean(
            run.affected_nodes_position_rmse_m for run in selected
        ),
        mean_affected_nodes_dropout_rmse_m=_optional_mean(
            run.affected_nodes_dropout_rmse_m for run in selected
        ),
        mean_affected_nodes_post_recovery_rmse_m=_optional_mean(
            run.affected_nodes_post_recovery_rmse_m for run in selected
        ),
    )


def _phase_metric(result, phase):
    return result.mean_position_rmse_by_node_and_phase.get(
        "sat_01", {}
    ).get(phase)


def _optional_mean(values):
    available = tuple(value for value in values if value is not None)
    return float(np.mean(available)) if available else None


def _node_mean(values_by_node, nodes):
    return (
        float(np.mean([values_by_node[node] for node in nodes]))
        if nodes else None
    )


def _phase_node_mean(result, nodes, phase):
    values = [
        result.mean_position_rmse_by_node_and_phase[node][phase]
        for node in nodes
        if phase in result.mean_position_rmse_by_node_and_phase.get(node, {})
    ]
    return float(np.mean(values)) if values else None
