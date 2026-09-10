from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from time import perf_counter
import tracemalloc

import numpy as np

from experiments.single_satellite_three_modal_cann_feedback import (
    run_single_satellite_three_modal_cann_feedback,
)
from experiments.v14_online_topology_resynchronization import (
    run_v14_online_topology_resynchronization_experiment,
)
from experiments.v14_three_satellite_local_observation import (
    run_v14_three_satellite_local_observation_experiment,
)
from experiments.v14_walker_filter_baseline import run_v14_walker_filter_smoke


@dataclass(frozen=True)
class SystemAcceptanceRecord:
    case_id: str
    passed: bool
    runtime_seconds: float
    python_peak_memory_mb: float
    metrics: dict[str, float | int | bool | str]
    failure_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class SystemAcceptanceReport:
    profile: str
    passed: bool
    records: tuple[SystemAcceptanceRecord, ...]


SYSTEM_ACCEPTANCE_CASE_IDS = (
    "single_three_modal_cann_boundary",
    "three_satellite_multimodal_visibility",
    "five_satellite_dynamic_resynchronization",
    "ten_satellite_dynamic_resynchronization",
    "walker20_distributed_filter",
)


def run_v15_system_acceptance(*, profile="smoke", case_ids=None):
    """Run non-learning system health checks without changing algorithms."""
    if profile != "smoke":
        raise ValueError("Only the bounded smoke profile is implemented.")
    selected = SYSTEM_ACCEPTANCE_CASE_IDS if case_ids is None else tuple(case_ids)
    unknown = set(selected) - set(SYSTEM_ACCEPTANCE_CASE_IDS)
    if unknown or not selected or len(set(selected)) != len(selected):
        raise ValueError("Acceptance case IDs must be unique supported names.")
    runners = {
        "single_three_modal_cann_boundary": _single_cann_boundary,
        "three_satellite_multimodal_visibility": _three_satellite_multimodal,
        "five_satellite_dynamic_resynchronization": _five_satellite_dynamic,
        "ten_satellite_dynamic_resynchronization": _ten_satellite_dynamic,
        "walker20_distributed_filter": _walker20_filter,
    }
    records = tuple(_timed(case_id, runners[case_id]) for case_id in selected)
    return SystemAcceptanceReport(
        profile=profile, passed=all(record.passed for record in records),
        records=records,
    )


def save_system_acceptance_report(report, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def save_system_acceptance_csv(report, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = tuple(sorted({
        name for record in report.records for name in record.metrics
    }))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "case_id", "passed", "runtime_seconds", "python_peak_memory_mb",
            "failure_reasons", *metric_names,
        ))
        writer.writeheader()
        for record in report.records:
            writer.writerow({
                "case_id": record.case_id,
                "passed": record.passed,
                "runtime_seconds": record.runtime_seconds,
                "python_peak_memory_mb": record.python_peak_memory_mb,
                "failure_reasons": "; ".join(record.failure_reasons),
                **record.metrics,
            })
    return path


def _timed(case_id, runner):
    tracemalloc.start()
    started = perf_counter()
    try:
        metrics, failures = runner()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return SystemAcceptanceRecord(
        case_id=case_id, passed=not failures,
        runtime_seconds=perf_counter() - started,
        python_peak_memory_mb=peak_bytes / (1024.0 ** 2),
        metrics=metrics, failure_reasons=tuple(failures),
    )


def _single_cann_boundary():
    result = run_single_satellite_three_modal_cann_feedback(
        inject_faults=False, duration=20.0, dt=2.0, outage_modalities=(),
    )
    baseline = result["summary"]["baseline"]
    processed = result["summary"]["three_modal_cann"]
    metrics = {
        "baseline_position_rmse_m": float(baseline["position_rmse_m"]),
        "cann_position_rmse_m": float(processed["position_rmse_m"]),
        "position_rmse_change_m": float(
            result["summary"]["position_rmse_change_m"]
        ),
    }
    return metrics, _nonfinite_failures(metrics)


def _three_satellite_multimodal():
    result = run_v14_three_satellite_local_observation_experiment(
        seeds=1, duration=8.0, dt=2.0,
        sensor_modalities=("RADAR", "INFRARED", "OPTICAL"),
    )
    summaries = tuple(result.summary_by_case_and_mode.values())
    metrics = {
        "comparison_count": len(summaries),
        "maximum_position_rmse_m": max(
            item.mean_position_rmse for item in summaries
        ),
        "message_rejection_count": sum(
            item.message_rejection_count for item in summaries
        ),
        "psd_failure_count": sum(item.psd_failure_count for item in summaries),
        "all_three_sensor_suite": all(
            item.full_three_sensor_suite for item in summaries
        ),
    }
    failures = _nonfinite_failures(metrics)
    if metrics["message_rejection_count"]:
        failures.append("unexpected message rejection")
    if metrics["psd_failure_count"]:
        failures.append("non-PSD covariance detected")
    if not metrics["all_three_sensor_suite"]:
        failures.append("three-sensor suite was not transported")
    return metrics, failures


def _five_satellite_dynamic():
    result = run_v14_online_topology_resynchronization_experiment(
        seeds=1, duration=12.0, dt=2.0, inactive_window=(2.0, 8.0),
        max_pinned_age=2.0, topology_type="ring", node_count=5,
    )
    metrics = {
        "mean_position_rmse_m": result.mean_position_rmse,
        "minimum_joint_covariance_eigenvalue": (
            result.minimum_joint_covariance_eigenvalue
        ),
        "resynchronization_count": result.resynchronization_count,
        "protocol_rejected_message_count": (
            result.protocol_rejected_message_count
        ),
        "maximum_local_dimension": result.maximum_local_dimension,
    }
    failures = _nonfinite_failures(metrics)
    if metrics["minimum_joint_covariance_eigenvalue"] < -1e-8:
        failures.append("joint covariance lost positive semidefiniteness")
    if metrics["protocol_rejected_message_count"]:
        failures.append("protocol rejected a message")
    if not metrics["resynchronization_count"]:
        failures.append("dynamic restoration did not resynchronize")
    return metrics, failures


def _ten_satellite_dynamic():
    result = run_v14_online_topology_resynchronization_experiment(
        seeds=1, duration=12.0, dt=2.0, inactive_window=(2.0, 8.0),
        max_pinned_age=2.0, topology_type="ring", node_count=10,
    )
    metrics = {
        "mean_position_rmse_m": result.mean_position_rmse,
        "minimum_joint_covariance_eigenvalue": (
            result.minimum_joint_covariance_eigenvalue
        ),
        "resynchronization_count": result.resynchronization_count,
        "protocol_rejected_message_count": (
            result.protocol_rejected_message_count
        ),
        "maximum_local_dimension": result.maximum_local_dimension,
    }
    failures = _nonfinite_failures(metrics)
    if metrics["minimum_joint_covariance_eigenvalue"] < -1e-8:
        failures.append("joint covariance lost positive semidefiniteness")
    if metrics["protocol_rejected_message_count"]:
        failures.append("protocol rejected a message")
    if not metrics["resynchronization_count"]:
        failures.append("dynamic restoration did not resynchronize")
    return metrics, failures


def _walker20_filter():
    result = run_v14_walker_filter_smoke(seeds=1, duration=4.0, dt=2.0)
    metrics = {
        "node_count": result.node_count,
        "mean_position_rmse_m": result.mean_position_rmse,
        "message_acceptance_rate": result.message_acceptance_rate,
        "message_rejection_count": result.message_rejection_count,
        "psd_failure_count": result.psd_failure_count,
        "minimum_joint_eigenvalue": result.minimum_joint_eigenvalue,
        "mean_run_seconds": result.mean_run_seconds,
    }
    failures = _nonfinite_failures(metrics)
    if result.node_count != 20:
        failures.append("Walker smoke did not contain 20 nodes")
    if result.message_rejection_count:
        failures.append("Walker smoke rejected a message")
    if result.psd_failure_count or result.minimum_joint_eigenvalue < -1e-8:
        failures.append("Walker covariance failed PSD check")
    return metrics, failures


def _nonfinite_failures(metrics):
    failures = []
    for name, value in metrics.items():
        if isinstance(value, float) and not np.isfinite(value):
            failures.append(f"{name} is non-finite")
    return failures
