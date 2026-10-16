"""Run the 15:15 finite-window three-observer failure prescan."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, fields
import json
from pathlib import Path

import numpy as np

from experiments.bilateral_multitarget_frontend import (
    BilateralFrontendConfig,
    build_bilateral_raw_frontend,
    run_bilateral_distributed_online_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--failure-start", type=float, default=6.0)
    parser.add_argument("--failure-end", type=float, default=13.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--failed-observer-indices", nargs="+", type=int, default=(0, 5, 10)
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/external_acceptance/bilateral_node_failure_prescan.json"
        ),
    )
    args = parser.parse_args()
    shared = dict(
        duration_seconds=args.duration,
        sample_interval_seconds=1.0,
        random_seed=args.seed,
        observer_failure_start_seconds=args.failure_start,
        observer_failure_end_seconds=args.failure_end,
    )
    control_config = BilateralFrontendConfig(
        **shared, failed_observer_indices=(),
    )
    failure_config = BilateralFrontendConfig(
        **shared, failed_observer_indices=tuple(args.failed_observer_indices),
    )
    control_frontend = build_bilateral_raw_frontend(control_config)
    failure_frontend = build_bilateral_raw_frontend(failure_config)
    control = run_bilateral_distributed_online_comparison(control_config)
    failure = run_bilateral_distributed_online_comparison(failure_config)
    control_payload = {
        "withoutCann": _arm(control.without_cann),
        "incrementalCann": _arm(control.incremental_cann),
    }
    failure_payload = {
        "withoutCann": _arm(failure.without_cann),
        "incrementalCann": _arm(failure.incremental_cann),
    }
    failed_ids = tuple(
        failure_frontend.truth.manifest.friendly_observer_ids[index]
        for index in args.failed_observer_indices
    )
    visibility_loss = _visibility_loss_by_target(
        control_frontend,
        failure_frontend,
        start=args.failure_start,
        end=args.failure_end,
    )
    payload = {
        "schemaVersion": "bilateral-node-failure-prescan.v1",
        "scope": "OBSERVATION_AND_FIXED_TOPOLOGY_COMMUNICATION_NODE_FAILURE",
        "scenario": {
            "friendlyObserverCount": 15,
            "hostileTargetCount": 15,
            "durationSeconds": args.duration,
            "sampleIntervalSeconds": 1.0,
            "failureStartSeconds": args.failure_start,
            "failureEndSeconds": args.failure_end,
            "failedObserverIndices": list(args.failed_observer_indices),
            "failedObserverIds": list(failed_ids),
            "seed": args.seed,
        },
        "noFailureControl": control_payload,
        "threeObserverFailure": failure_payload,
        "failureVsControlPercent": _failure_vs_control(
            failure_payload, control_payload
        ),
        "visibilityLossByTarget": visibility_loss,
        "timingNote": (
            "processing metrics cover local full-frame estimation and are not "
            "the P09 estimate-message synchronization latency"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "failedObservers=", failed_ids,
        "controlWindowPositionMeanError=",
        control.incremental_cann.phase_metrics["DROPOUT"].position_mean_error_m,
        "failureWindowPositionMeanError=",
        failure.incremental_cann.phase_metrics["DROPOUT"].position_mean_error_m,
    )
    print(f"wrote {args.output}")


def _arm(value):
    payload = {
        item.name: getattr(value, item.name)
        for item in fields(value)
        if item.name not in {
            "phase_metrics",
            "consensus_phase_metrics",
            "representative_node_metrics",
            "independent_node_metrics",
        }
    }
    payload["phase_metrics"] = {
        name: asdict(metrics) for name, metrics in value.phase_metrics.items()
    }
    payload["consensus_phase_metrics"] = {
        name: asdict(metrics)
        for name, metrics in value.consensus_phase_metrics.items()
    }
    payload["representative_node_metrics"] = _node_view(
        value.representative_node_metrics
    )
    payload["independent_node_metrics"] = _node_view(
        value.independent_node_metrics
    )
    return _jsonable(payload)


def _node_view(value):
    payload = {
        item.name: getattr(value, item.name)
        for item in fields(value)
        if item.name != "phase_metrics"
    }
    payload["phase_metrics"] = {
        name: asdict(metrics) for name, metrics in value.phase_metrics.items()
    }
    return payload


def _failure_vs_control(failure, control):
    result = {}
    for arm_name in ("withoutCann", "incrementalCann"):
        affected = failure[arm_name]["phase_metrics"]["DROPOUT"]
        reference = control[arm_name]["phase_metrics"]["DROPOUT"]
        result[arm_name] = {
            metric: _percent_change(affected[metric], reference[metric])
            for metric in (
                "position_mean_error_m",
                "velocity_mean_error_mps",
                "acceleration_mean_error_mps2",
                "position_rmse_m",
                "velocity_rmse_mps",
                "acceleration_rmse_mps2",
            )
        }
    return result


def _visibility_loss_by_target(control, failure, *, start, end):
    result = {}
    for target_id in control.truth.manifest.hostile_target_ids:
        control_count = 0
        failure_count = 0
        for timestamp in control.truth.timestamps:
            epoch = float(timestamp)
            if not start <= epoch <= end:
                continue
            control_count += len(
                control.visible_observers_by_target_and_timestamp[
                    (target_id, epoch)
                ]
            )
            failure_count += len(
                failure.visible_observers_by_target_and_timestamp[
                    (target_id, epoch)
                ]
            )
        result[target_id] = {
            "controlVisibleObserverEpochs": control_count,
            "failureVisibleObserverEpochs": failure_count,
            "lostVisibleObserverEpochs": control_count - failure_count,
        }
    return dict(sorted(result.items()))


def _percent_change(value, baseline):
    if not np.isfinite(value) or not np.isfinite(baseline) or baseline == 0.0:
        return None
    return 100.0 * (value / baseline - 1.0)


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    main()
