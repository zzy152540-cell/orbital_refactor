"""Run the 15:15 finite-window three-observer failure prescan."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass
import json
from pathlib import Path

import numpy as np

from experiments.bilateral_multitarget_frontend import (
    BilateralFrontendConfig,
    build_bilateral_raw_frontend,
    derive_bilateral_fault_frontend,
    run_bilateral_cann_acceptance_frontend,
    run_bilateral_distributed_online_comparison,
)
from experiments.run_bilateral_visualization_recording import (
    write_bilateral_visualization_recording,
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
        "--local-unified-multimodal-update", action="store_true",
        help=(
            "Use the adjusted same-prior radar-plus-LOS local estimator before "
            "fixed-topology consensus."
        ),
    )
    parser.add_argument("--backup-radial-range-sigma-m", type=float)
    parser.add_argument("--backup-radial-range-rate-sigma-mps", type=float)
    parser.add_argument(
        "--independent-radial-backup-channel", action="store_true",
        help="Separate the numerical radial backup from its legacy IR carrier.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/external_acceptance/bilateral_node_failure_prescan.json"
        ),
    )
    parser.add_argument(
        "--control-input", type=Path,
        help="Reuse a fingerprint-matched nominal acceptance control artifact.",
    )
    parser.add_argument(
        "--acceptance-cann-only", action="store_true",
        help="Run only the cooperative CANN arm used by the acceptance metric.",
    )
    parser.add_argument("--recording-output", type=Path)
    args = parser.parse_args()
    if args.recording_output is not None and not args.acceptance_cann_only:
        parser.error("--recording-output requires --acceptance-cann-only")
    shared = dict(
        duration_seconds=args.duration,
        sample_interval_seconds=1.0,
        random_seed=args.seed,
        retain_raw_frames=args.recording_output is not None,
        observer_failure_start_seconds=args.failure_start,
        observer_failure_end_seconds=args.failure_end,
        use_local_unified_multimodal_measurement_update=(
            args.local_unified_multimodal_update
        ),
        backup_radial_range_sigma_m=args.backup_radial_range_sigma_m,
        backup_radial_range_rate_sigma_mps=(
            args.backup_radial_range_rate_sigma_mps
        ),
        independent_radial_backup_channel=(
            args.independent_radial_backup_channel
        ),
    )
    control_config = BilateralFrontendConfig(
        **shared, failed_observer_indices=(),
    )
    failure_config = BilateralFrontendConfig(
        **shared, failed_observer_indices=tuple(args.failed_observer_indices),
    )
    control_frontend = build_bilateral_raw_frontend(control_config)
    failure_frontend = derive_bilateral_fault_frontend(
        control_frontend, failure_config,
    )
    iod_initialization_cache = {}
    control = None
    control_provenance = {"reused": False, "source": None}
    if args.control_input is not None:
        from experiments.run_bilateral_acceptance_control import (
            build_control_contract,
            load_matching_control,
        )

        if (
            args.backup_radial_range_sigma_m is not None
            or args.backup_radial_range_rate_sigma_mps is not None
            or args.independent_radial_backup_channel
            or not args.local_unified_multimodal_update
        ):
            raise ValueError(
                "--control-input requires the frozen nominal R08 configuration."
            )
        expected_contract = build_control_contract(
            duration=args.duration,
            event_start=args.failure_start,
            event_end=args.failure_end,
            seed=args.seed,
            family="nominal",
        )
        control_payload = load_matching_control(
            args.control_input, expected_contract,
        )
        control_provenance = {
            "reused": True, "source": str(args.control_input),
            "family": "nominal",
        }
        print(f"CONTROL reused from {args.control_input}")
    else:
        if args.acceptance_cann_only:
            control_arm = run_bilateral_cann_acceptance_frontend(
                control_frontend,
                enable_consensus=True,
                iod_initialization_cache=iod_initialization_cache,
            )
            control_payload = {"incrementalCann": _arm(control_arm)}
        else:
            control = run_bilateral_distributed_online_comparison(
                control_config
            )
            control_payload = {
                "withoutCann": _arm(control.without_cann),
                "incrementalCann": _arm(control.incremental_cann),
            }
    if args.acceptance_cann_only:
        captured = []
        failure_arm = run_bilateral_cann_acceptance_frontend(
            failure_frontend,
            enable_consensus=True,
            iod_initialization_cache=iod_initialization_cache,
            epoch_callback=(captured.append if args.recording_output else None),
        )
        failure_payload = {"incrementalCann": _arm(failure_arm)}
    else:
        failure = run_bilateral_distributed_online_comparison(failure_config)
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
            "localUnifiedMultimodalMeasurementUpdate": (
                args.local_unified_multimodal_update
            ),
            "independentRadialBackup": {
                "enabled": args.backup_radial_range_sigma_m is not None,
                "rangeSigmaM": args.backup_radial_range_sigma_m,
                "rangeRateSigmaMps": args.backup_radial_range_rate_sigma_mps,
                "independentChannel": args.independent_radial_backup_channel,
                "physicalValidation": False,
            },
        },
        "noFailureControl": control_payload,
        "controlProvenance": control_provenance,
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
    if args.recording_output is not None:
        write_bilateral_visualization_recording(
            args.recording_output,
            frontend=failure_frontend,
            captured=captured,
            include_cann=True,
            cann_node_limit=0,
            enable_consensus=True,
            show_communication_topology=True,
            radial_backup_profile="none",
            seed=args.seed,
            source_result_path=args.output,
        )
    print(
        "failedObservers=", failed_ids,
        "controlWindowPositionMeanError=",
        control_payload["incrementalCann"]["phase_metrics"]["DROPOUT"][
            "position_mean_error_m"
        ],
        "failureWindowPositionMeanError=",
        failure_payload["incrementalCann"]["phase_metrics"]["DROPOUT"][
            "position_mean_error_m"
        ],
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
        if arm_name not in failure or arm_name not in control:
            continue
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
    try:
        value = float(value)
        baseline = float(baseline)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or not np.isfinite(baseline) or baseline == 0.0:
        return None
    return 100.0 * (value / baseline - 1.0)


def _jsonable(value):
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
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
