"""Run the 15:15 finite-window single-modality dropout prescan."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, fields
import json
from pathlib import Path

import numpy as np

from experiments.bilateral_multitarget_frontend import (
    BilateralFrontendConfig,
    run_bilateral_online_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dropout-start", type=float, default=6.0)
    parser.add_argument("--dropout-end", type=float, default=13.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--process-noise-acceleration", type=float, default=1e-4,
    )
    parser.add_argument("--bearing-window-epochs", type=int, default=1)
    parser.add_argument(
        "--disable-multi-observer-bearing", action="store_true",
        help="Diagnostic ablation: retain local LOS filters but disable the joint LOS update.",
    )
    parser.add_argument(
        "--unified-multimodal-update", action="store_true",
        help=(
            "Use the opt-in centralized radar-plus-LOS measurement-level "
            "update instead of node-local modality fusion."
        ),
    )
    parser.add_argument(
        "--inverse-depth-los-update", action="store_true",
        help=(
            "Use the experimental inverse-depth iterated update on LOS-only "
            "epochs. Requires --unified-multimodal-update."
        ),
    )
    parser.add_argument("--unified-window-epochs", type=int, default=1)
    parser.add_argument(
        "--modalities", nargs="+", default=("OPTICAL", "INFRARED", "RADAR")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/external_acceptance/bilateral_dropout_prescan.json"),
    )
    args = parser.parse_args()
    control_result = run_bilateral_online_comparison(BilateralFrontendConfig(
        duration_seconds=args.duration,
        sample_interval_seconds=1.0,
        random_seed=args.seed,
        process_noise_acceleration=args.process_noise_acceleration,
        multi_epoch_bearing_window_epochs=args.bearing_window_epochs,
        use_multi_observer_bearing_update=(
            not args.disable_multi_observer_bearing
        ),
        use_unified_multimodal_measurement_update=(
            args.unified_multimodal_update
        ),
        use_inverse_depth_los_update=args.inverse_depth_los_update,
        unified_multimodal_window_epochs=args.unified_window_epochs,
        dropout_modality="OPTICAL",
        dropout_start_seconds=args.dropout_start,
        dropout_end_seconds=args.dropout_end,
        dropout_target_indices=(),
    ))
    control = {
        "withoutCann": _arm(control_result.without_cann),
        "incrementalCann": _arm(control_result.incremental_cann),
    }
    print(
        "CONTROL tracks=",
        control_result.incremental_cann.final_active_track_count,
        "window_position_rmse_m=",
        control_result.incremental_cann.phase_metrics["DROPOUT"].position_rmse_m,
    )
    records = {}
    for raw_modality in args.modalities:
        modality = str(raw_modality).upper()
        result = run_bilateral_online_comparison(BilateralFrontendConfig(
            duration_seconds=args.duration,
            sample_interval_seconds=1.0,
            random_seed=args.seed,
            process_noise_acceleration=args.process_noise_acceleration,
            multi_epoch_bearing_window_epochs=args.bearing_window_epochs,
            use_multi_observer_bearing_update=(
                not args.disable_multi_observer_bearing
            ),
            use_unified_multimodal_measurement_update=(
                args.unified_multimodal_update
            ),
            use_inverse_depth_los_update=args.inverse_depth_los_update,
            unified_multimodal_window_epochs=args.unified_window_epochs,
            dropout_modality=modality,
            dropout_start_seconds=args.dropout_start,
            dropout_end_seconds=args.dropout_end,
        ))
        record = {
            "withoutCann": _arm(result.without_cann),
            "incrementalCann": _arm(result.incremental_cann),
            "messageCountByTimestamp": {
                str(timestamp): count
                for timestamp, count in result.message_count_by_timestamp.items()
            },
        }
        record["pairedCannDeltaPercent"] = _paired_cann_delta(record)
        record["dropoutVsControlPercent"] = _dropout_vs_control(
            record, control
        )
        records[modality] = record
        print(
            modality,
            "tracks=",
            result.incremental_cann.final_active_track_count,
            "dropout_position_rmse_m=",
            result.incremental_cann.phase_metrics["DROPOUT"].position_rmse_m,
        )
    payload = {
        "schemaVersion": "bilateral-dropout-prescan.v1",
        "scenario": {
            "friendlyObserverCount": 15,
            "hostileTargetCount": 15,
            "durationSeconds": args.duration,
            "sampleIntervalSeconds": 1.0,
            "dropoutStartSeconds": args.dropout_start,
            "dropoutEndSeconds": args.dropout_end,
            "dropoutTargets": "ALL_15_HOSTILE_TARGETS",
            "seed": args.seed,
            "processNoiseAcceleration": args.process_noise_acceleration,
            "bearingWindowEpochs": args.bearing_window_epochs,
            "multiObserverBearingEnabled": (
                not args.disable_multi_observer_bearing
            ),
            "unifiedMultimodalMeasurementUpdate": (
                args.unified_multimodal_update
            ),
            "inverseDepthLosUpdate": args.inverse_depth_los_update,
            "unifiedMultimodalWindowEpochs": args.unified_window_epochs,
        },
        "noDropoutControl": control,
        "resultsByMissingModality": records,
        "timingNote": (
            "processing metrics cover local full-frame estimation and are not "
            "the P09 estimate-message synchronization latency"
        ),
        "accelerationNote": (
            "prescan acceleration is derived by finite differencing matched "
            "estimated velocities; it is not an augmented filter state"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
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
    payload["dropoutIncreasePercent"] = _dropout_increase(payload)
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


def _dropout_increase(arm):
    before = arm["phase_metrics"]["PRE_DROPOUT"]
    dropout = arm["phase_metrics"]["DROPOUT"]
    return {
        metric: _percent_change(dropout[metric], before[metric])
        for metric in (
            "position_rmse_m",
            "velocity_rmse_mps",
            "acceleration_rmse_mps2",
        )
    }


def _paired_cann_delta(record):
    without = record["withoutCann"]["phase_metrics"]["DROPOUT"]
    with_cann = record["incrementalCann"]["phase_metrics"]["DROPOUT"]
    return {
        metric: _percent_change(with_cann[metric], without[metric])
        for metric in (
            "position_rmse_m",
            "velocity_rmse_mps",
            "acceleration_rmse_mps2",
        )
    }


def _dropout_vs_control(record, control):
    result = {}
    for arm_name in ("withoutCann", "incrementalCann"):
        dropout = record[arm_name]["phase_metrics"]["DROPOUT"]
        reference = control[arm_name]["phase_metrics"]["DROPOUT"]
        result[arm_name] = {
            metric: _percent_change(dropout[metric], reference[metric])
            for metric in (
                "position_rmse_m",
                "velocity_rmse_mps",
                "acceleration_rmse_mps2",
            )
        }
    return result


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
