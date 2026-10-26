"""Run the 15:15 finite-window single-modality dropout prescan."""

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
    electro_optical_radial_failover_sensor_config,
    run_bilateral_cann_acceptance_frontend,
    run_bilateral_distributed_online_comparison,
    run_bilateral_online_comparison,
)
from experiments.run_bilateral_visualization_recording import (
    write_bilateral_visualization_recording,
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
        "--local-unified-multimodal-update", action="store_true",
        help=(
            "Use one same-prior radar-plus-LOS update independently at each "
            "observer; this is the adjusted P05/P07 candidate architecture."
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
        "--backup-radial-range-sigma-m", type=float, default=None,
        help=(
            "Enable an explicitly non-physical independent radial-backup "
            "shadow measurement with this per-observer range sigma."
        ),
    )
    parser.add_argument(
        "--backup-radial-range-rate-sigma-mps", type=float, default=None,
        help=(
            "Enable an explicitly non-physical independent radial-backup "
            "shadow measurement with this per-observer range-rate sigma."
        ),
    )
    parser.add_argument(
        "--backup-radial-observer-indices", type=int, nargs="+", default=None,
        help=(
            "Restrict the independent radial-backup shadow to these friendly "
            "observer indices; omit to enable it on every observer."
        ),
    )
    parser.add_argument(
        "--independent-radial-backup-channel", action="store_true",
        help="Separate the numerical radial backup from its legacy IR carrier.",
    )
    parser.add_argument(
        "--electro-optical-radial-failover", action="store_true",
        help=(
            "Use the frozen R06 radar-primary optical/infrared radial-proxy "
            "failover configuration."
        ),
    )
    parser.add_argument(
        "--eo-radial-range-sigma-m", type=float, default=2.5,
    )
    parser.add_argument(
        "--eo-radial-range-rate-sigma-mps", type=float, default=0.010,
    )
    parser.add_argument(
        "--prefer-complete-radial-lineage", action="store_true",
        help=(
            "Experimental consensus gate: when complete radial lineage is "
            "available for a target, exclude pure-LOS reports from CI."
        ),
    )
    parser.add_argument(
        "--distributed-consensus", action="store_true",
        help="Run fixed-topology target-level Consensus-CI with feedback.",
    )
    parser.add_argument(
        "--acceptance-cann-only", action="store_true",
        help="Run only the CANN arm used by the frozen acceptance metric.",
    )
    parser.add_argument(
        "--consensus-exchange-hops", type=int, default=1,
        help=(
            "Synchronous graph-hop bound for each target report; use 7 for "
            "full reachability on the 15-node ring."
        ),
    )
    parser.add_argument(
        "--modalities", nargs="+", default=("OPTICAL", "INFRARED", "RADAR")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/external_acceptance/bilateral_dropout_prescan.json"),
    )
    parser.add_argument(
        "--control-input", type=Path,
        help="Reuse a fingerprint-matched acceptance control artifact.",
    )
    parser.add_argument("--recording-output", type=Path)
    args = parser.parse_args()
    if args.recording_output is not None and (
        not args.acceptance_cann_only or len(args.modalities) != 1
    ):
        parser.error(
            "--recording-output requires --acceptance-cann-only and exactly "
            "one missing modality"
        )
    if args.prefer_complete_radial_lineage and not args.distributed_consensus:
        parser.error(
            "--prefer-complete-radial-lineage requires "
            "--distributed-consensus"
        )
    legacy_radial_requested = (
        args.backup_radial_range_sigma_m is not None
        or args.backup_radial_range_rate_sigma_mps is not None
        or args.independent_radial_backup_channel
    )
    if args.electro_optical_radial_failover and legacy_radial_requested:
        parser.error(
            "--electro-optical-radial-failover cannot be combined with the "
            "legacy independent radial-backup options"
        )
    sensor_config = (
        electro_optical_radial_failover_sensor_config(
            range_sigma_m=args.eo_radial_range_sigma_m,
            range_rate_sigma_mps=args.eo_radial_range_rate_sigma_mps,
        )
        if args.electro_optical_radial_failover else None
    )
    comparison_runner = (
        run_bilateral_distributed_online_comparison
        if args.distributed_consensus
        else run_bilateral_online_comparison
    )
    iod_initialization_cache = {}
    control_frontend = None
    recording_frontend = None
    recording_capture = None
    control_provenance = {"reused": False, "source": None}
    if args.control_input is not None:
        from experiments.run_bilateral_acceptance_control import (
            build_control_contract,
            load_matching_control,
        )

        family = _acceptance_control_family(args)
        expected_contract = build_control_contract(
            duration=args.duration,
            event_start=args.dropout_start,
            event_end=args.dropout_end,
            seed=args.seed,
            family=family,
            distributed_consensus=args.distributed_consensus,
        )
        control = load_matching_control(args.control_input, expected_contract)
        control_provenance = {
            "reused": True, "source": str(args.control_input),
            "family": family,
        }
        print(f"CONTROL reused from {args.control_input}")
    else:
        control_config = BilateralFrontendConfig(
            duration_seconds=args.duration,
            sample_interval_seconds=1.0,
            random_seed=args.seed,
            sensor_config=sensor_config,
            retain_raw_frames=args.recording_output is not None,
            backup_radial_range_sigma_m=args.backup_radial_range_sigma_m,
            backup_radial_range_rate_sigma_mps=(
                args.backup_radial_range_rate_sigma_mps
            ),
            backup_radial_observer_indices=(
                None if args.backup_radial_observer_indices is None
                else tuple(args.backup_radial_observer_indices)
            ),
            independent_radial_backup_channel=(
                args.independent_radial_backup_channel
            ),
            prefer_complete_radial_lineage=args.prefer_complete_radial_lineage,
            consensus_exchange_hops=args.consensus_exchange_hops,
            process_noise_acceleration=args.process_noise_acceleration,
            multi_epoch_bearing_window_epochs=args.bearing_window_epochs,
            use_multi_observer_bearing_update=(
                not args.disable_multi_observer_bearing
            ),
            use_unified_multimodal_measurement_update=(
                args.unified_multimodal_update
            ),
            use_local_unified_multimodal_measurement_update=(
                args.local_unified_multimodal_update
            ),
            use_inverse_depth_los_update=args.inverse_depth_los_update,
            unified_multimodal_window_epochs=args.unified_window_epochs,
            dropout_modality="OPTICAL",
            dropout_start_seconds=args.dropout_start,
            dropout_end_seconds=args.dropout_end,
            dropout_target_indices=(),
        )
        if args.acceptance_cann_only:
            control_frontend = build_bilateral_raw_frontend(control_config)
            control_arm = run_bilateral_cann_acceptance_frontend(
                control_frontend,
                enable_consensus=args.distributed_consensus,
                iod_initialization_cache=iod_initialization_cache,
            )
            control = {"incrementalCann": _arm(control_arm)}
        else:
            control_result = comparison_runner(control_config)
            control = {
                "withoutCann": _arm(control_result.without_cann),
                "incrementalCann": _arm(control_result.incremental_cann),
            }
        print(
            "CONTROL tracks=",
            control["incrementalCann"]["final_active_track_count"],
            "window_position_rmse_m=",
            control["incrementalCann"]["phase_metrics"]["DROPOUT"][
                "position_rmse_m"
            ],
        )
    records = {}
    for raw_modality in args.modalities:
        modality = str(raw_modality).upper()
        experiment_config = BilateralFrontendConfig(
            duration_seconds=args.duration,
            sample_interval_seconds=1.0,
            random_seed=args.seed,
            sensor_config=sensor_config,
            retain_raw_frames=args.recording_output is not None,
            backup_radial_range_sigma_m=args.backup_radial_range_sigma_m,
            backup_radial_range_rate_sigma_mps=(
                args.backup_radial_range_rate_sigma_mps
            ),
            backup_radial_observer_indices=(
                None if args.backup_radial_observer_indices is None
                else tuple(args.backup_radial_observer_indices)
            ),
            independent_radial_backup_channel=(
                args.independent_radial_backup_channel
            ),
            prefer_complete_radial_lineage=(
                args.prefer_complete_radial_lineage
            ),
            consensus_exchange_hops=args.consensus_exchange_hops,
            process_noise_acceleration=args.process_noise_acceleration,
            multi_epoch_bearing_window_epochs=args.bearing_window_epochs,
            use_multi_observer_bearing_update=(
                not args.disable_multi_observer_bearing
            ),
            use_unified_multimodal_measurement_update=(
                args.unified_multimodal_update
            ),
            use_local_unified_multimodal_measurement_update=(
                args.local_unified_multimodal_update
            ),
            use_inverse_depth_los_update=args.inverse_depth_los_update,
            unified_multimodal_window_epochs=args.unified_window_epochs,
            dropout_modality=modality,
            dropout_start_seconds=args.dropout_start,
            dropout_end_seconds=args.dropout_end,
        )
        if args.acceptance_cann_only:
            experiment_frontend = (
                build_bilateral_raw_frontend(experiment_config)
                if control_frontend is None
                else derive_bilateral_fault_frontend(
                    control_frontend, experiment_config,
                )
            )
            captured = []
            experiment_arm = run_bilateral_cann_acceptance_frontend(
                experiment_frontend,
                enable_consensus=args.distributed_consensus,
                iod_initialization_cache=iod_initialization_cache,
                epoch_callback=(
                    captured.append if args.recording_output else None
                ),
            )
            if args.recording_output is not None:
                recording_frontend = experiment_frontend
                recording_capture = captured
            record = {
                "incrementalCann": _arm(experiment_arm),
                "messageCountByTimestamp": {},
            }
        else:
            result = comparison_runner(experiment_config)
            record = {
                "withoutCann": _arm(result.without_cann),
                "incrementalCann": _arm(result.incremental_cann),
                "messageCountByTimestamp": {
                    str(timestamp): count
                    for timestamp, count in (
                        result.message_count_by_timestamp.items()
                    )
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
            record["incrementalCann"]["final_active_track_count"],
            "dropout_position_rmse_m=",
            record["incrementalCann"]["phase_metrics"]["DROPOUT"][
                "position_rmse_m"
            ],
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
            "localUnifiedMultimodalMeasurementUpdate": (
                args.local_unified_multimodal_update
            ),
            "inverseDepthLosUpdate": args.inverse_depth_los_update,
            "unifiedMultimodalWindowEpochs": args.unified_window_epochs,
            "radialBackup": {
                "source": "INDEPENDENT_RADIAL_BACKUP_SHADOW",
                "physicalValidation": False,
                "rangeSigmaM": args.backup_radial_range_sigma_m,
                "rangeRateSigmaMps": (
                    args.backup_radial_range_rate_sigma_mps
                ),
                "observerIndices": (
                    "ALL"
                    if args.backup_radial_observer_indices is None
                    else list(args.backup_radial_observer_indices)
                ),
                "independentChannel": args.independent_radial_backup_channel,
            },
            "electroOpticalRadialFailover": {
                "enabled": args.electro_optical_radial_failover,
                "policy": (
                    "radar_primary_failover"
                    if args.electro_optical_radial_failover else None
                ),
                "rangeSigmaM": (
                    args.eo_radial_range_sigma_m
                    if args.electro_optical_radial_failover else None
                ),
                "rangeRateSigmaMps": (
                    args.eo_radial_range_rate_sigma_mps
                    if args.electro_optical_radial_failover else None
                ),
                "physicalValidation": False,
            },
            "preferCompleteRadialLineage": (
                args.prefer_complete_radial_lineage
            ),
            "distributedConsensus": args.distributed_consensus,
            "consensusExchangeHops": args.consensus_exchange_hops,
        },
        "noDropoutControl": control,
        "controlProvenance": control_provenance,
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
    if args.recording_output is not None:
        radial_profile = (
            "independent" if args.independent_radial_backup_channel
            else "legacy" if args.backup_radial_range_sigma_m is not None
            else "none"
        )
        write_bilateral_visualization_recording(
            args.recording_output,
            frontend=recording_frontend,
            captured=recording_capture,
            include_cann=True,
            cann_node_limit=0,
            enable_consensus=args.distributed_consensus,
            show_communication_topology=args.distributed_consensus,
            radial_backup_profile=radial_profile,
            electro_optical_radial_failover=(
                args.electro_optical_radial_failover
            ),
            dropout_modality=str(args.modalities[0]).upper(),
            seed=args.seed,
            source_result_path=args.output,
        )
    print(f"wrote {args.output}")


def _acceptance_control_family(args) -> str:
    nominal = (
        args.backup_radial_range_sigma_m is None
        and args.backup_radial_range_rate_sigma_mps is None
    )
    radar_backup = (
        args.backup_radial_range_sigma_m == 1.0
        and args.backup_radial_range_rate_sigma_mps == 0.0125
        and args.backup_radial_observer_indices is None
        and not args.independent_radial_backup_channel
    )
    fixed_algorithm = (
        args.local_unified_multimodal_update
        and not args.unified_multimodal_update
        and not args.inverse_depth_los_update
        and not args.disable_multi_observer_bearing
        and args.consensus_exchange_hops == 1
        and args.process_noise_acceleration == 1e-4
        and args.bearing_window_epochs == 1
        and args.unified_window_epochs == 1
        and not args.prefer_complete_radial_lineage
    )
    eo_failover = (
        args.electro_optical_radial_failover
        and nominal
        and args.eo_radial_range_sigma_m == 2.5
        and args.eo_radial_range_rate_sigma_mps == 0.010
    )
    if not fixed_algorithm or not (nominal or radar_backup) or (
        args.electro_optical_radial_failover and not eo_failover
    ):
        raise ValueError(
            "--control-input is only supported for a frozen nominal, "
            "radar-backup or electro-optical-failover acceptance configuration."
        )
    if eo_failover:
        return "electro-optical-failover"
    return "nominal" if nominal else "radar-backup"


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
        if arm_name not in record or arm_name not in control:
            continue
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
