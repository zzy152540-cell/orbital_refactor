"""Run the shared-frame bilateral estimator baseline matrix."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass
import json
from pathlib import Path

import numpy as np

from experiments.bilateral_multitarget_frontend import (
    BilateralFrontendConfig,
    run_bilateral_baseline_matrix,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--backup-radial-range-sigma-m", type=float)
    parser.add_argument("--backup-radial-range-rate-sigma-mps", type=float)
    parser.add_argument(
        "--fixed-federated-weights", type=float, nargs=3,
        metavar=("RADAR", "INFRARED", "OPTICAL"),
        help=(
            "Prior-information allocation used only by "
            "p05-fixed-weight-pilot."
        ),
    )
    parser.add_argument(
        "--independent-radial-backup-channel", action="store_true",
        help="Separate the numerical radial backup from its legacy IR carrier.",
    )
    parser.add_argument(
        "--primary-acceptance-only", action="store_true",
        help=(
            "Run only the five estimator arms needed by the frozen P05/P07 "
            "primary acceptance metrics."
        ),
    )
    parser.add_argument(
        "--acceptance-scope",
        choices=(
            "p05-control", "p05-candidate", "p05-optical-pilot",
            "p05-fixed-weight-pilot",
            "p07-control", "p07-candidate",
        ),
        help=(
            "Run only the one control arm or matched control/candidate pair "
            "required by the selected acceptance item."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/external_acceptance/bilateral_baseline_matrix.json"
        ),
    )
    args = parser.parse_args()
    config = BilateralFrontendConfig(
        duration_seconds=args.duration,
        sample_interval_seconds=1.0,
        random_seed=args.seed,
        backup_radial_range_sigma_m=args.backup_radial_range_sigma_m,
        backup_radial_range_rate_sigma_mps=(
            args.backup_radial_range_rate_sigma_mps
        ),
        independent_radial_backup_channel=(
            args.independent_radial_backup_channel
        ),
    )
    result = run_bilateral_baseline_matrix(
        config,
        primary_acceptance_only=args.primary_acceptance_only,
        acceptance_scope=args.acceptance_scope,
        fixed_federated_weights=(
            None if args.fixed_federated_weights is None else {
                "rad": args.fixed_federated_weights[0],
                "ir": args.fixed_federated_weights[1],
                "opt": args.fixed_federated_weights[2],
            }
        ),
    )
    payload = {
        "schemaVersion": "bilateral-baseline-matrix.v3",
        "scenario": {
            "friendlyObserverCount": 15,
            "hostileTargetCount": 15,
            "durationSeconds": args.duration,
            "sampleIntervalSeconds": 1.0,
            "seed": args.seed,
            "armSelection": (
                args.acceptance_scope.upper().replace("-", "_")
                if args.acceptance_scope else (
                    "P05_P07_PRIMARY"
                    if args.primary_acceptance_only else "FULL"
                )
            ),
            "independentRadialBackup": {
                "enabled": args.backup_radial_range_sigma_m is not None,
                "rangeSigmaM": args.backup_radial_range_sigma_m,
                "rangeRateSigmaMps": args.backup_radial_range_rate_sigma_mps,
                "independentChannel": args.independent_radial_backup_channel,
                "physicalValidation": False,
            },
        },
        "comparisonContract": {
            "sharedRenderedSensorFrames": True,
            "fixedTruthAndNoiseAcrossArms": True,
            "usesCommonMultimodalInitialization": (
                result.uses_common_multimodal_initialization
            ),
            "singleModalityMeaning": (
                "Configured modality only after common radar-plus-LOS IOD"
            ),
        },
        "trackingModalitiesByArm": result.tracking_modalities_by_arm,
        "messageCountByTimestamp": result.message_count_by_timestamp,
        "localNodeBaselines": {
            name: _node_view_payload(metrics)
            for name, metrics in result.local_node_baselines.items()
        },
        "matchedSingleObserverCann": (
            asdict(result.matched_single_observer_cann_metrics)
            if result.matched_single_observer_cann_metrics is not None
            else None
        ),
        "matchedSingleObserverJointCann": (
            asdict(result.matched_single_observer_joint_cann_metrics)
            if result.matched_single_observer_joint_cann_metrics is not None
            else None
        ),
        "matchedSingleObserverAblation": {
            name: asdict(metrics)
            for name, metrics in (
                result.matched_single_observer_ablation_metrics.items()
            )
        },
        "matchedSingleObserverCannVsModality": {
            name: asdict(metrics)
            for name, metrics in (
                result.matched_single_observer_cann_vs_modality.items()
            )
        },
        "matchedSingleObserverJointCannVsModality": {
            name: asdict(metrics)
            for name, metrics in (
                result.matched_single_observer_joint_cann_vs_modality.items()
            )
        },
        "matchedIndependentConsensusByCann": {
            name: asdict(metrics)
            for name, metrics in (
                result.matched_independent_consensus_metrics_by_cann.items()
            )
        },
        "matchedJointIndependentConsensus": (
            asdict(result.matched_joint_independent_consensus_metrics)
            if result.matched_joint_independent_consensus_metrics is not None
            else None
        ),
        "arms": {
            name: _arm_payload(arm) for name, arm in result.arms.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for name, arm in result.arms.items():
        print(
            name,
            f"positionRmse={arm.final_position_rmse_m:.6f}",
            f"velocityRmse={arm.final_velocity_rmse_mps:.6f}",
            f"identitySwitches={arm.identity_switch_count}",
        )
    print(f"wrote {args.output}")


def _arm_payload(value):
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
    return payload


def _node_view_payload(value):
    payload = {
        item.name: getattr(value, item.name)
        for item in fields(value)
        if item.name != "phase_metrics"
    }
    payload["phase_metrics"] = {
        name: asdict(metrics) for name, metrics in value.phase_metrics.items()
    }
    return payload


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
