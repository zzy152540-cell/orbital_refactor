"""Run the shared-frame bilateral estimator baseline matrix."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, fields
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
    )
    result = run_bilateral_baseline_matrix(config)
    payload = {
        "schemaVersion": "bilateral-baseline-matrix.v1",
        "scenario": {
            "friendlyObserverCount": 15,
            "hostileTargetCount": 15,
            "durationSeconds": args.duration,
            "sampleIntervalSeconds": 1.0,
            "seed": args.seed,
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
