"""Compare direct versus CANN-preprocessed radial backup during radar loss."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from experiments.bilateral_multitarget_frontend import (
    BilateralFrontendConfig,
    build_bilateral_raw_frontend,
    derive_bilateral_fault_frontend,
    run_bilateral_cann_acceptance_frontend,
)


def run_pilot(*, duration, start, end, seed):
    base_config = BilateralFrontendConfig(
        duration_seconds=float(duration),
        sample_interval_seconds=1.0,
        random_seed=int(seed),
        use_local_unified_multimodal_measurement_update=True,
        backup_radial_range_sigma_m=1.0,
        backup_radial_range_rate_sigma_mps=0.0125,
    )
    nominal = build_bilateral_raw_frontend(base_config)
    direct_config = replace(
        base_config,
        dropout_modality="RADAR",
        dropout_start_seconds=float(start),
        dropout_end_seconds=float(end),
        cann_process_radial_backup=False,
    )
    experimental_config = replace(
        direct_config, cann_process_radial_backup=True,
    )
    direct_frontend = derive_bilateral_fault_frontend(
        nominal, direct_config,
    )
    experimental_frontend = derive_bilateral_fault_frontend(
        nominal, experimental_config,
    )
    cache = {}
    direct = run_bilateral_cann_acceptance_frontend(
        direct_frontend, iod_initialization_cache=cache,
    )
    experimental = run_bilateral_cann_acceptance_frontend(
        experimental_frontend, iod_initialization_cache=cache,
    )
    direct_phase = direct.phase_metrics["DROPOUT"]
    experimental_phase = experimental.phase_metrics["DROPOUT"]
    return {
        "schemaVersion": "radial-backup-cann-pilot.v1",
        "role": "experimental_only",
        "affectsFrozenAcceptanceConfiguration": False,
        "scenario": {
            "durationSeconds": float(duration),
            "sampleIntervalSeconds": 1.0,
            "dropoutStartSeconds": float(start),
            "dropoutEndSeconds": float(end),
            "seed": int(seed),
            "missingModality": "RADAR",
            "radialBackupRangeSigmaM": 1.0,
            "radialBackupRangeRateSigmaMps": 0.0125,
            "physicalValidation": False,
        },
        "comparisonContract": {
            "sharedTruthNoiseAndRenderedMessages": True,
            "directArm": "backup values enter unified filter directly",
            "experimentalArm": (
                "backup values pass through dedicated range/range-rate "
                "Line-CANN before the same unified filter"
            ),
        },
        "direct": asdict(direct_phase),
        "radialBackupCann": asdict(experimental_phase),
        "experimentalVsDirectPercent": {
            "position_rmse_m": _percent_change(
                experimental_phase.position_rmse_m,
                direct_phase.position_rmse_m,
            ),
            "velocity_rmse_mps": _percent_change(
                experimental_phase.velocity_rmse_mps,
                direct_phase.velocity_rmse_mps,
            ),
            "acceleration_rmse_mps2": _percent_change(
                experimental_phase.acceleration_rmse_mps2,
                direct_phase.acceleration_rmse_mps2,
            ),
        },
        "processing": {
            "directMeanMs": direct.processing_mean_ms,
            "experimentalMeanMs": experimental.processing_mean_ms,
            "increasePercent": _percent_change(
                experimental.processing_mean_ms, direct.processing_mean_ms,
            ),
        },
    }


def _percent_change(value, baseline):
    value = float(value)
    baseline = float(baseline)
    if not np.isfinite(value) or not np.isfinite(baseline) or baseline == 0.0:
        return None
    return 100.0 * (value / baseline - 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--start", type=float, default=6.0)
    parser.add_argument("--end", type=float, default=13.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "results/external_acceptance/"
            "radial_backup_cann_pilot_seed0_20s.json"
        ),
    )
    args = parser.parse_args()
    payload = run_pilot(
        duration=args.duration, start=args.start, end=args.end,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({
        "directPositionRmseM": payload["direct"]["position_rmse_m"],
        "cannPositionRmseM": payload["radialBackupCann"]["position_rmse_m"],
        "experimentalVsDirectPercent": payload[
            "experimentalVsDirectPercent"
        ],
        "processing": payload["processing"],
    }, ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
