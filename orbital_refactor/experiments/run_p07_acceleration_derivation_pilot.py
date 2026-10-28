"""Compare causal acceleration derivations for the frozen P07 candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.bilateral_multitarget_frontend import (
    BilateralFrontendConfig,
    _run_bilateral_online_arm,
    build_bilateral_raw_frontend,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--windows", type=int, nargs="+", default=(3, 5, 9))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = BilateralFrontendConfig(
        duration_seconds=args.duration,
        sample_interval_seconds=1.0,
        random_seed=args.seed,
    )
    frontend = build_bilateral_raw_frontend(config)
    windows = tuple(args.windows)
    local = {}
    cooperative = {}
    initialization_cache = {}
    _run_bilateral_online_arm(
        frontend,
        enable_cann=True,
        enable_local_unified=True,
        sample_collector=local,
        acceleration_pilot_windows=windows,
        iod_initialization_cache=initialization_cache,
    )
    _run_bilateral_online_arm(
        frontend,
        enable_cann=True,
        enable_consensus=True,
        enable_local_unified=True,
        sample_collector=cooperative,
        acceleration_pilot_windows=windows,
        iod_initialization_cache=initialization_cache,
    )
    records = []
    local_variants = local["localAccelerationPilot"]
    cooperative_variants = cooperative["consensusAccelerationPilot"]
    for method in sorted(local_variants):
        local_samples = local_variants[method]
        cooperative_samples = cooperative_variants[method]
        common = sorted(set(local_samples) & set(cooperative_samples))
        local_rmse = _rmse(local_samples[key] for key in common)
        cooperative_rmse = _rmse(cooperative_samples[key] for key in common)
        improvement = 100.0 * (
            local_rmse - cooperative_rmse
        ) / local_rmse
        records.append({
            "method": method,
            "commonSampleCount": len(common),
            "independentRmseMps2": local_rmse,
            "cooperativeRmseMps2": cooperative_rmse,
            "improvementPercent": improvement,
            "passesSixPercent": bool(improvement >= 6.0),
        })
    payload = {
        "schemaVersion": "p07-acceleration-derivation-pilot.v1",
        "seed": args.seed,
        "durationSeconds": args.duration,
        "causalOnly": True,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _rmse(values) -> float:
    array = np.asarray(tuple(values), dtype=float)
    return float(np.sqrt(np.mean(np.square(array))))


if __name__ == "__main__":
    main()
