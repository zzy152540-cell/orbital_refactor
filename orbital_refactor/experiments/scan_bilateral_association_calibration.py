"""Focused scan of association prediction-uncertainty floors.

The rendered sensor stream is kept identical across profiles.  This scan is
intended to find the smallest uncertainty floor that preserves all expected
tracks before robustness percentages are interpreted.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from experiments.bilateral_multitarget_frontend import (
    BilateralFrontendConfig,
    _run_bilateral_online_arm,
    build_bilateral_raw_frontend,
)


PROFILES = (
    (1_500.0, 0.5),
    (1_500.0, 1.0),
    (1_500.0, 2.0),
    (2_000.0, 0.25),
    (2_000.0, 0.5),
    (2_000.0, 1.0),
    (2_000.0, 2.0),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--profile",
        type=float,
        nargs=2,
        action="append",
        metavar=("POSITION_SIGMA_M", "VELOCITY_SIGMA_MPS"),
        help="Association uncertainty profile; repeat to scan several profiles.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/external_acceptance/"
            "bilateral_association_calibration_scan.json"
        ),
    )
    args = parser.parse_args()
    base_config = BilateralFrontendConfig(
        duration_seconds=args.duration,
        sample_interval_seconds=1.0,
        random_seed=args.seed,
    )
    base_frontend = build_bilateral_raw_frontend(base_config)
    records = []
    profiles = tuple(map(tuple, args.profile)) if args.profile else PROFILES
    for position_floor, velocity_floor in profiles:
        config = replace(
            base_config,
            association_position_sigma_floor_m=position_floor,
            association_velocity_sigma_floor_mps=velocity_floor,
        )
        result = _run_bilateral_online_arm(
            replace(base_frontend, config=config),
            enable_cann=False,
        )
        record = {
            "positionSigmaFloorM": position_floor,
            "velocitySigmaFloorMps": velocity_floor,
            "finalActiveTrackCount": result.final_active_track_count,
            "initializedTrackCount": result.initialized_track_count,
            "retiredTrackCount": result.retired_track_count,
            "trackReplacementCount": result.track_replacement_count,
            "missingTruthSampleCount": result.missing_truth_sample_count,
            "associationMatchCount": result.association_match_count,
            "associationUnassignedObservationCount": (
                result.association_unassigned_observation_count
            ),
            "associationAmbiguousObservationCount": (
                result.association_ambiguous_observation_count
            ),
            "associationUnobservedTrackCount": (
                result.association_unobserved_track_count
            ),
            "suppressedBirthCount": result.suppressed_birth_count,
            "positionRmseM": result.final_position_rmse_m,
            "velocityRmseMps": result.final_velocity_rmse_mps,
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False))
    payload = {
        "schemaVersion": "bilateral-association-calibration-scan.v1",
        "scenario": {
            "friendlyObserverCount": 15,
            "hostileTargetCount": 15,
            "durationSeconds": args.duration,
            "sampleIntervalSeconds": 1.0,
            "seed": args.seed,
        },
        "fixedSensorStream": True,
        "profiles": records,
        "baseConfig": asdict(base_config),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
