"""Compare a conventional sequential multimodal EKF with joint CANN fusion.

This is an analysis-only P05 pilot.  Both arms consume the same rendered
sensor frames, observer, target set, initialization cache and scoring epochs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from experiments.bilateral_multitarget_frontend import (
    BilateralFrontendConfig,
    _matched_view_metrics,
    _run_bilateral_online_arm,
    _sensor_config,
    build_bilateral_raw_frontend,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--observer-index", type=int, default=5)
    parser.add_argument("--radar-noise-scale", type=float, default=1.0)
    parser.add_argument("--candidate-radar-alpha", type=float)
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "results/current_audit/"
            "p05_sequential_baseline_vs_joint_cann_seed0_120s.json"
        ),
    )
    args = parser.parse_args()

    noise_scale = float(args.radar_noise_scale)
    if noise_scale <= 0.0:
        raise ValueError("radar-noise-scale must be positive")
    sensor_config = _sensor_config(17)
    sensor_config = replace(
        sensor_config,
        radar=replace(
            sensor_config.radar,
            range_jitter_sigma_m=(
                sensor_config.radar.range_jitter_sigma_m * noise_scale
            ),
            range_rate_jitter_sigma_mps=(
                sensor_config.radar.range_rate_jitter_sigma_mps
                * noise_scale
            ),
        ),
    )
    frontend = build_bilateral_raw_frontend(BilateralFrontendConfig(
        duration_seconds=float(args.duration),
        sample_interval_seconds=1.0,
        random_seed=int(args.seed),
        sensor_config=sensor_config,
    ))
    observer_ids = frontend.truth.manifest.friendly_observer_ids
    observer_index = int(args.observer_index)
    if not 0 <= observer_index < len(observer_ids):
        raise ValueError("observer-index must identify a friendly observer")
    observer_id = observer_ids[observer_index]

    initialization_cache = {}
    collectors = {}
    candidate_grid = (
        (
            float(args.candidate_radar_alpha),
            float(args.candidate_radar_alpha)
            / (2.0 - float(args.candidate_radar_alpha)),
            1.0, 1,
        ),
    ) if args.candidate_radar_alpha is not None else (
        (1.0, 1.0, 1.0, 1),
        (0.9, 0.9 / 1.1, 1.0, 1),
        (0.85, 0.85 / 1.15, 1.0, 1),
        (0.8, 0.8 / 1.2, 1.0, 1),
        (0.75, 0.75 / 1.25, 1.0, 1),
        (0.7, 0.7 / 1.3, 1.0, 1),
        (0.65, 0.65 / 1.35, 1.0, 1),
        (0.6, 0.6 / 1.4, 1.0, 1),
        (0.55, 0.55 / 1.45, 1.0, 1),
        (0.5, 0.5 / 1.5, 1.0, 1),
        (0.45, 0.45 / 1.55, 1.0, 1),
    )
    arm_specs = (
        (
            "sequential_multimodal_ekf", False, "sequential",
            ("RADAR", "INFRARED", "OPTICAL"), 1.0, 1.0, 1.0, 1,
        ),
        (
            "joint_multimodal_ekf", False, "batch",
            ("RADAR", "INFRARED", "OPTICAL"), 1.0, 1.0, 1.0, 1,
        ),
        *((
            f"joint_multimodal_cann_r{radar_alpha:g}_"
            f"c{covariance_scale:g}_o{optical_alpha:g}_w{window_epochs}",
            True, "batch", ("RADAR", "INFRARED", "OPTICAL"),
            radar_alpha, covariance_scale, optical_alpha, window_epochs,
        ) for radar_alpha, covariance_scale, optical_alpha, window_epochs
        in candidate_grid),
    )
    for (
        name, enable_cann, update_mode, modalities,
        radar_blend, radar_covariance_scale, optical_blend, window_epochs,
    ) in arm_specs:
        collector = {}
        _run_bilateral_online_arm(
            frontend,
            enable_cann=enable_cann,
            enable_consensus=False,
            enable_local_unified=True,
            local_multimodal_update_mode=update_mode,
            cann_radar_nominal_blend_alpha=radar_blend,
            cann_radar_nominal_covariance_scale=radar_covariance_scale,
            cann_optical_nominal_blend_alpha=optical_blend,
            local_multimodal_window_epochs=window_epochs,
            velocity_refinement_alpha=(0.3 if enable_cann else None),
            tracking_modalities=modalities,
            tracking_observer_ids=(observer_id,),
            sample_collector=collector,
            iod_initialization_cache=initialization_cache,
        )
        collectors[name] = collector

    baseline = _observer_samples(
        collectors["sequential_multimodal_ekf"], observer_id,
    )
    comparisons = {}
    for name, collector in collectors.items():
        if not name.startswith("joint_multimodal_cann_"):
            continue
        candidate = _observer_samples(collector, observer_id)
        comparisons[name] = asdict(_matched_view_metrics(
            baseline["state"], candidate["state"],
            baseline["acceleration"], candidate["acceleration"],
        ))
    payload = {
        "schemaVersion": "p05-sequential-baseline-pilot.v1",
        "acceptanceDefaultChanged": False,
        "scenario": {
            "durationSeconds": float(args.duration),
            "sampleIntervalSeconds": 1.0,
            "seed": int(args.seed),
            "observerIndex": observer_index,
            "observerId": observer_id,
            "radarNoiseScale": noise_scale,
        },
        "comparisonContract": {
            "sharedRenderedSensorFrames": True,
            "sharedInitialization": True,
            "sharedObserverTargetAndEpochKeys": True,
            "baseline": (
                "RADAR then INFRARED then OPTICAL sequential EKF; "
                "relinearize after every measurement; CANN disabled"
            ),
            "candidate": (
                "same-epoch stacked joint multimodal EKF; CANN enabled"
            ),
        },
        "sampleCounts": {
            name: len(_observer_samples(collector, observer_id)["state"])
            for name, collector in collectors.items()
        },
        "exactMatchedSequentialBaselineToCandidates": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(args.output)


def _observer_samples(collector, observer_id: str) -> dict:
    return {
        "state": {
            key: sample for key, sample in collector["local"].items()
            if key[0] == observer_id
        },
        "acceleration": {
            key: sample
            for key, sample in collector["localAcceleration"].items()
            if key[0] == observer_id
        },
    }


if __name__ == "__main__":
    main()
