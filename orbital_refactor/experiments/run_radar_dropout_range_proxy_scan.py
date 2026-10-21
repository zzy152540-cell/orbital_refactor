"""Scan idealized infrared weak-range precision during radar dropout."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from adapters.infrared_image_adapter import InfraredCameraConfig
from adapters.multimodal_sensor_simulator import MultimodalSensorSimulationConfig
from adapters.optical_image_adapter import OpticalCameraConfig
from adapters.radar_range_doppler_adapter import RadarRangeDopplerConfig
from experiments.bilateral_multitarget_frontend import (
    BilateralFrontendConfig,
    run_bilateral_online_smoke,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dropout-start", type=float, default=6.0)
    parser.add_argument("--dropout-end", type=float, default=13.0)
    parser.add_argument(
        "--proxy-sigmas-m",
        type=float,
        nargs="+",
        default=(0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0),
        help="Per-observer 1-sigma idealized infrared range error; 0 disables it.",
    )
    parser.add_argument(
        "--proxy-range-rate-sigma-mps",
        type=float,
        default=0.0,
        help=(
            "Per-observer 1-sigma idealized infrared range-rate error; "
            "0 disables it."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/external_acceptance/"
            "p08_radar_dropout_infrared_range_proxy_seed0_20s.json"
        ),
    )
    args = parser.parse_args()
    if args.proxy_range_rate_sigma_mps < 0.0:
        raise ValueError("Proxy range-rate sigma must be nonnegative.")
    range_rate_sigma = float(args.proxy_range_rate_sigma_mps)
    records = []
    for raw_sigma in args.proxy_sigmas_m:
        sigma = float(raw_sigma)
        if sigma < 0.0:
            raise ValueError("Proxy sigma values must be nonnegative.")
        sensor_config = _sensor_config(
            None if sigma == 0.0 else sigma,
            None if range_rate_sigma == 0.0 else range_rate_sigma,
        )
        common = dict(
            duration_seconds=args.duration,
            sample_interval_seconds=1.0,
            random_seed=args.seed,
            sensor_config=sensor_config,
            use_unified_multimodal_measurement_update=True,
            dropout_start_seconds=args.dropout_start,
            dropout_end_seconds=args.dropout_end,
        )
        control = run_bilateral_online_smoke(
            BilateralFrontendConfig(
                **common,
                dropout_modality="OPTICAL",
                dropout_target_indices=(),
            ),
            enable_cann=False,
        )
        dropout = run_bilateral_online_smoke(
            BilateralFrontendConfig(
                **common,
                dropout_modality="RADAR",
            ),
            enable_cann=False,
        )
        control_phase = control.phase_metrics["DROPOUT"]
        dropout_phase = dropout.phase_metrics["DROPOUT"]
        increases = {
            field: _increase_percent(
                getattr(control_phase, field), getattr(dropout_phase, field)
            )
            for field in (
                "position_rmse_m",
                "velocity_rmse_mps",
                "acceleration_rmse_mps2",
            )
        }
        records.append({
            "infraredRangeProxySigmaM": None if sigma == 0.0 else sigma,
            "infraredRangeRateProxySigmaMps": (
                None if range_rate_sigma == 0.0 else range_rate_sigma
            ),
            "control": asdict(control_phase),
            "radarDropout": asdict(dropout_phase),
            "increasePercent": increases,
            "coverage": {
                "controlFinalActiveTrackCount": control.final_active_track_count,
                "dropoutFinalActiveTrackCount": dropout.final_active_track_count,
                "controlMissingTruthSampleCount": control.missing_truth_sample_count,
                "dropoutMissingTruthSampleCount": dropout.missing_truth_sample_count,
            },
            "processingMeanMs": {
                "control": control.processing_mean_ms,
                "radarDropout": dropout.processing_mean_ms,
            },
        })
        label = "disabled" if sigma == 0.0 else f"{sigma:g} m"
        print(
            f"proxy_sigma={label} "
            f"control={control_phase.position_rmse_m:.6f} m "
            f"dropout={dropout_phase.position_rmse_m:.6f} m "
            f"increase={increases['position_rmse_m']:.3f}%"
        )
    payload = {
        "schemaVersion": "radar-dropout-range-proxy-scan.v1",
        "scenario": {
            "friendlyObserverCount": 15,
            "hostileTargetCount": 15,
            "seed": args.seed,
            "durationSeconds": args.duration,
            "sampleIntervalSeconds": 1.0,
            "dropoutStartSeconds": args.dropout_start,
            "dropoutEndSeconds": args.dropout_end,
            "dropoutTargets": "ALL_15_HOSTILE_TARGETS",
            "unifiedMultimodalMeasurementUpdate": True,
            "acceptanceThresholdPercent": 15.0,
            "infraredRangeRateProxySigmaMps": (
                None if range_rate_sigma == 0.0 else range_rate_sigma
            ),
        },
        "method": (
            "Idealized sensitivity-only infrared range proxy. Each valid infrared "
            "observation receives an independent Gaussian range sample generated "
            "inside the sensor simulator. This is not yet a validated radiometric "
            "or angular-size sensor model."
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.output}")


def _sensor_config(
    infrared_range_proxy_sigma_m: float | None,
    infrared_range_rate_proxy_sigma_mps: float | None,
) -> MultimodalSensorSimulationConfig:
    common = {"width": 17, "height": 17, "read_noise_sigma": 1.0}
    return MultimodalSensorSimulationConfig(
        optical=OpticalCameraConfig(
            **common,
            reported_centroid_sigma_pixels=0.2,
            pointing_jitter_sigma_pixels=0.2,
        ),
        infrared=InfraredCameraConfig(
            **common,
            reported_centroid_sigma_pixels=0.25,
            pointing_jitter_sigma_pixels=0.25,
        ),
        radar=RadarRangeDopplerConfig(
            **common,
            range_jitter_sigma_m=2.5,
            range_rate_jitter_sigma_mps=0.0125,
        ),
        infrared_range_proxy_sigma_m=infrared_range_proxy_sigma_m,
        infrared_range_rate_proxy_sigma_mps=(
            infrared_range_rate_proxy_sigma_mps
        ),
    )


def _increase_percent(control: float, affected: float) -> float:
    baseline = float(control)
    return float((float(affected) - baseline) / baseline * 100.0)


if __name__ == "__main__":
    main()
