"""Scan statistically matched optical/infrared precision during radar dropout."""

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
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dropout-start", type=float, default=6.0)
    parser.add_argument("--dropout-end", type=float, default=13.0)
    parser.add_argument(
        "--angular-scales", type=float, nargs="+", default=(1.0, 0.5, 0.25, 0.1)
    )
    parser.add_argument("--unified-window-epochs", type=int, default=1)
    parser.add_argument("--cann", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/external_acceptance/"
            "p08_radar_dropout_angular_precision_scan_seed2_20s.json"
        ),
    )
    args = parser.parse_args()
    records = []
    for scale in args.angular_scales:
        sensor_config = _scaled_sensor_config(scale)
        common = dict(
            duration_seconds=args.duration,
            sample_interval_seconds=1.0,
            random_seed=args.seed,
            sensor_config=sensor_config,
            use_unified_multimodal_measurement_update=True,
            unified_multimodal_window_epochs=args.unified_window_epochs,
            dropout_start_seconds=args.dropout_start,
            dropout_end_seconds=args.dropout_end,
        )
        control = run_bilateral_online_smoke(
            BilateralFrontendConfig(
                **common,
                dropout_modality="OPTICAL",
                dropout_target_indices=(),
            ),
            enable_cann=args.cann,
        )
        dropout = run_bilateral_online_smoke(
            BilateralFrontendConfig(
                **common,
                dropout_modality="RADAR",
            ),
            enable_cann=args.cann,
        )
        control_phase = control.phase_metrics["DROPOUT"]
        dropout_phase = dropout.phase_metrics["DROPOUT"]
        record = {
            "angularScale": float(scale),
            "opticalCentroidSigmaPixels": 0.2 * float(scale),
            "infraredCentroidSigmaPixels": 0.25 * float(scale),
            "control": asdict(control_phase),
            "radarDropout": asdict(dropout_phase),
            "increasePercent": {
                field: _increase_percent(
                    getattr(control_phase, field), getattr(dropout_phase, field)
                )
                for field in (
                    "position_rmse_m",
                    "velocity_rmse_mps",
                    "acceleration_rmse_mps2",
                )
            },
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
        }
        records.append(record)
        print(
            f"scale={scale:g} "
            f"control={control_phase.position_rmse_m:.6f} m "
            f"dropout={dropout_phase.position_rmse_m:.6f} m "
            f"increase={record['increasePercent']['position_rmse_m']:.3f}%"
        )
    payload = {
        "schemaVersion": "radar-dropout-angular-precision-scan.v1",
        "scenario": {
            "friendlyObserverCount": 15,
            "hostileTargetCount": 15,
            "seed": args.seed,
            "durationSeconds": args.duration,
            "sampleIntervalSeconds": 1.0,
            "dropoutStartSeconds": args.dropout_start,
            "dropoutEndSeconds": args.dropout_end,
            "dropoutTargets": "ALL_15_HOSTILE_TARGETS",
            "cannEnabled": args.cann,
            "unifiedMultimodalMeasurementUpdate": True,
            "unifiedMultimodalWindowEpochs": args.unified_window_epochs,
            "acceptanceThresholdPercent": 15.0,
        },
        "method": (
            "Scale rendered optical/infrared pointing jitter and reported centroid "
            "covariance together; radar settings and all other conditions are fixed."
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.output}")


def _scaled_sensor_config(scale: float) -> MultimodalSensorSimulationConfig:
    value = float(scale)
    if not 0.0 < value <= 1.0:
        raise ValueError("Every angular scale must lie in (0, 1].")
    common = {"width": 17, "height": 17, "read_noise_sigma": 1.0}
    return MultimodalSensorSimulationConfig(
        optical=OpticalCameraConfig(
            **common,
            reported_centroid_sigma_pixels=0.2 * value,
            pointing_jitter_sigma_pixels=0.2 * value,
        ),
        infrared=InfraredCameraConfig(
            **common,
            reported_centroid_sigma_pixels=0.25 * value,
            pointing_jitter_sigma_pixels=0.25 * value,
        ),
        radar=RadarRangeDopplerConfig(
            **common,
            range_jitter_sigma_m=2.5,
            range_rate_jitter_sigma_mps=0.0125,
        ),
    )


def _increase_percent(control: float, affected: float) -> float:
    baseline = float(control)
    return float((float(affected) - baseline) / baseline * 100.0)


if __name__ == "__main__":
    main()
