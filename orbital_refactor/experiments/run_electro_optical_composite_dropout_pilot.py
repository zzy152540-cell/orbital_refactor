"""Pilot idealized optical/infrared radial components without a backup channel."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from adapters.infrared_image_adapter import InfraredCameraConfig
from adapters.multimodal_sensor_simulator import MultimodalSensorSimulationConfig
from adapters.optical_image_adapter import OpticalCameraConfig
from adapters.radar_range_doppler_adapter import RadarRangeDopplerConfig
from experiments.bilateral_multitarget_frontend import (
    BilateralFrontendConfig,
    build_bilateral_raw_frontend,
    derive_bilateral_fault_frontend,
    run_bilateral_cann_acceptance_frontend,
    run_bilateral_online_smoke,
)


def run_pilot(
    *, duration, start, end, seed, range_sigma, range_rate_sigma,
    enable_cann=False,
):
    common_image = {"width": 17, "height": 17, "read_noise_sigma": 1.0}
    sensors = MultimodalSensorSimulationConfig(
        optical=OpticalCameraConfig(
            **common_image,
            reported_centroid_sigma_pixels=0.2,
            pointing_jitter_sigma_pixels=0.2,
        ),
        infrared=InfraredCameraConfig(
            **common_image,
            reported_centroid_sigma_pixels=0.25,
            pointing_jitter_sigma_pixels=0.25,
        ),
        radar=RadarRangeDopplerConfig(
            **common_image,
            range_jitter_sigma_m=2.5,
            range_rate_jitter_sigma_mps=0.0125,
        ),
        optical_range_proxy_sigma_m=float(range_sigma),
        optical_range_rate_proxy_sigma_mps=float(range_rate_sigma),
        infrared_range_proxy_sigma_m=float(range_sigma),
        infrared_range_rate_proxy_sigma_mps=float(range_rate_sigma),
        electro_optical_radial_proxy_policy="radar_primary_failover",
    )
    common = dict(
        duration_seconds=float(duration),
        sample_interval_seconds=1.0,
        random_seed=int(seed),
        sensor_config=sensors,
        use_unified_multimodal_measurement_update=not enable_cann,
        use_local_unified_multimodal_measurement_update=enable_cann,
        dropout_start_seconds=float(start),
        dropout_end_seconds=float(end),
    )
    control_config = BilateralFrontendConfig(
        **common,
        dropout_modality="OPTICAL",
        dropout_target_indices=(),
    )
    nominal_frontend = None
    initialization_cache = None
    if enable_cann:
        nominal_frontend = build_bilateral_raw_frontend(control_config)
        initialization_cache = {}
        control = run_bilateral_cann_acceptance_frontend(
            nominal_frontend,
            iod_initialization_cache=initialization_cache,
        )
    else:
        control = run_bilateral_online_smoke(
            control_config, enable_cann=False,
        )
    control_phase = control.phase_metrics["DROPOUT"]
    dropout_records = {}
    for modality in ("OPTICAL", "INFRARED", "RADAR"):
        fault_config = replace(
            control_config,
            dropout_modality=modality,
            dropout_target_indices=None,
        )
        if enable_cann:
            fault_frontend = derive_bilateral_fault_frontend(
                nominal_frontend, fault_config,
            )
            result = run_bilateral_cann_acceptance_frontend(
                fault_frontend,
                iod_initialization_cache=initialization_cache,
            )
        else:
            result = run_bilateral_online_smoke(
                fault_config, enable_cann=False,
            )
        phase = result.phase_metrics["DROPOUT"]
        dropout_records[modality] = {
            "metrics": asdict(phase),
            "increasePercent": {
                field: _percent_increase(
                    getattr(phase, field), getattr(control_phase, field)
                )
                for field in (
                    "position_rmse_m",
                    "velocity_rmse_mps",
                    "acceleration_rmse_mps2",
                )
            },
            "finalActiveTrackCount": result.final_active_track_count,
            "missingTruthSampleCount": result.missing_truth_sample_count,
            "processingMeanMs": result.processing_mean_ms,
        }
    return {
        "schemaVersion": "electro-optical-composite-dropout-pilot.v1",
        "role": "experimental_only",
        "affectsFrozenAcceptanceConfiguration": False,
        "scenario": {
            "friendlyObserverCount": 15,
            "hostileTargetCount": 15,
            "durationSeconds": float(duration),
            "sampleIntervalSeconds": 1.0,
            "dropoutStartSeconds": float(start),
            "dropoutEndSeconds": float(end),
            "seed": int(seed),
            "opticalAndInfraredRangeSigmaM": float(range_sigma),
            "opticalAndInfraredRangeRateSigmaMps": float(range_rate_sigma),
            "independentRadialBackupEnabled": False,
            "physicalSensorPerformanceValidated": False,
            "incrementalCannEnabled": bool(enable_cann),
            "localUnifiedMultimodalMeasurementUpdate": bool(enable_cann),
        },
        "method": (
            "Optical and infrared observations each carry independently-noised "
            "idealized range and range-rate components. RADAR is primary; when "
            "it is absent one electro-optical proxy is selected and the other "
            "remains on standby. No RADIAL_BACKUP message or independent backup "
            "channel is configured."
        ),
        "control": {
            "metrics": asdict(control_phase),
            "finalActiveTrackCount": control.final_active_track_count,
            "missingTruthSampleCount": control.missing_truth_sample_count,
            "processingMeanMs": control.processing_mean_ms,
        },
        "dropout": dropout_records,
    }


def _percent_increase(value, baseline):
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
    parser.add_argument("--range-sigma-m", type=float, default=1.0)
    parser.add_argument("--range-rate-sigma-mps", type=float, default=0.01)
    parser.add_argument("--enable-cann", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/external_acceptance/"
            "electro_optical_composite_dropout_pilot_seed0_20s.json"
        ),
    )
    args = parser.parse_args()
    payload = run_pilot(
        duration=args.duration,
        start=args.start,
        end=args.end,
        seed=args.seed,
        range_sigma=args.range_sigma_m,
        range_rate_sigma=args.range_rate_sigma_mps,
        enable_cann=args.enable_cann,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for modality, record in payload["dropout"].items():
        increases = record["increasePercent"]
        print(
            f"{modality}: position={increases['position_rmse_m']:.3f}% "
            f"velocity={increases['velocity_rmse_mps']:.3f}% "
            f"acceleration={increases['acceleration_rmse_mps2']:.3f}%"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
