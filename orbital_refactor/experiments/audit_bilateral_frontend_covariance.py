"""Audit raw multimodal measurement bias and reported covariance consistency.

This diagnostic uses simulation truth only for offline acceptance analysis.  It
does not feed truth, empirical bias, or covariance corrections back into the
online tracker.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from adapters.infrared_image_adapter import InfraredCameraConfig
from adapters.multimodal_sensor_simulator import MultimodalSensorSimulationConfig
from adapters.optical_image_adapter import OpticalCameraConfig
from adapters.radar_range_doppler_adapter import RadarRangeDopplerConfig
from orbital_core.measurements import (
    measure_relative_az_el,
    measure_relative_optical_uv,
    measure_relative_range,
    measure_relative_range_rate,
    wrap_angle,
)

from experiments.bilateral_multitarget_frontend import (
    BilateralFrontendConfig,
    build_bilateral_raw_frontend,
)


CHI2_2DOF_95 = 5.991464547107979


def audit_bilateral_frontend_covariance(
    *,
    seeds=(0,),
    duration_seconds=20.0,
    sample_interval_seconds=1.0,
    image_size=17,
    sensor_config=None,
):
    """Return pooled residual/covariance diagnostics for each raw modality."""

    residuals = {name: [] for name in ("RADAR", "INFRARED", "OPTICAL")}
    covariances = {name: [] for name in residuals}
    squared_normalized_residuals = {name: [] for name in residuals}
    total_counts = {name: 0 for name in residuals}
    valid_counts = {name: 0 for name in residuals}

    normalized_seeds = tuple(int(value) for value in seeds)
    for seed in normalized_seeds:
        frontend = build_bilateral_raw_frontend(BilateralFrontendConfig(
            duration_seconds=float(duration_seconds),
            sample_interval_seconds=float(sample_interval_seconds),
            image_size=int(image_size),
            random_seed=seed,
            sensor_config=sensor_config,
        ))
        timestamp_index = {
            float(timestamp): index
            for index, timestamp in enumerate(frontend.truth.timestamps)
        }
        target_histories = (
            frontend.truth.hostile_truth_state_history_by_target
        )
        for timestamp, messages in frontend.messages_by_timestamp.items():
            target_index = timestamp_index[float(timestamp)]
            for message in messages:
                modality = str(message.modality).upper()
                total_counts[modality] += 1
                if not message.valid_flag:
                    continue
                target_id = frontend.truth_target_by_detection_group[
                    message.detection_group_id
                ]
                observer = frontend.observer_state_by_epoch[
                    (message.observer_id, float(timestamp))
                ]
                target = target_histories[target_id][target_index]
                ideal = _ideal_measurement(message, observer, target)
                residual = np.asarray(message.measurement, dtype=float) - ideal
                if modality == "INFRARED":
                    residual = wrap_angle(residual)
                covariance = np.asarray(message.covariance, dtype=float)
                residuals[modality].append(residual)
                covariances[modality].append(covariance)
                squared_normalized_residuals[modality].append(float(
                    residual @ np.linalg.pinv(covariance) @ residual
                ))
                valid_counts[modality] += 1

    return {
        "schemaVersion": "bilateral-frontend-covariance-audit.v1",
        "truthUsage": "offline_diagnostic_only",
        "seedCount": len(normalized_seeds),
        "seeds": list(normalized_seeds),
        "durationSeconds": float(duration_seconds),
        "sampleIntervalSeconds": float(sample_interval_seconds),
        "imageSize": int(image_size),
        "modalities": {
            modality: _summarize_modality(
                residuals[modality],
                covariances[modality],
                squared_normalized_residuals[modality],
                total_count=total_counts[modality],
                valid_count=valid_counts[modality],
            )
            for modality in residuals
        },
    }


def reported_sigma_matched_sensor_config(image_size=17):
    """Enable simulated jitter matching each frontend's reported 1-sigma."""

    common = {
        "width": int(image_size),
        "height": int(image_size),
        "read_noise_sigma": 1.0,
    }
    return MultimodalSensorSimulationConfig(
        optical=OpticalCameraConfig(
            **common, pointing_jitter_sigma_pixels=0.2,
        ),
        infrared=InfraredCameraConfig(
            **common, pointing_jitter_sigma_pixels=0.25,
        ),
        radar=RadarRangeDopplerConfig(
            **common,
            range_jitter_sigma_m=2.5,
            range_rate_jitter_sigma_mps=0.0125,
        ),
    )


def _ideal_measurement(message, observer, target):
    modality = str(message.modality).upper()
    if modality == "RADAR":
        return np.array([
            measure_relative_range(observer, target),
            measure_relative_range_rate(observer, target),
        ])
    quaternion = np.asarray(
        message.metadata["quaternion_i2b_wxyz"], dtype=float,
    )
    if modality == "INFRARED":
        return measure_relative_az_el(
            observer, target, frame="BODY",
            quaternion_i2b_wxyz=quaternion,
        )
    if modality == "OPTICAL":
        return measure_relative_optical_uv(
            observer, target, frame="BODY",
            quaternion_i2b_wxyz=quaternion,
        )
    raise ValueError(f"Unsupported modality: {modality}")


def _summarize_modality(
    residual_values,
    covariance_values,
    squared_normalized_residual_values,
    *,
    total_count,
    valid_count,
):
    if not residual_values:
        return {
            "totalCount": int(total_count),
            "validCount": int(valid_count),
            "validFraction": 0.0,
        }
    residual = np.asarray(residual_values, dtype=float)
    covariance = np.asarray(covariance_values, dtype=float)
    squared_normalized = np.asarray(
        squared_normalized_residual_values, dtype=float,
    )
    empirical_covariance = (
        np.cov(residual, rowvar=False, ddof=1)
        if len(residual) > 1 else np.zeros((residual.shape[1],) * 2)
    )
    degrees_of_freedom = int(residual.shape[1])
    mean_nsr = float(np.mean(squared_normalized))
    return {
        "totalCount": int(total_count),
        "validCount": int(valid_count),
        "validFraction": float(valid_count / total_count),
        "degreesOfFreedom": degrees_of_freedom,
        "meanResidual": np.mean(residual, axis=0).tolist(),
        "componentRmse": np.sqrt(np.mean(np.square(residual), axis=0)).tolist(),
        "empiricalCovariance": empirical_covariance.tolist(),
        "meanReportedCovariance": np.mean(covariance, axis=0).tolist(),
        "meanSquaredNormalizedResidual": mean_nsr,
        "normalizedMeanSquaredResidual": float(
            mean_nsr / degrees_of_freedom
        ),
        "coverageAtChiSquare95": float(
            np.mean(squared_normalized <= CHI2_2DOF_95)
        ),
        "scalarCovarianceScaleCandidate": float(
            mean_nsr / degrees_of_freedom
        ),
        "biasCorrectionCandidate": np.mean(residual, axis=0).tolist(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=17)
    parser.add_argument(
        "--noise-profile",
        choices=("current", "reported-sigma-matched"),
        default="current",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = audit_bilateral_frontend_covariance(
        seeds=args.seeds,
        duration_seconds=args.duration,
        sample_interval_seconds=args.dt,
        image_size=args.image_size,
        sensor_config=(
            reported_sigma_matched_sensor_config(args.image_size)
            if args.noise_profile == "reported-sigma-matched" else None
        ),
    )
    result["noiseProfile"] = args.noise_profile
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
