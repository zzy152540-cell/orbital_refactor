"""Paired Walker filter shadow for radar and optical covariance calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np

from adapters.frontend_covariance_calibration import (
    calibration_table_from_error_budget,
)
from adapters.optical_image_adapter import OpticalCameraConfig
from adapters.radar_range_doppler_adapter import RadarRangeDopplerConfig
from experiments.radar_optical_error_budget_audit import (
    DEFAULT_OPTICAL_PROFILES,
    DEFAULT_RADAR_PROFILES,
    run_radar_optical_error_budget_audit,
)
from experiments.walker_optical_image_comparison import (
    replace_optical_messages_with_images,
)
from experiments.walker_raw_sensor_comparison import (
    _mean_modality_nis,
    _position_rmse,
    _relative_position_rmse,
    _run,
    _walker_case,
    replace_radar_messages_with_power_maps,
)


@dataclass(frozen=True)
class RadarOpticalFilterCalibrationArm:
    modality: str
    name: str
    position_rmse_m: float
    relative_position_rmse_m: float
    mean_nis: float
    valid_measurement_count: int


@dataclass(frozen=True)
class RadarOpticalFilterCalibrationShadowReport:
    duration_seconds: float
    dt_seconds: float
    seed: int
    calibration_seed: int
    calibration_sample_count: int
    profile: str
    paired_raw_measurements: bool
    default_filter_unchanged: bool
    arms: tuple[RadarOpticalFilterCalibrationArm, ...]


def run_radar_optical_filter_calibration_shadow(
    *, duration=20.0, dt=2.0, seed=0,
    calibration_seed=701, calibration_samples=300,
):
    profile_name = "moderate_combined"
    optical_profile = next(
        item for item in DEFAULT_OPTICAL_PROFILES if item.name == profile_name
    )
    radar_profile = next(
        item for item in DEFAULT_RADAR_PROFILES if item.name == profile_name
    )
    calibration = run_radar_optical_error_budget_audit(
        seed=calibration_seed, monte_carlo_samples=calibration_samples,
        optical_profiles=(optical_profile,), radar_profiles=(radar_profile,),
    )
    tables = {
        modality: calibration_table_from_error_budget(
            calibration, modality=modality, profile=profile_name,
        )
        for modality in ("OPTICAL", "RADAR")
    }
    corrected = {
        modality: replace(table, apply_bias_correction=True)
        for modality, table in tables.items()
    }
    optical_config = OpticalCameraConfig(
        width=128, height=128, focal_length_x_pixels=400.0,
        focal_length_y_pixels=400.0, source_peak=1000.0,
        background=10.0, read_noise_sigma=1.0,
        photon_noise_enabled=optical_profile.photon_noise_enabled,
        stray_light_drift_sigma=optical_profile.stray_light_drift_sigma,
        pointing_jitter_sigma_pixels=optical_profile.pointing_jitter_sigma_pixels,
        radial_distortion_k1_per_pixel2=(
            optical_profile.radial_distortion_k1_per_pixel2
        ),
        fixed_pixel_bias_x=optical_profile.fixed_pixel_bias_x,
        fixed_pixel_bias_y=optical_profile.fixed_pixel_bias_y,
    )
    radar_config = RadarRangeDopplerConfig(
        width=129, height=129, source_peak=1000.0,
        background=10.0, read_noise_sigma=1.0,
        background_drift_sigma=radar_profile.background_drift_sigma,
        echo_amplitude_sigma_fraction=radar_profile.echo_amplitude_sigma_fraction,
        range_bias_m=radar_profile.range_bias_m,
        range_rate_bias_mps=radar_profile.range_rate_bias_mps,
        range_jitter_sigma_m=radar_profile.range_jitter_sigma_m,
        range_rate_jitter_sigma_mps=radar_profile.range_rate_jitter_sigma_mps,
    )
    case = _walker_case(seed=seed, duration=duration, dt=dt)
    arms = []
    for modality, config, replacer, rng_seed in (
        ("OPTICAL", optical_config, replace_optical_messages_with_images, 9200),
        ("RADAR", radar_config, replace_radar_messages_with_power_maps, 9300),
    ):
        for name, table in (
            ("fixed_reported_covariance", None),
            ("empirical_covariance", tables[modality]),
            ("bias_corrected_empirical_covariance", corrected[modality]),
        ):
            kwargs = dict(
                messages=case["observations"], timestamps=case["timestamps"],
                truth_state_history_by_node=case["truth"], config=config,
                rng=np.random.default_rng(rng_seed + seed),
                covariance_calibration=table,
            )
            replaced = replacer(**kwargs)
            messages = replaced[0] if modality == "OPTICAL" else replaced
            history = _run(case, messages)
            arms.append(RadarOpticalFilterCalibrationArm(
                modality=modality, name=name,
                position_rmse_m=_position_rmse(history, case["truth"]),
                relative_position_rmse_m=_relative_position_rmse(
                    history, case["truth"], case["topology"],
                ),
                mean_nis=_mean_modality_nis(history, messages, modality),
                valid_measurement_count=sum(
                    message.modality.upper() == modality and message.valid_flag
                    for message in messages
                ),
            ))
    return RadarOpticalFilterCalibrationShadowReport(
        duration_seconds=float(duration), dt_seconds=float(dt), seed=int(seed),
        calibration_seed=int(calibration_seed),
        calibration_sample_count=int(calibration_samples), profile=profile_name,
        paired_raw_measurements=True, default_filter_unchanged=True,
        arms=tuple(arms),
    )


def save_radar_optical_filter_calibration_shadow(report, output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "summary.json"
    path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path
