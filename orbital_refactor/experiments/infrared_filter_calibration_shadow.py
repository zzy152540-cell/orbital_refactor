"""Walker filter shadow comparison for infrared bias/covariance calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from adapters.infrared_covariance_calibration import (
    calibration_table_from_error_budget,
)
from adapters.infrared_image_adapter import InfraredCameraConfig
from experiments.infrared_error_budget_audit import (
    DEFAULT_PROFILES,
    run_infrared_error_budget_audit,
)
from experiments.walker_raw_sensor_comparison import (
    _mean_modality_nis,
    _measurement_rmse,
    _position_rmse,
    _relative_position_rmse,
    _run,
    _walker_case,
    replace_infrared_messages_with_body_pair,
)


@dataclass(frozen=True)
class InfraredFilterCalibrationArm:
    name: str
    position_rmse_m: float
    relative_position_rmse_m: float
    infrared_measurement_rmse_deg: float
    infrared_mean_nis: float


@dataclass(frozen=True)
class InfraredFilterCalibrationShadowReport:
    duration_seconds: float
    dt_seconds: float
    seed: int
    calibration_seed: int
    calibration_sample_count: int
    profile: str
    default_filter_unchanged: bool
    arms: tuple[InfraredFilterCalibrationArm, ...]


def run_infrared_filter_calibration_shadow(
    *, duration=20.0, dt=2.0, seed=0, calibration_seed=701,
    calibration_samples=300,
):
    profile = next(
        item for item in DEFAULT_PROFILES if item.name == "moderate_combined"
    )
    config = InfraredCameraConfig(
        width=128, height=128,
        focal_length_x_pixels=600.0, focal_length_y_pixels=600.0,
        psf_sigma_pixels=1.2, source_peak=1000.0,
        background=10.0, read_noise_sigma=1.0,
        photon_noise_enabled=profile.photon_noise_enabled,
        thermal_background_drift_sigma=profile.thermal_background_drift_sigma,
        pointing_jitter_sigma_pixels=profile.pointing_jitter_sigma_pixels,
        radial_distortion_k1_per_pixel2=profile.radial_distortion_k1_per_pixel2,
        fixed_pixel_bias_x=profile.fixed_pixel_bias_x,
        fixed_pixel_bias_y=profile.fixed_pixel_bias_y,
    )
    calibration_report = run_infrared_error_budget_audit(
        seed=calibration_seed, monte_carlo_samples=calibration_samples,
        focal_lengths_pixels=(600.0,), peak_snrs=(1000.0,),
        profiles=(profile,),
    )
    empirical = calibration_table_from_error_budget(
        calibration_report, profile=profile.name,
    )
    corrected = type(empirical)(
        entries=empirical.entries, profile=empirical.profile,
        off_axis_radius_pixels=empirical.off_axis_radius_pixels,
        apply_bias_correction=True,
    )
    case = _walker_case(seed=seed, duration=duration, dt=dt)
    arms = []
    for name, table in (
        ("fixed_reported_covariance", None),
        ("empirical_covariance", empirical),
        ("bias_corrected_empirical_covariance", corrected),
    ):
        _, messages = replace_infrared_messages_with_body_pair(
            case["observations"], timestamps=case["timestamps"],
            truth_state_history_by_node=case["truth"], config=config,
            analytic_rng=np.random.default_rng(8100 + seed),
            image_rng=np.random.default_rng(9100 + seed),
            covariance_calibration=table,
        )
        history = _run(case, messages)
        arms.append(InfraredFilterCalibrationArm(
            name=name,
            position_rmse_m=_position_rmse(history, case["truth"]),
            relative_position_rmse_m=_relative_position_rmse(
                history, case["truth"], case["topology"],
            ),
            infrared_measurement_rmse_deg=float(np.rad2deg(
                _measurement_rmse(messages, case, "INFRARED")
            )),
            infrared_mean_nis=_mean_modality_nis(
                history, messages, "INFRARED",
            ),
        ))
    return InfraredFilterCalibrationShadowReport(
        duration_seconds=float(duration), dt_seconds=float(dt), seed=int(seed),
        calibration_seed=int(calibration_seed),
        calibration_sample_count=int(calibration_samples), profile=profile.name,
        default_filter_unchanged=True, arms=tuple(arms),
    )


def save_infrared_filter_calibration_shadow(report, output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "summary.json"
    path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path
