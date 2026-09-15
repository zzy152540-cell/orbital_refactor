"""Development sensitivity audit for optional infrared image-error models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path

import numpy as np

from adapters.infrared_image_adapter import (
    InfraredCameraConfig,
    infrared_frame_to_observation_message,
    render_infrared_point_source_frame,
)


@dataclass(frozen=True)
class InfraredErrorProfile:
    name: str
    photon_noise_enabled: bool = False
    thermal_background_drift_sigma: float = 0.0
    pointing_jitter_sigma_pixels: float = 0.0
    radial_distortion_k1_per_pixel2: float = 0.0
    fixed_pixel_bias_x: float = 0.0
    fixed_pixel_bias_y: float = 0.0


@dataclass(frozen=True)
class InfraredErrorBudgetRecord:
    profile: str
    focal_length_pixels: float
    peak_snr: float
    field_position: str
    monte_carlo_sample_count: int
    detection_fraction: float
    azimuth_bias_deg: float
    elevation_bias_deg: float
    angular_rmse_deg: float
    covariance_az_az_rad2: float
    covariance_az_el_rad2: float
    covariance_el_el_rad2: float
    angular_target_deg: float
    angular_target_met: bool


@dataclass(frozen=True)
class InfraredErrorBudgetReport:
    target_angular_rmse_deg: float
    model_scope: str
    calibration_warning: str
    profiles: tuple[InfraredErrorProfile, ...]
    records: tuple[InfraredErrorBudgetRecord, ...]


DEFAULT_PROFILES = (
    InfraredErrorProfile("simplified_baseline"),
    InfraredErrorProfile("photon_only", photon_noise_enabled=True),
    InfraredErrorProfile(
        "moderate_combined", photon_noise_enabled=True,
        thermal_background_drift_sigma=2.0,
        pointing_jitter_sigma_pixels=0.03,
        radial_distortion_k1_per_pixel2=1.0e-7,
        fixed_pixel_bias_x=0.02, fixed_pixel_bias_y=-0.02,
    ),
    InfraredErrorProfile(
        "stressed_combined", photon_noise_enabled=True,
        thermal_background_drift_sigma=5.0,
        pointing_jitter_sigma_pixels=0.10,
        radial_distortion_k1_per_pixel2=5.0e-7,
        fixed_pixel_bias_x=0.10, fixed_pixel_bias_y=-0.10,
    ),
)


def run_infrared_error_budget_audit(
    *, seed=0, monte_carlo_samples=200,
    focal_lengths_pixels=(300.0, 600.0), peak_snrs=(100.0, 1000.0),
    profiles=DEFAULT_PROFILES, target_angular_rmse_deg=0.00625,
):
    if monte_carlo_samples < 1:
        raise ValueError("monte_carlo_samples must be positive.")
    rng = np.random.default_rng(seed)
    field_positions = {"center": (0.0, 0.0), "off_axis": (0.08, 0.04)}
    records = []
    for profile in profiles:
        for focal in map(float, focal_lengths_pixels):
            for peak_snr in map(float, peak_snrs):
                for field_name, normalized_center in field_positions.items():
                    records.append(_evaluate_cell(
                        profile=profile, focal=focal, peak_snr=peak_snr,
                        field_name=field_name,
                        normalized_center=np.asarray(normalized_center),
                        samples=int(monte_carlo_samples), rng=rng,
                        target_deg=float(target_angular_rmse_deg),
                    ))
    return InfraredErrorBudgetReport(
        target_angular_rmse_deg=float(target_angular_rmse_deg),
        model_scope=(
            "Infrared focal-plane front end only; EKF, CI, optical and radar "
            "implementations are unchanged."
        ),
        calibration_warning=(
            "Development sensitivity profiles use assumed detector units and "
            "are not hardware-calibrated performance claims."
        ),
        profiles=tuple(profiles), records=tuple(records),
    )


def save_infrared_error_budget_audit(report, output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary, records = output / "summary.json", output / "records.csv"
    summary.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with records.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=tuple(InfraredErrorBudgetRecord.__dataclass_fields__)
        )
        writer.writeheader()
        writer.writerows(asdict(record) for record in report.records)
    return summary, records


def _evaluate_cell(
    *, profile, focal, peak_snr, field_name, normalized_center,
    samples, rng, target_deg,
):
    config = InfraredCameraConfig(
        width=128, height=128,
        focal_length_x_pixels=focal, focal_length_y_pixels=focal,
        psf_sigma_pixels=1.2, source_peak=peak_snr,
        background=10.0, read_noise_sigma=1.0,
        photon_noise_enabled=profile.photon_noise_enabled,
        thermal_background_drift_sigma=profile.thermal_background_drift_sigma,
        pointing_jitter_sigma_pixels=profile.pointing_jitter_sigma_pixels,
        radial_distortion_k1_per_pixel2=profile.radial_distortion_k1_per_pixel2,
        fixed_pixel_bias_x=profile.fixed_pixel_bias_x,
        fixed_pixel_bias_y=profile.fixed_pixel_bias_y,
    )
    residuals = []
    observer = np.zeros(6)
    for index in range(samples):
        subpixel = rng.uniform(-0.5, 0.5, size=2) / focal
        normalized = normalized_center + subpixel
        target = np.array([1.0e5, 1.0e5 * normalized[0],
                           1.0e5 * normalized[1], 0.0, 0.0, 0.0])
        frame = render_infrared_point_source_frame(
            timestamp=float(index), observer_id="observer", target_id="target",
            observer_state=observer, target_state=target,
            quaternion_i2b_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            config=config, rng=rng,
        )
        message = infrared_frame_to_observation_message(frame, config=config)
        if message.valid_flag:
            residuals.append(message.measurement - frame.ideal_az_el)
    detection = len(residuals) / samples
    if residuals:
        residuals = np.asarray(residuals)
        bias = np.rad2deg(np.mean(residuals, axis=0))
        rmse = float(np.rad2deg(np.sqrt(np.mean(np.sum(residuals**2, axis=1)))))
        covariance = (
            np.cov(residuals, rowvar=False, ddof=1)
            if len(residuals) > 1 else np.zeros((2, 2))
        )
    else:
        bias, rmse = np.full(2, np.nan), float("inf")
        covariance = np.full((2, 2), np.nan)
    return InfraredErrorBudgetRecord(
        profile=profile.name, focal_length_pixels=focal, peak_snr=peak_snr,
        field_position=field_name, monte_carlo_sample_count=samples,
        detection_fraction=float(detection), azimuth_bias_deg=float(bias[0]),
        elevation_bias_deg=float(bias[1]), angular_rmse_deg=rmse,
        covariance_az_az_rad2=float(covariance[0, 0]),
        covariance_az_el_rad2=float(covariance[0, 1]),
        covariance_el_el_rad2=float(covariance[1, 1]),
        angular_target_deg=target_deg,
        angular_target_met=bool(rmse <= target_deg and detection >= 0.99),
    )
