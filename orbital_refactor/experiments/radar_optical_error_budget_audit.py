"""Monte Carlo error-budget audits for radar and optical raw front ends."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path

import numpy as np

from adapters.optical_image_adapter import (
    OpticalCameraConfig,
    optical_frame_to_observation_message,
    render_optical_point_source_frame,
)
from adapters.radar_range_doppler_adapter import (
    RadarRangeDopplerConfig,
    radar_frame_to_observation_message,
    render_radar_range_doppler_frame,
)
from orbital_core.measurements import (
    measure_relative_range,
    measure_relative_range_rate,
)


@dataclass(frozen=True)
class OpticalErrorProfile:
    name: str
    photon_noise_enabled: bool = False
    stray_light_drift_sigma: float = 0.0
    pointing_jitter_sigma_pixels: float = 0.0
    radial_distortion_k1_per_pixel2: float = 0.0
    fixed_pixel_bias_x: float = 0.0
    fixed_pixel_bias_y: float = 0.0


@dataclass(frozen=True)
class RadarErrorProfile:
    name: str
    background_drift_sigma: float = 0.0
    echo_amplitude_sigma_fraction: float = 0.0
    range_bias_m: float = 0.0
    range_rate_bias_mps: float = 0.0
    range_jitter_sigma_m: float = 0.0
    range_rate_jitter_sigma_mps: float = 0.0


@dataclass(frozen=True)
class FrontendErrorBudgetRecord:
    modality: str
    profile: str
    operating_point: str
    sample_count: int
    detection_fraction: float
    component_0_bias: float
    component_1_bias: float
    component_0_rmse: float
    component_1_rmse: float
    joint_rmse: float
    empirical_covariance_00: float
    empirical_covariance_01: float
    empirical_covariance_11: float
    mean_reported_covariance_00: float
    mean_reported_covariance_11: float
    mean_nis: float


@dataclass(frozen=True)
class RadarOpticalErrorBudgetReport:
    model_scope: str
    calibration_warning: str
    optical_profiles: tuple[OpticalErrorProfile, ...]
    radar_profiles: tuple[RadarErrorProfile, ...]
    records: tuple[FrontendErrorBudgetRecord, ...]


DEFAULT_OPTICAL_PROFILES = (
    OpticalErrorProfile("simplified_baseline"),
    OpticalErrorProfile(
        "moderate_combined", photon_noise_enabled=True,
        stray_light_drift_sigma=1.0, pointing_jitter_sigma_pixels=0.03,
        radial_distortion_k1_per_pixel2=1.0e-7,
        fixed_pixel_bias_x=0.02, fixed_pixel_bias_y=-0.02,
    ),
    OpticalErrorProfile(
        "stressed_combined", photon_noise_enabled=True,
        stray_light_drift_sigma=3.0, pointing_jitter_sigma_pixels=0.10,
        radial_distortion_k1_per_pixel2=5.0e-7,
        fixed_pixel_bias_x=0.10, fixed_pixel_bias_y=-0.10,
    ),
)

DEFAULT_RADAR_PROFILES = (
    RadarErrorProfile("simplified_baseline"),
    RadarErrorProfile(
        "moderate_combined", background_drift_sigma=1.0,
        echo_amplitude_sigma_fraction=0.05, range_bias_m=0.5,
        range_rate_bias_mps=0.002, range_jitter_sigma_m=0.5,
        range_rate_jitter_sigma_mps=0.005,
    ),
    RadarErrorProfile(
        "stressed_combined", background_drift_sigma=3.0,
        echo_amplitude_sigma_fraction=0.20, range_bias_m=2.0,
        range_rate_bias_mps=0.01, range_jitter_sigma_m=2.0,
        range_rate_jitter_sigma_mps=0.02,
    ),
)


def run_radar_optical_error_budget_audit(
    *, seed=0, monte_carlo_samples=200,
    optical_profiles=DEFAULT_OPTICAL_PROFILES,
    radar_profiles=DEFAULT_RADAR_PROFILES,
):
    if monte_carlo_samples < 2:
        raise ValueError("monte_carlo_samples must be at least two.")
    rng = np.random.default_rng(seed)
    records = []
    for profile in optical_profiles:
        for label, normalized_uv in {
            "center_high_snr": (np.array([0.0, 0.0]), 1000.0),
            "center_low_snr": (np.array([0.0, 0.0]), 100.0),
            "off_axis_high_snr": (np.array([0.08, 0.04]), 1000.0),
            "off_axis_low_snr": (np.array([0.08, 0.04]), 100.0),
        }.items():
            records.append(_evaluate_optical(
                profile, label, normalized_uv[0], normalized_uv[1],
                monte_carlo_samples, rng,
            ))
    for profile in radar_profiles:
        for label, values in {
            "center_high_snr": (1000.0, 0.0, 0.0),
            "center_low_snr": (100.0, 0.0, 0.0),
            "offset_high_snr": (1000.0, 12.0, 0.08),
            "offset_low_snr": (100.0, 12.0, 0.08),
        }.items():
            records.append(_evaluate_radar(
                profile, label, values[0], values[1], values[2],
                monte_carlo_samples, rng,
            ))
    return RadarOpticalErrorBudgetReport(
        model_scope=(
            "Raw optical focal-plane and radar range-Doppler front ends; "
            "filter state, EKF updates and CI fusion are unchanged."
        ),
        calibration_warning=(
            "Profiles are engineering sensitivity assumptions, not "
            "hardware-qualified sensor specifications."
        ),
        optical_profiles=tuple(optical_profiles),
        radar_profiles=tuple(radar_profiles), records=tuple(records),
    )


def save_radar_optical_error_budget_audit(report, output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary = output / "summary.json"
    records = output / "records.csv"
    summary.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with records.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=tuple(FrontendErrorBudgetRecord.__dataclass_fields__)
        )
        writer.writeheader()
        writer.writerows(asdict(record) for record in report.records)
    return summary, records


def _summarize(modality, profile, operating_point, residuals, covariances, count):
    detection_fraction = len(residuals) / count
    if not residuals:
        values = np.full(2, np.nan)
        covariance = np.full((2, 2), np.nan)
        mean_covariance = np.full((2, 2), np.nan)
        rmse = values
        mean_nis = float("nan")
    else:
        residuals = np.asarray(residuals)
        covariances = np.asarray(covariances)
        values = np.mean(residuals, axis=0)
        rmse = np.sqrt(np.mean(residuals**2, axis=0))
        covariance = np.cov(residuals, rowvar=False, ddof=1)
        mean_covariance = np.mean(covariances, axis=0)
        mean_nis = float(np.mean([
            residual @ np.linalg.solve(reported, residual)
            for residual, reported in zip(residuals, covariances)
        ]))
    return FrontendErrorBudgetRecord(
        modality=modality, profile=profile.name,
        operating_point=operating_point, sample_count=count,
        detection_fraction=float(detection_fraction),
        component_0_bias=float(values[0]), component_1_bias=float(values[1]),
        component_0_rmse=float(rmse[0]), component_1_rmse=float(rmse[1]),
        joint_rmse=float(np.linalg.norm(rmse)),
        empirical_covariance_00=float(covariance[0, 0]),
        empirical_covariance_01=float(covariance[0, 1]),
        empirical_covariance_11=float(covariance[1, 1]),
        mean_reported_covariance_00=float(mean_covariance[0, 0]),
        mean_reported_covariance_11=float(mean_covariance[1, 1]),
        mean_nis=mean_nis,
    )


def _evaluate_optical(profile, label, normalized_uv, peak_snr, count, rng):
    config = OpticalCameraConfig(
        width=128, height=128, focal_length_x_pixels=400.0,
        focal_length_y_pixels=400.0, source_peak=peak_snr,
        background=10.0, read_noise_sigma=1.0,
        photon_noise_enabled=profile.photon_noise_enabled,
        stray_light_drift_sigma=profile.stray_light_drift_sigma,
        pointing_jitter_sigma_pixels=profile.pointing_jitter_sigma_pixels,
        radial_distortion_k1_per_pixel2=profile.radial_distortion_k1_per_pixel2,
        fixed_pixel_bias_x=profile.fixed_pixel_bias_x,
        fixed_pixel_bias_y=profile.fixed_pixel_bias_y,
    )
    observer = np.zeros(6)
    residuals, covariances = [], []
    for index in range(count):
        subpixel = rng.uniform(-0.5, 0.5, 2) / 400.0
        truth = normalized_uv + subpixel
        target = np.array([1.0e5, 1.0e5 * truth[0], 1.0e5 * truth[1], 0, 0, 0])
        frame = render_optical_point_source_frame(
            timestamp=index, observer_id="observer", target_id="target",
            observer_state=observer, target_state=target,
            quaternion_i2b_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            config=config, rng=rng,
        )
        message = optical_frame_to_observation_message(frame, config=config)
        if message.valid_flag:
            residuals.append(message.measurement - frame.ideal_normalized_uv)
            covariances.append(message.covariance)
    return _summarize("OPTICAL", profile, label, residuals, covariances, count)


def _evaluate_radar(profile, label, peak_snr, center_offset, rate_offset, count, rng):
    config = RadarRangeDopplerConfig(
        width=129, height=129, source_peak=peak_snr,
        background=10.0, read_noise_sigma=1.0,
        background_drift_sigma=profile.background_drift_sigma,
        echo_amplitude_sigma_fraction=profile.echo_amplitude_sigma_fraction,
        range_bias_m=profile.range_bias_m,
        range_rate_bias_mps=profile.range_rate_bias_mps,
        range_jitter_sigma_m=profile.range_jitter_sigma_m,
        range_rate_jitter_sigma_mps=profile.range_rate_jitter_sigma_mps,
    )
    observer = np.array([7e6, 0, 0, 0, 7500, 0], dtype=float)
    target = observer + np.array([1000, 100, -50, 1, -0.2, 0.1], dtype=float)
    truth = np.array([
        measure_relative_range(observer, target),
        measure_relative_range_rate(observer, target),
    ])
    residuals, covariances = [], []
    for index in range(count):
        frame = render_radar_range_doppler_frame(
            timestamp=index, observer_id="observer", target_id="target",
            observer_state=observer, target_state=target,
            acquisition_range_m=truth[0] + center_offset,
            acquisition_range_rate_mps=truth[1] + rate_offset,
            config=config, rng=rng,
        )
        message = radar_frame_to_observation_message(frame, config=config)
        if message.valid_flag:
            residuals.append(message.measurement - truth)
            covariances.append(message.covariance)
    return _summarize("RADAR", profile, label, residuals, covariances, count)
