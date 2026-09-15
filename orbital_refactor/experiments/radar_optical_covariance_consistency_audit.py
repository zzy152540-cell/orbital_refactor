"""Held-out covariance-consistency audit for radar and optical front ends."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
import json
from pathlib import Path

import numpy as np

from adapters.frontend_covariance_calibration import (
    calibration_table_from_error_budget,
)
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
from experiments.radar_optical_error_budget_audit import (
    DEFAULT_OPTICAL_PROFILES,
    DEFAULT_RADAR_PROFILES,
    run_radar_optical_error_budget_audit,
)
from orbital_core.measurements import (
    measure_relative_range,
    measure_relative_range_rate,
)


@dataclass(frozen=True)
class FrontendCovarianceConsistencyRecord:
    modality: str
    operating_point: str
    validation_sample_count: int
    fixed_mean_nis: float
    empirical_mean_nis: float
    bias_corrected_mean_nis: float
    fixed_nis_95_exceedance_fraction: float
    empirical_nis_95_exceedance_fraction: float
    bias_corrected_nis_95_exceedance_fraction: float


@dataclass(frozen=True)
class RadarOpticalCovarianceConsistencyReport:
    profile: str
    calibration_seed: int
    validation_seed: int
    calibration_sample_count: int
    validation_sample_count: int
    chi_square_2d_95_threshold: float
    interpretation: str
    records: tuple[FrontendCovarianceConsistencyRecord, ...]


def run_radar_optical_covariance_consistency_audit(
    *, calibration_seed=0, validation_seed=101,
    calibration_samples=300, validation_samples=300,
    profile_name="moderate_combined",
):
    optical_profile = _profile(DEFAULT_OPTICAL_PROFILES, profile_name, "optical")
    radar_profile = _profile(DEFAULT_RADAR_PROFILES, profile_name, "radar")
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
    rng = np.random.default_rng(validation_seed)
    records = [
        _validate_optical(
            optical_profile, "center_high_snr", np.array([0.0, 0.0]),
            1000.0, validation_samples, rng, tables["OPTICAL"],
            corrected["OPTICAL"],
        ),
        _validate_optical(
            optical_profile, "off_axis_low_snr", np.array([0.08, 0.04]),
            100.0, validation_samples, rng, tables["OPTICAL"],
            corrected["OPTICAL"],
        ),
        _validate_radar(
            radar_profile, "center_high_snr", 1000.0, 0.0, 0.0,
            validation_samples, rng, tables["RADAR"], corrected["RADAR"],
        ),
        _validate_radar(
            radar_profile, "offset_low_snr", 100.0, 12.0, 0.08,
            validation_samples, rng, tables["RADAR"], corrected["RADAR"],
        ),
    ]
    return RadarOpticalCovarianceConsistencyReport(
        profile=profile_name, calibration_seed=int(calibration_seed),
        validation_seed=int(validation_seed),
        calibration_sample_count=int(calibration_samples),
        validation_sample_count=int(validation_samples),
        chi_square_2d_95_threshold=5.991,
        interpretation=(
            "Held-out two-dimensional NIS should have mean near 2 and "
            "95-percent threshold exceedance near 0.05. Bias correction is "
            "reported separately and never enabled implicitly."
        ), records=tuple(records),
    )


def save_radar_optical_covariance_consistency_audit(report, output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary, records = output / "summary.json", output / "records.csv"
    summary.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with records.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=tuple(FrontendCovarianceConsistencyRecord.__dataclass_fields__),
        )
        writer.writeheader()
        writer.writerows(asdict(record) for record in report.records)
    return summary, records


def _profile(profiles, name, modality):
    selected = next((item for item in profiles if item.name == name), None)
    if selected is None:
        raise ValueError(f"Unknown {modality} profile {name!r}.")
    return selected


def _record(modality, operating_point, fixed, empirical, corrected):
    threshold = 5.991
    return FrontendCovarianceConsistencyRecord(
        modality=modality, operating_point=operating_point,
        validation_sample_count=len(fixed),
        fixed_mean_nis=float(np.mean(fixed)),
        empirical_mean_nis=float(np.mean(empirical)),
        bias_corrected_mean_nis=float(np.mean(corrected)),
        fixed_nis_95_exceedance_fraction=float(np.mean(np.asarray(fixed) > threshold)),
        empirical_nis_95_exceedance_fraction=float(
            np.mean(np.asarray(empirical) > threshold)
        ),
        bias_corrected_nis_95_exceedance_fraction=float(
            np.mean(np.asarray(corrected) > threshold)
        ),
    )


def _append_nis(frame, truth, fixed, empirical, corrected, output):
    if not fixed.valid_flag:
        return
    for message, bucket in zip((fixed, empirical, corrected), output):
        residual = message.measurement - truth
        bucket.append(float(residual @ np.linalg.solve(message.covariance, residual)))


def _validate_optical(profile, label, center, peak_snr, count, rng, table, corrected):
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
    values = ([], [], [])
    for index in range(count):
        truth = center + rng.uniform(-0.5, 0.5, 2) / 400.0
        target = np.array([1e5, 1e5 * truth[0], 1e5 * truth[1], 0, 0, 0])
        frame = render_optical_point_source_frame(
            timestamp=index, observer_id="observer", target_id="target",
            observer_state=observer, target_state=target,
            quaternion_i2b_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            config=config, rng=rng,
        )
        messages = (
            optical_frame_to_observation_message(frame, config=config),
            optical_frame_to_observation_message(
                frame, config=config, covariance_calibration=table,
            ),
            optical_frame_to_observation_message(
                frame, config=config, covariance_calibration=corrected,
            ),
        )
        _append_nis(frame, frame.ideal_normalized_uv, *messages, values)
    return _record("OPTICAL", label, *values)


def _validate_radar(profile, label, peak_snr, range_offset, rate_offset,
                    count, rng, table, corrected):
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
    values = ([], [], [])
    for index in range(count):
        frame = render_radar_range_doppler_frame(
            timestamp=index, observer_id="observer", target_id="target",
            observer_state=observer, target_state=target,
            acquisition_range_m=truth[0] + range_offset,
            acquisition_range_rate_mps=truth[1] + rate_offset,
            config=config, rng=rng,
        )
        messages = (
            radar_frame_to_observation_message(frame, config=config),
            radar_frame_to_observation_message(
                frame, config=config, covariance_calibration=table,
            ),
            radar_frame_to_observation_message(
                frame, config=config, covariance_calibration=corrected,
            ),
        )
        _append_nis(frame, truth, *messages, values)
    return _record("RADAR", label, *values)
