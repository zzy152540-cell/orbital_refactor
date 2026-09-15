"""Held-out NIS audit for opt-in infrared empirical covariance calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path

import numpy as np

from adapters.infrared_covariance_calibration import (
    calibration_table_from_error_budget,
)
from adapters.infrared_image_adapter import (
    InfraredCameraConfig,
    infrared_frame_to_observation_message,
    render_infrared_point_source_frame,
)
from experiments.infrared_error_budget_audit import (
    DEFAULT_PROFILES,
    run_infrared_error_budget_audit,
)


@dataclass(frozen=True)
class InfraredCovarianceConsistencyRecord:
    focal_length_pixels: float
    peak_snr: float
    field_position: str
    validation_sample_count: int
    angular_rmse_deg: float
    fixed_mean_nis: float
    empirical_mean_nis: float
    bias_corrected_mean_nis: float
    fixed_nis_95_exceedance_fraction: float
    empirical_nis_95_exceedance_fraction: float
    bias_corrected_nis_95_exceedance_fraction: float


@dataclass(frozen=True)
class InfraredCovarianceConsistencyReport:
    profile: str
    calibration_seed: int
    validation_seed: int
    calibration_sample_count: int
    validation_sample_count: int
    chi_square_2d_95_threshold: float
    interpretation: str
    records: tuple[InfraredCovarianceConsistencyRecord, ...]


def run_infrared_covariance_consistency_audit(
    *, calibration_seed=0, validation_seed=101,
    calibration_samples=300, validation_samples=300,
    focal_lengths_pixels=(300.0, 600.0), peak_snrs=(100.0, 1000.0),
    profile_name="moderate_combined",
):
    profile = next(
        (item for item in DEFAULT_PROFILES if item.name == profile_name), None
    )
    if profile is None:
        raise ValueError(f"Unknown infrared profile {profile_name!r}.")
    calibration = run_infrared_error_budget_audit(
        seed=calibration_seed, monte_carlo_samples=calibration_samples,
        focal_lengths_pixels=focal_lengths_pixels, peak_snrs=peak_snrs,
        profiles=(profile,),
    )
    table = calibration_table_from_error_budget(
        calibration, profile=profile_name,
    )
    corrected_table = type(table)(
        entries=table.entries, profile=table.profile,
        off_axis_radius_pixels=table.off_axis_radius_pixels,
        apply_bias_correction=True,
    )
    rng = np.random.default_rng(validation_seed)
    records = []
    for focal in map(float, focal_lengths_pixels):
        for peak_snr in map(float, peak_snrs):
            for field_name, center in (
                ("center", np.array([0.0, 0.0])),
                ("off_axis", np.array([0.08, 0.04])),
            ):
                records.append(_validate_cell(
                        table=table, corrected_table=corrected_table,
                        profile=profile, focal=focal,
                    peak_snr=peak_snr, field_name=field_name, center=center,
                    samples=int(validation_samples), rng=rng,
                ))
    return InfraredCovarianceConsistencyReport(
        profile=profile_name, calibration_seed=int(calibration_seed),
        validation_seed=int(validation_seed),
        calibration_sample_count=int(calibration_samples),
        validation_sample_count=int(validation_samples),
        chi_square_2d_95_threshold=5.991,
        interpretation=(
            "NIS uses held-out image residuals. A mean near 2 and a 95-percent "
            "threshold exceedance near 0.05 indicate approximate consistency; "
            "fixed bias is deliberately not removed."
        ), records=tuple(records),
    )


def save_infrared_covariance_consistency_audit(report, output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary, records = output / "summary.json", output / "records.csv"
    summary.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with records.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=tuple(InfraredCovarianceConsistencyRecord.__dataclass_fields__),
        )
        writer.writeheader()
        writer.writerows(asdict(record) for record in report.records)
    return summary, records


def _validate_cell(*, table, corrected_table, profile, focal, peak_snr,
                   field_name, center,
                   samples, rng):
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
    residuals, fixed_nis, empirical_nis, corrected_nis = [], [], [], []
    observer = np.zeros(6)
    for index in range(samples):
        normalized = center + rng.uniform(-0.5, 0.5, size=2) / focal
        target = np.array([1.0e5, 1.0e5 * normalized[0],
                           1.0e5 * normalized[1], 0.0, 0.0, 0.0])
        frame = render_infrared_point_source_frame(
            timestamp=index, observer_id="observer", target_id="target",
            observer_state=observer, target_state=target,
            quaternion_i2b_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            config=config, rng=rng,
        )
        fixed = infrared_frame_to_observation_message(frame, config=config)
        empirical = infrared_frame_to_observation_message(
            frame, config=config, covariance_calibration=table,
        )
        corrected = infrared_frame_to_observation_message(
            frame, config=config, covariance_calibration=corrected_table,
        )
        if not fixed.valid_flag:
            continue
        residual = fixed.measurement - frame.ideal_az_el
        residuals.append(residual)
        fixed_nis.append(float(residual @ np.linalg.solve(fixed.covariance, residual)))
        empirical_nis.append(float(
            residual @ np.linalg.solve(empirical.covariance, residual)
        ))
        corrected_residual = corrected.measurement - frame.ideal_az_el
        corrected_nis.append(float(
            corrected_residual @ np.linalg.solve(
                corrected.covariance, corrected_residual
            )
        ))
    residuals = np.asarray(residuals)
    threshold = 5.991
    angular_rmse = float(np.rad2deg(np.sqrt(
        np.mean(np.sum(residuals**2, axis=1))
    )))
    return InfraredCovarianceConsistencyRecord(
        focal_length_pixels=focal, peak_snr=peak_snr,
        field_position=field_name, validation_sample_count=len(residuals),
        angular_rmse_deg=angular_rmse,
        fixed_mean_nis=float(np.mean(fixed_nis)),
        empirical_mean_nis=float(np.mean(empirical_nis)),
        bias_corrected_mean_nis=float(np.mean(corrected_nis)),
        fixed_nis_95_exceedance_fraction=float(np.mean(np.asarray(fixed_nis) > threshold)),
        empirical_nis_95_exceedance_fraction=float(
            np.mean(np.asarray(empirical_nis) > threshold)
        ),
        bias_corrected_nis_95_exceedance_fraction=float(
            np.mean(np.asarray(corrected_nis) > threshold)
        ),
    )
