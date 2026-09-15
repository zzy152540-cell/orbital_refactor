"""Raw-image feasibility audit for the infrared angular-precision target."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path

import numpy as np

from adapters.infrared_image_adapter import (
    InfraredCameraConfig,
    infrared_pixel_to_az_el,
)
from adapters.point_source_image import (
    extract_point_source_centroid,
    render_gaussian_point_source,
)
from experiments.single_modality_observability_audit import _build_scenario


@dataclass(frozen=True)
class InfraredImagePrecisionRecord:
    focal_length_pixels: float
    psf_sigma_pixels: float
    peak_snr: float
    monte_carlo_sample_count: int
    detection_fraction: float
    centroid_rmse_pixels: float
    angular_rmse_deg: float
    angular_target_deg: float
    angular_target_met: bool
    fixed_boresight_fov_coverage_fraction: float
    coverage_target_met: bool
    jointly_feasible: bool


@dataclass(frozen=True)
class InfraredImagePrecisionReport:
    duration_seconds: float
    dt_seconds: float
    frame_size_pixels: tuple[int, int]
    target_angular_rmse_deg: float
    target_fov_coverage_fraction: float
    boresight_definition: str
    centroid_monte_carlo_definition: str
    baseline_focal_length_pixels: float
    baseline_psf_sigma_pixels: float
    records: tuple[InfraredImagePrecisionRecord, ...]


def run_infrared_image_precision_audit(
    *, duration=120.0, dt=2.0, seed=0,
    focal_lengths_pixels=(300.0, 600.0, 1200.0, 2400.0),
    psf_sigmas_pixels=(0.8, 1.2), peak_snrs=(20.0, 100.0, 1000.0),
    monte_carlo_samples=200, target_angular_rmse_deg=0.00625,
    target_fov_coverage_fraction=0.95,
):
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt)
    scenario = _build_scenario(timestamps, 15.5)
    relative = scenario.relative_state_eci_by_node["sat_01"][:, :3]
    observer = scenario.observer_trajectories["sat_01"].state_history_eci
    normalized_track = _fixed_boresight_normalized_track(relative, observer[0])
    rng = np.random.default_rng(seed)
    records = []
    for focal in map(float, focal_lengths_pixels):
        for psf in map(float, psf_sigmas_pixels):
            for peak_snr in map(float, peak_snrs):
                if min(focal, psf, peak_snr) <= 0.0:
                    raise ValueError("All image scan axes must be positive.")
                centroid_rmse, angle_rmse, detection = _centroid_precision(
                    focal=focal, psf=psf, peak_snr=peak_snr,
                    samples=int(monte_carlo_samples), rng=rng,
                )
                coverage = _fov_coverage(
                    normalized_track, focal=focal, psf=psf,
                    width=512, height=512,
                )
                angular_met = angle_rmse <= target_angular_rmse_deg
                coverage_met = coverage >= target_fov_coverage_fraction
                records.append(InfraredImagePrecisionRecord(
                    focal_length_pixels=focal, psf_sigma_pixels=psf,
                    peak_snr=peak_snr,
                    monte_carlo_sample_count=int(monte_carlo_samples),
                    detection_fraction=detection,
                    centroid_rmse_pixels=centroid_rmse,
                    angular_rmse_deg=angle_rmse,
                    angular_target_deg=float(target_angular_rmse_deg),
                    angular_target_met=bool(angular_met),
                    fixed_boresight_fov_coverage_fraction=coverage,
                    coverage_target_met=bool(coverage_met),
                    jointly_feasible=bool(
                        angular_met and coverage_met and detection >= 0.99
                    ),
                ))
    return InfraredImagePrecisionReport(
        duration_seconds=float(duration), dt_seconds=float(dt),
        frame_size_pixels=(512, 512),
        target_angular_rmse_deg=float(target_angular_rmse_deg),
        target_fov_coverage_fraction=float(target_fov_coverage_fraction),
        boresight_definition=(
            "Fixed at the initial line of sight; coverage includes natural "
            "120-second relative angular drift."
        ),
        centroid_monte_carlo_definition=(
            "Existing Gaussian point-source renderer and centroid extractor "
            "on a local 32x32 detector patch with random subpixel phase."
        ),
        baseline_focal_length_pixels=300.0,
        baseline_psf_sigma_pixels=1.2,
        records=tuple(records),
    )


def save_infrared_image_precision_audit(report, output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary, records = output / "summary.json", output / "records.csv"
    summary.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with records.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=tuple(InfraredImagePrecisionRecord.__dataclass_fields__)
        )
        writer.writeheader()
        writer.writerows(asdict(record) for record in report.records)
    return summary, records


def _centroid_precision(*, focal, psf, peak_snr, samples, rng):
    if samples < 1:
        raise ValueError("monte_carlo_samples must be positive.")
    config = InfraredCameraConfig(
        width=32, height=32,
        focal_length_x_pixels=focal, focal_length_y_pixels=focal,
        psf_sigma_pixels=psf, source_peak=peak_snr,
        background=10.0, read_noise_sigma=1.0,
    )
    pixel_errors, angle_errors = [], []
    for _ in range(samples):
        ideal_pixel = np.array([config.principal_x, config.principal_y]) + (
            rng.uniform(-0.5, 0.5, size=2)
        )
        image, in_frame = render_gaussian_point_source(
            ideal_pixel, config=config, rng=rng
        )
        estimate, detected = extract_point_source_centroid(
            image, config=config, target_in_frame=in_frame
        )
        if not detected:
            continue
        pixel_errors.append(estimate - ideal_pixel)
        angle_errors.append(
            infrared_pixel_to_az_el(estimate, config)
            - infrared_pixel_to_az_el(ideal_pixel, config)
        )
    detection = len(pixel_errors) / samples
    if not pixel_errors:
        return float("inf"), float("inf"), detection
    pixel_norm = np.linalg.norm(np.asarray(pixel_errors), axis=1)
    angle_norm = np.linalg.norm(np.asarray(angle_errors), axis=1)
    return (
        float(np.sqrt(np.mean(pixel_norm**2))),
        float(np.rad2deg(np.sqrt(np.mean(angle_norm**2)))),
        float(detection),
    )


def _fixed_boresight_normalized_track(relative_positions, initial_observer_state):
    relative = np.asarray(relative_positions, dtype=float)
    observer = np.asarray(initial_observer_state, dtype=float).reshape(6)
    boresight = relative[0] / np.linalg.norm(relative[0])
    normal = np.cross(observer[:3], observer[3:])
    normal /= np.linalg.norm(normal)
    lateral = np.cross(normal, boresight)
    lateral /= np.linalg.norm(lateral)
    vertical = np.cross(boresight, lateral)
    components = relative @ np.column_stack((boresight, lateral, vertical))
    return components[:, 1:] / components[:, :1]


def _fov_coverage(track, *, focal, psf, width, height):
    pixels = np.asarray(track) * focal + np.array([
        0.5 * (width - 1), 0.5 * (height - 1),
    ])
    margin = 4.0 * psf
    valid = (
        (pixels[:, 0] >= margin) & (pixels[:, 0] <= width - 1 - margin)
        & (pixels[:, 1] >= margin) & (pixels[:, 1] <= height - 1 - margin)
    )
    return float(np.mean(valid))
