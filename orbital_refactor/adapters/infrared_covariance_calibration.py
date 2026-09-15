"""Opt-in empirical covariance lookup for infrared image observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InfraredCovarianceCalibrationEntry:
    focal_length_pixels: float
    peak_snr: float
    field_position: str
    covariance_rad2: np.ndarray
    bias_rad: np.ndarray

    def __post_init__(self):
        covariance = np.asarray(self.covariance_rad2, dtype=float).reshape(2, 2)
        covariance = 0.5 * (covariance + covariance.T)
        if not np.all(np.isfinite(covariance)):
            raise ValueError("Empirical infrared covariance must be finite.")
        if np.linalg.eigvalsh(covariance).min() < -1.0e-15:
            raise ValueError("Empirical infrared covariance must be PSD.")
        object.__setattr__(self, "covariance_rad2", covariance)
        bias = np.asarray(self.bias_rad, dtype=float).reshape(2)
        if not np.all(np.isfinite(bias)):
            raise ValueError("Empirical infrared bias must be finite.")
        object.__setattr__(self, "bias_rad", bias)


@dataclass(frozen=True)
class InfraredCovarianceCalibrationTable:
    entries: tuple[InfraredCovarianceCalibrationEntry, ...]
    profile: str
    off_axis_radius_pixels: float = 8.0
    apply_bias_correction: bool = False

    def __post_init__(self):
        if not self.entries:
            raise ValueError("Infrared covariance table cannot be empty.")
        if self.off_axis_radius_pixels <= 0.0:
            raise ValueError("off_axis_radius_pixels must be positive.")

    def lookup(self, *, config, pixel_xy, peak_snr):
        pixel = np.asarray(pixel_xy, dtype=float).reshape(2)
        principal = np.array([config.principal_x, config.principal_y])
        field = (
            "off_axis" if np.linalg.norm(pixel - principal)
            >= self.off_axis_radius_pixels else "center"
        )
        candidates = [entry for entry in self.entries
                      if entry.field_position == field] or list(self.entries)
        focal = np.sqrt(
            config.focal_length_x_pixels * config.focal_length_y_pixels
        )
        snr = max(float(peak_snr), 1.0e-12)
        selected = min(candidates, key=lambda entry: (
            abs(np.log(focal / entry.focal_length_pixels))
            + abs(np.log(snr / entry.peak_snr))
        ))
        return selected.covariance_rad2.copy(), selected.bias_rad.copy(), {
            "profile": self.profile,
            "field_position": field,
            "matched_focal_length_pixels": selected.focal_length_pixels,
            "matched_peak_snr": selected.peak_snr,
        }


def calibration_table_from_error_budget(report, *, profile):
    records = [record for record in report.records if record.profile == profile]
    if not records:
        raise ValueError(f"No infrared error-budget records for {profile!r}.")
    entries = tuple(InfraredCovarianceCalibrationEntry(
        focal_length_pixels=record.focal_length_pixels,
        peak_snr=record.peak_snr, field_position=record.field_position,
        covariance_rad2=np.array([
            [record.covariance_az_az_rad2, record.covariance_az_el_rad2],
            [record.covariance_az_el_rad2, record.covariance_el_el_rad2],
        ]),
        bias_rad=np.deg2rad(np.array([
            record.azimuth_bias_deg, record.elevation_bias_deg,
        ])),
    ) for record in records)
    return InfraredCovarianceCalibrationTable(
        entries=entries, profile=profile,
    )
