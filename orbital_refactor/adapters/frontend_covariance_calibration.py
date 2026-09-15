"""Opt-in empirical covariance tables for radar and optical front ends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrontendCovarianceCalibrationEntry:
    modality: str
    operating_point: str
    nominal_peak_snr: float
    covariance: np.ndarray
    bias: np.ndarray

    def __post_init__(self):
        modality = str(self.modality).upper()
        if modality not in {"RADAR", "OPTICAL"}:
            raise ValueError("Calibration modality must be RADAR or OPTICAL.")
        covariance = np.asarray(self.covariance, dtype=float).reshape(2, 2)
        covariance = 0.5 * (covariance + covariance.T)
        if not np.all(np.isfinite(covariance)):
            raise ValueError("Empirical covariance must be finite.")
        if np.linalg.eigvalsh(covariance).min() < -1.0e-15:
            raise ValueError("Empirical covariance must be PSD.")
        bias = np.asarray(self.bias, dtype=float).reshape(2)
        if not np.all(np.isfinite(bias)):
            raise ValueError("Empirical bias must be finite.")
        object.__setattr__(self, "modality", modality)
        if not np.isfinite(self.nominal_peak_snr) or self.nominal_peak_snr <= 0.0:
            raise ValueError("nominal_peak_snr must be finite and positive.")
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "bias", bias)


@dataclass(frozen=True)
class FrontendCovarianceCalibrationTable:
    entries: tuple[FrontendCovarianceCalibrationEntry, ...]
    modality: str
    profile: str
    off_center_radius_bins: float = 8.0
    apply_bias_correction: bool = False

    def __post_init__(self):
        modality = str(self.modality).upper()
        if not self.entries:
            raise ValueError("Frontend covariance table cannot be empty.")
        if any(entry.modality != modality for entry in self.entries):
            raise ValueError("Calibration entries must match the table modality.")
        if self.off_center_radius_bins <= 0.0:
            raise ValueError("off_center_radius_bins must be positive.")
        object.__setattr__(self, "modality", modality)

    def lookup(self, *, coordinate_xy, principal_xy, peak_snr):
        coordinate = np.asarray(coordinate_xy, dtype=float).reshape(2)
        principal = np.asarray(principal_xy, dtype=float).reshape(2)
        off_center = np.linalg.norm(coordinate - principal) >= self.off_center_radius_bins
        prefix = "off_axis" if self.modality == "OPTICAL" else "offset"
        preferred = prefix if off_center else "center"
        snr = max(float(peak_snr), 1.0e-12)
        selected = min(self.entries, key=lambda entry: (
            10.0 * (not entry.operating_point.startswith(preferred))
            + abs(np.log(snr / entry.nominal_peak_snr))
        ))
        return selected.covariance.copy(), selected.bias.copy(), {
            "profile": self.profile,
            "operating_point": selected.operating_point,
            "matched_peak_snr": selected.nominal_peak_snr,
            "off_center": bool(off_center),
        }


def calibration_table_from_error_budget(report, *, modality, profile):
    modality = str(modality).upper()
    records = [
        record for record in report.records
        if record.modality == modality and record.profile == profile
    ]
    if not records:
        raise ValueError(
            f"No {modality} error-budget records for profile {profile!r}."
        )
    entries = tuple(FrontendCovarianceCalibrationEntry(
        modality=modality, operating_point=record.operating_point,
        nominal_peak_snr=(
            1000.0 if "high_snr" in record.operating_point else 100.0
        ),
        covariance=np.array([
            [record.empirical_covariance_00, record.empirical_covariance_01],
            [record.empirical_covariance_01, record.empirical_covariance_11],
        ]),
        bias=np.array([record.component_0_bias, record.component_1_bias]),
    ) for record in records)
    return FrontendCovarianceCalibrationTable(
        entries=entries, modality=modality, profile=profile,
        off_center_radius_bins=8.0 if modality == "OPTICAL" else 1.0,
    )
