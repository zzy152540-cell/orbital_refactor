"""Cross-profile and crossed-operating-point calibration generalization audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
import json
from pathlib import Path

import numpy as np

from adapters.frontend_covariance_calibration import (
    calibration_table_from_error_budget,
)
from experiments.radar_optical_covariance_consistency_audit import (
    _validate_optical,
    _validate_radar,
)
from experiments.radar_optical_error_budget_audit import (
    DEFAULT_OPTICAL_PROFILES,
    DEFAULT_RADAR_PROFILES,
    run_radar_optical_error_budget_audit,
)


@dataclass(frozen=True)
class CalibrationGeneralizationRecord:
    modality: str
    validation_profile: str
    operating_point: str
    calibration_expected_applicable: bool
    validation_sample_count: int
    fixed_mean_nis: float
    calibrated_mean_nis: float
    calibrated_nis_95_exceedance_fraction: float
    mean_nis_distance_from_two: float


@dataclass(frozen=True)
class CalibrationGeneralizationReport:
    calibration_profile: str
    calibration_seed: int
    validation_seed: int
    calibration_sample_count: int
    validation_sample_count: int
    records: tuple[CalibrationGeneralizationRecord, ...]


def run_radar_optical_calibration_generalization_audit(
    *, calibration_seed=0, validation_seed=202,
    calibration_samples=300, validation_samples=200,
    calibration_profile="moderate_combined",
    validation_profiles=(
        "simplified_baseline", "moderate_combined", "stressed_combined",
    ),
):
    optical_calibration_profile = _find(
        DEFAULT_OPTICAL_PROFILES, calibration_profile,
    )
    radar_calibration_profile = _find(
        DEFAULT_RADAR_PROFILES, calibration_profile,
    )
    calibration = run_radar_optical_error_budget_audit(
        seed=calibration_seed, monte_carlo_samples=calibration_samples,
        optical_profiles=(optical_calibration_profile,),
        radar_profiles=(radar_calibration_profile,),
    )
    tables = {
        modality: calibration_table_from_error_budget(
            calibration, modality=modality, profile=calibration_profile,
        )
        for modality in ("OPTICAL", "RADAR")
    }
    corrected = {
        modality: replace(table, apply_bias_correction=True)
        for modality, table in tables.items()
    }
    rng = np.random.default_rng(validation_seed)
    records = []
    optical_points = (
        ("center_high_snr", np.array([0.0, 0.0]), 1000.0),
        ("center_low_snr", np.array([0.0, 0.0]), 100.0),
        ("off_axis_high_snr", np.array([0.08, 0.04]), 1000.0),
        ("off_axis_low_snr", np.array([0.08, 0.04]), 100.0),
    )
    radar_points = (
        ("center_high_snr", 1000.0, 0.0, 0.0),
        ("center_low_snr", 100.0, 0.0, 0.0),
        ("offset_high_snr", 1000.0, 12.0, 0.08),
        ("offset_low_snr", 100.0, 12.0, 0.08),
    )
    for profile_name in validation_profiles:
        optical_profile = _find(DEFAULT_OPTICAL_PROFILES, profile_name)
        radar_profile = _find(DEFAULT_RADAR_PROFILES, profile_name)
        for label, center, snr in optical_points:
            value = _validate_optical(
                optical_profile, label, center, snr, validation_samples, rng,
                tables["OPTICAL"], corrected["OPTICAL"],
            )
            records.append(_convert(
                profile_name, value,
                expected_applicable=profile_name == calibration_profile,
            ))
        for label, snr, range_offset, rate_offset in radar_points:
            value = _validate_radar(
                radar_profile, label, snr, range_offset, rate_offset,
                validation_samples, rng, tables["RADAR"], corrected["RADAR"],
            )
            records.append(_convert(
                profile_name, value,
                expected_applicable=profile_name == calibration_profile,
            ))
    return CalibrationGeneralizationReport(
        calibration_profile=calibration_profile,
        calibration_seed=int(calibration_seed), validation_seed=int(validation_seed),
        calibration_sample_count=int(calibration_samples),
        validation_sample_count=int(validation_samples), records=tuple(records),
    )


def save_radar_optical_calibration_generalization_audit(report, output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary, records = output / "summary.json", output / "records.csv"
    summary.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with records.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=tuple(CalibrationGeneralizationRecord.__dataclass_fields__)
        )
        writer.writeheader()
        writer.writerows(asdict(record) for record in report.records)
    return summary, records


def _find(profiles, name):
    selected = next((profile for profile in profiles if profile.name == name), None)
    if selected is None:
        raise ValueError(f"Unknown error profile {name!r}.")
    return selected


def _convert(profile_name, record, *, expected_applicable):
    mean_nis = record.bias_corrected_mean_nis
    return CalibrationGeneralizationRecord(
        modality=record.modality, validation_profile=profile_name,
        operating_point=record.operating_point,
        calibration_expected_applicable=bool(expected_applicable),
        validation_sample_count=record.validation_sample_count,
        fixed_mean_nis=record.fixed_mean_nis,
        calibrated_mean_nis=mean_nis,
        calibrated_nis_95_exceedance_fraction=(
            record.bias_corrected_nis_95_exceedance_fraction
        ),
        mean_nis_distance_from_two=abs(mean_nis - 2.0),
    )
