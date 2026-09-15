"""Multi-seed acceptance for aligned single/swarm raw measurement sources."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path

import numpy as np

from experiments.single_multi_measurement_source_shadow import (
    build_moderate_calibrated_measurement_source,
    run_single_multi_measurement_source_shadow,
)


@dataclass(frozen=True)
class MeasurementSourceAcceptanceSummary:
    duration_seconds: float
    dt_seconds: float
    seeds: tuple[int, ...]
    calibration_samples: int
    run_count: int
    all_single_modalities_observable: bool
    all_raw_frontend_nis_aligned: bool
    all_position_rmse_finite: bool
    walker_raw_mean_nis: dict[str, float]
    mean_position_rmse_m: dict[str, float]
    acceptance_passed: bool


def run_single_multi_measurement_source_acceptance(
    *, seeds=(0, 1, 2), duration=120.0, dt=2.0,
    calibration_seed=701, calibration_samples=300,
):
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if not normalized_seeds:
        raise ValueError("At least one validation seed is required.")
    sensors, calibrations = build_moderate_calibrated_measurement_source(
        calibration_seed=calibration_seed,
        calibration_samples=calibration_samples,
    )
    reports = tuple(
        run_single_multi_measurement_source_shadow(
            duration=duration, dt=dt, seed=seed, sensors=sensors,
            covariance_calibration_by_modality=calibrations,
        )
        for seed in normalized_seeds
    )
    rows = tuple(
        {"seed": report.seed, **asdict(arm)}
        for report in reports for arm in report.arms
    )
    walker_raw = tuple(
        row for row in rows
        if row["scope"] == "walker" and row["source"] == "raw_frontend"
    )
    nis_keys = {
        "radar": "radar_mean_nis",
        "infrared": "infrared_mean_nis",
        "optical": "optical_mean_nis",
    }
    mean_nis = {
        modality: float(np.mean([row[key] for row in walker_raw]))
        for modality, key in nis_keys.items()
    }
    arm_names = tuple(dict.fromkeys(
        f"{row['scope']}:{row['source']}" for row in rows
    ))
    mean_rmse = {
        name: float(np.mean([
            row["position_rmse_m"] for row in rows
            if f"{row['scope']}:{row['source']}" == name
        ]))
        for name in arm_names
    }
    observable = all(
        report.single_all_modalities_observable for report in reports
    )
    nis_aligned = all(
        report.raw_frontend_nis_alignment_ready for report in reports
    )
    finite = all(np.isfinite(row["position_rmse_m"]) for row in rows)
    summary = MeasurementSourceAcceptanceSummary(
        duration_seconds=float(duration), dt_seconds=float(dt),
        seeds=normalized_seeds, calibration_samples=int(calibration_samples),
        run_count=len(reports),
        all_single_modalities_observable=observable,
        all_raw_frontend_nis_aligned=nis_aligned,
        all_position_rmse_finite=finite,
        walker_raw_mean_nis=mean_nis,
        mean_position_rmse_m=mean_rmse,
        acceptance_passed=bool(observable and nis_aligned and finite),
    )
    return summary, rows


def save_single_multi_measurement_source_acceptance(
    summary, rows, output_directory,
):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    runs_path = output / "runs.csv"
    summary_path.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with runs_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary_path, runs_path
