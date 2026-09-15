"""Paired multi-seed optical-outage shadow for infrared calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
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
    _position_rmse,
    _run,
    _walker_case,
    replace_infrared_messages_with_body_pair,
)


@dataclass(frozen=True)
class InfraredFilterOutageRun:
    seed: int
    arm: str
    reference_position_rmse_m: float
    optical_outage_position_rmse_m: float
    position_rmse_increase_percent: float
    reference_infrared_mean_nis: float
    optical_outage_infrared_mean_nis: float


@dataclass(frozen=True)
class InfraredFilterOutageGroup:
    arm: str
    run_count: int
    mean_position_rmse_increase_percent: float
    worst_position_rmse_increase_percent: float
    mean_reference_infrared_nis: float
    mean_optical_outage_infrared_nis: float


@dataclass(frozen=True)
class InfraredFilterOutageShadowReport:
    duration_seconds: float
    dt_seconds: float
    seeds: tuple[int, ...]
    profile: str
    calibration_sample_count: int
    paired_raw_images: bool
    ci_weight_available: bool
    runs: tuple[InfraredFilterOutageRun, ...]
    groups: tuple[InfraredFilterOutageGroup, ...]


def run_infrared_filter_outage_shadow(
    *, seeds=(0, 1, 2), duration=120.0, dt=2.0,
    calibration_seed=701, calibration_samples=300,
):
    seeds = tuple(map(int, seeds))
    profile = next(
        item for item in DEFAULT_PROFILES if item.name == "moderate_combined"
    )
    config = _camera_config(profile)
    calibration = run_infrared_error_budget_audit(
        seed=calibration_seed, monte_carlo_samples=calibration_samples,
        focal_lengths_pixels=(600.0,), peak_snrs=(1000.0,),
        profiles=(profile,),
    )
    empirical = calibration_table_from_error_budget(
        calibration, profile=profile.name,
    )
    corrected = type(empirical)(
        entries=empirical.entries, profile=empirical.profile,
        off_axis_radius_pixels=empirical.off_axis_radius_pixels,
        apply_bias_correction=True,
    )
    runs = []
    for seed in seeds:
        case = _walker_case(seed=seed, duration=duration, dt=dt)
        _, fixed_messages = replace_infrared_messages_with_body_pair(
            case["observations"], timestamps=case["timestamps"],
            truth_state_history_by_node=case["truth"], config=config,
            analytic_rng=np.random.default_rng(8100 + seed),
            image_rng=np.random.default_rng(9100 + seed),
        )
        for arm, table in (
            ("fixed_reported_covariance", None),
            ("empirical_covariance", empirical),
            ("bias_corrected_empirical_covariance", corrected),
        ):
            reference_messages = _apply_table(
                fixed_messages, table=table, config=config,
            )
            outage_messages = [replace(
                message,
                valid_flag=False if message.modality.upper() == "OPTICAL"
                else message.valid_flag,
            ) for message in reference_messages]
            reference_history = _run(case, reference_messages)
            outage_history = _run(case, outage_messages)
            reference_rmse = _position_rmse(reference_history, case["truth"])
            outage_rmse = _position_rmse(outage_history, case["truth"])
            runs.append(InfraredFilterOutageRun(
                seed=seed, arm=arm,
                reference_position_rmse_m=reference_rmse,
                optical_outage_position_rmse_m=outage_rmse,
                position_rmse_increase_percent=float(
                    100.0 * (outage_rmse - reference_rmse) / reference_rmse
                ),
                reference_infrared_mean_nis=_mean_modality_nis(
                    reference_history, reference_messages, "INFRARED",
                ),
                optical_outage_infrared_mean_nis=_mean_modality_nis(
                    outage_history, outage_messages, "INFRARED",
                ),
            ))
    groups = tuple(_summarize(runs, arm) for arm in (
        "fixed_reported_covariance", "empirical_covariance",
        "bias_corrected_empirical_covariance",
    ))
    return InfraredFilterOutageShadowReport(
        duration_seconds=float(duration), dt_seconds=float(dt), seeds=seeds,
        profile=profile.name, calibration_sample_count=int(calibration_samples),
        paired_raw_images=True,
        ci_weight_available=False,
        runs=tuple(runs), groups=groups,
    )


def save_infrared_filter_outage_shadow(report, output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary, runs = output / "summary.json", output / "runs.csv"
    summary.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with runs.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=tuple(InfraredFilterOutageRun.__dataclass_fields__)
        )
        writer.writeheader()
        writer.writerows(asdict(run) for run in report.runs)
    return summary, runs


def _camera_config(profile):
    return InfraredCameraConfig(
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


def _apply_table(messages, *, table, config):
    if table is None:
        return [replace(message) for message in messages]
    result = []
    for message in messages:
        if message.modality.upper() != "INFRARED" or not message.valid_flag:
            result.append(replace(message))
            continue
        covariance, bias, match = table.lookup(
            config=config,
            pixel_xy=message.metadata["centroid_pixel_xy"],
            peak_snr=message.metadata["peak_snr"],
        )
        result.append(replace(
            message,
            measurement=(message.measurement - bias
                         if table.apply_bias_correction else message.measurement.copy()),
            covariance=covariance,
            metadata={
                **message.metadata,
                "covariance_source": "infrared_empirical_table",
                "infrared_covariance_calibration": match,
                "infrared_bias_correction_applied": table.apply_bias_correction,
            },
        ))
    return result


def _summarize(runs, arm):
    selected = [run for run in runs if run.arm == arm]
    return InfraredFilterOutageGroup(
        arm=arm, run_count=len(selected),
        mean_position_rmse_increase_percent=float(np.mean([
            run.position_rmse_increase_percent for run in selected
        ])),
        worst_position_rmse_increase_percent=float(np.max([
            run.position_rmse_increase_percent for run in selected
        ])),
        mean_reference_infrared_nis=float(np.mean([
            run.reference_infrared_mean_nis for run in selected
        ])),
        mean_optical_outage_infrared_nis=float(np.mean([
            run.optical_outage_infrared_mean_nis for run in selected
        ])),
    )
