"""Truth-tracking upper bound versus lagged-posterior infrared boresight."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path

import numpy as np

from adapters.infrared_covariance_calibration import calibration_table_from_error_budget
from experiments.infrared_error_budget_audit import DEFAULT_PROFILES, run_infrared_error_budget_audit
from experiments.infrared_filter_outage_shadow import _camera_config
from experiments.infrared_precision_robustness_shadow import _mean_weight
from experiments.single_satellite_cann_comparison import run_single_satellite_cann_comparison


@dataclass(frozen=True)
class InfraredBoresightRun:
    seed: int
    mode: str
    reference_position_rmse_m: float
    optical_outage_position_rmse_m: float
    position_rmse_increase_percent: float
    reference_valid_fraction: float
    outage_valid_fraction: float
    reference_boresight_mean_error_deg: float
    outage_boresight_mean_error_deg: float
    outage_boresight_max_error_deg: float
    outage_infrared_mean_nis: float
    outage_infrared_ci_weight: float
    reference_rmse_first_third_m: float
    reference_rmse_last_third_m: float
    outage_rmse_first_third_m: float
    outage_rmse_last_third_m: float
    reference_frontend_diagnostics: dict
    outage_frontend_diagnostics: dict
    pre_outage_state_max_abs_difference: float


@dataclass(frozen=True)
class InfraredBoresightReport:
    duration_seconds: float
    dt_seconds: float
    seeds: tuple[int, ...]
    calibration_profile: str
    optical_outage_start_seconds: float
    lagged_definition: str
    runs: tuple[InfraredBoresightRun, ...]


def run_infrared_boresight_shadow(
    *, seeds=(0, 1, 2), duration=120.0, dt=2.0,
    calibration_seed=701, calibration_samples=300,
    optical_outage_start=0.0,
):
    profile = next(x for x in DEFAULT_PROFILES if x.name == "moderate_combined")
    config = _camera_config(profile)
    calibration = run_infrared_error_budget_audit(
        seed=calibration_seed, monte_carlo_samples=calibration_samples,
        focal_lengths_pixels=(600.0,), peak_snrs=(1000.0,), profiles=(profile,),
    )
    table = calibration_table_from_error_budget(calibration, profile=profile.name)
    table = type(table)(
        entries=table.entries, profile=table.profile,
        off_axis_radius_pixels=table.off_axis_radius_pixels,
        apply_bias_correction=True,
    )
    runs = []
    for seed in map(int, seeds):
        tracker_common = dict(
            duration=duration, dt=dt, seed=seed, enable_cann=False,
            filter_architecture="federated_ci", observer_raan_deg=15.5,
        )
        reference_tracker = run_single_satellite_cann_comparison(
            **tracker_common, outage_modalities=(),
        )
        outage_tracker = run_single_satellite_cann_comparison(
            **tracker_common, outage_modalities=(),
            outage_windows={"opt": (optical_outage_start, duration)},
        )
        reference_boresight = _lagged_boresight(reference_tracker, dt)
        outage_boresight = _lagged_boresight(outage_tracker, dt)
        for mode, reference_pointing, outage_pointing in (
            ("exact_center_truth_tracking_diagnostic", None, None),
            ("lagged_posterior_prediction", reference_boresight, outage_boresight),
        ):
            common = dict(
                duration=duration, dt=dt, seed=seed, enable_cann=False,
                filter_architecture="federated_ci", observer_raan_deg=15.5,
                infrared_image_config=config,
                infrared_covariance_calibration=table,
            )
            reference = run_single_satellite_cann_comparison(
                **common, outage_modalities=(),
                infrared_boresight_spri_history=reference_pointing,
            )
            outage = run_single_satellite_cann_comparison(
                **common, outage_modalities=(),
                outage_windows={"opt": (optical_outage_start, duration)},
                infrared_boresight_spri_history=outage_pointing,
            )
            reference_rmse = reference["summary"]["position_rmse_m"]
            outage_rmse = outage["summary"]["position_rmse_m"]
            runs.append(InfraredBoresightRun(
                seed=seed, mode=mode,
                reference_position_rmse_m=reference_rmse,
                optical_outage_position_rmse_m=outage_rmse,
                position_rmse_increase_percent=float(
                    100.0 * (outage_rmse - reference_rmse) / reference_rmse
                ),
                reference_valid_fraction=(
                    reference["summary"]["infrared_valid_count"]
                    / len(reference["timestamps"])
                ),
                outage_valid_fraction=(
                    outage["summary"]["infrared_valid_count"]
                    / len(outage["timestamps"])
                ),
                reference_boresight_mean_error_deg=reference["summary"][
                    "infrared_boresight_error_mean_deg"
                ],
                outage_boresight_mean_error_deg=outage["summary"][
                    "infrared_boresight_error_mean_deg"
                ],
                outage_boresight_max_error_deg=outage["summary"][
                    "infrared_boresight_error_max_deg"
                ],
                outage_infrared_mean_nis=_finite_mean(outage["nis_by_modality"]["ir"]),
                outage_infrared_ci_weight=_mean_weight(outage["ci_weight_history"], "ir"),
                reference_rmse_first_third_m=_third_rmse(
                    reference["position_error_m"], first=True,
                ),
                reference_rmse_last_third_m=_third_rmse(
                    reference["position_error_m"], first=False,
                ),
                outage_rmse_first_third_m=_third_rmse(
                    outage["position_error_m"], first=True,
                ),
                outage_rmse_last_third_m=_third_rmse(
                    outage["position_error_m"], first=False,
                ),
                reference_frontend_diagnostics=reference["summary"][
                    "infrared_frontend_diagnostics"
                ],
                outage_frontend_diagnostics=outage["summary"][
                    "infrared_frontend_diagnostics"
                ],
                pre_outage_state_max_abs_difference=_pre_outage_difference(
                    reference, outage, start=optical_outage_start,
                ),
            ))
    return InfraredBoresightReport(
        duration_seconds=float(duration), dt_seconds=float(dt),
        seeds=tuple(map(int, seeds)), calibration_profile=profile.name,
        optical_outage_start_seconds=float(optical_outage_start),
        lagged_definition=(
            "At k>0, propagate the corresponding analytic tracker's posterior "
            "from k-1 with constant velocity; k=0 uses the configured initial error."
        ), runs=tuple(runs),
    )


def save_infrared_boresight_shadow(report, output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary, runs = output / "summary.json", output / "runs.csv"
    summary.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    with runs.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(InfraredBoresightRun.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(run) for run in report.runs)
    return summary, runs


def _lagged_boresight(result, dt):
    estimate = np.asarray(result["estimated_relative_state_history_spri"], dtype=float)
    truth = np.asarray(result["truth_relative_state_history_spri"], dtype=float)
    boresight = np.empty((len(estimate), 3))
    boresight[0] = truth[0, :3] + np.array([50.0, -40.0, 30.0])
    boresight[1:] = estimate[:-1, :3] + float(dt) * estimate[:-1, 3:]
    return boresight


def _finite_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def _third_rmse(values, *, first):
    values = np.asarray(values, dtype=float)
    width = max(1, len(values) // 3)
    selected = values[:width] if first else values[-width:]
    return float(np.sqrt(np.mean(selected**2)))


def _pre_outage_difference(reference, outage, *, start):
    mask = np.asarray(reference["timestamps"], dtype=float) < float(start)
    if not np.any(mask):
        return 0.0
    left = np.asarray(reference["estimated_relative_state_history_spri"])[mask]
    right = np.asarray(outage["estimated_relative_state_history_spri"])[mask]
    return float(np.max(np.abs(left - right)))
