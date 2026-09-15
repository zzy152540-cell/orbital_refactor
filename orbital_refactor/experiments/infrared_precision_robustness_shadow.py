"""Paired RMSE shadow scan for improved infrared angular precision."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path

import numpy as np

from experiments.single_satellite_cann_comparison import (
    run_single_satellite_cann_comparison,
)


@dataclass(frozen=True)
class InfraredPrecisionRun:
    seed: int
    infrared_angle_sigma_deg: float
    reference_position_rmse_m: float
    optical_outage_position_rmse_m: float
    position_rmse_increase_percent: float
    reference_mean_infrared_ci_weight: float
    optical_outage_mean_infrared_ci_weight: float


@dataclass(frozen=True)
class InfraredPrecisionGroup:
    infrared_angle_sigma_deg: float
    run_count: int
    mean_position_rmse_increase_percent: float
    position_increase_95_half_width_percent: float
    worst_position_rmse_increase_percent: float
    threshold_met: bool


@dataclass(frozen=True)
class InfraredPrecisionShadowReport:
    duration_seconds: float
    dt_seconds: float
    threshold_percent: float
    baseline_sigma_deg: float
    baseline_unchanged: bool
    runs: tuple[InfraredPrecisionRun, ...]
    groups: tuple[InfraredPrecisionGroup, ...]


def run_infrared_precision_robustness_shadow(
    *, seeds=(0, 1, 2, 3, 4), duration=120.0, dt=2.0,
    infrared_angle_sigmas_deg=(0.05, 0.025, 0.0125, 0.00625, 0.005),
    threshold_percent=20.0,
):
    seeds = tuple(map(int, seeds))
    sigmas = tuple(map(float, infrared_angle_sigmas_deg))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique.")
    if not sigmas or min(sigmas) <= 0.0:
        raise ValueError("infrared_angle_sigmas_deg must be positive.")
    runs = []
    for sigma in sigmas:
        for seed in seeds:
            common = {
                "duration": duration, "dt": dt, "seed": seed,
                "enable_cann": False, "filter_architecture": "federated_ci",
                "observer_raan_deg": 15.5,
                "infrared_angle_sigma_deg": sigma,
            }
            reference = run_single_satellite_cann_comparison(
                **common, outage_modalities=(),
            )
            outage = run_single_satellite_cann_comparison(
                **common, outage_modalities=(),
                outage_windows={"opt": (0.0, duration)},
            )
            reference_rmse = float(reference["summary"]["position_rmse_m"])
            outage_rmse = float(outage["summary"]["position_rmse_m"])
            runs.append(InfraredPrecisionRun(
                seed=seed, infrared_angle_sigma_deg=sigma,
                reference_position_rmse_m=reference_rmse,
                optical_outage_position_rmse_m=outage_rmse,
                position_rmse_increase_percent=float(
                    100.0 * (outage_rmse - reference_rmse) / reference_rmse
                ),
                reference_mean_infrared_ci_weight=_mean_weight(
                    reference["ci_weight_history"], "ir"
                ),
                optical_outage_mean_infrared_ci_weight=_mean_weight(
                    outage["ci_weight_history"], "ir"
                ),
            ))
    groups = tuple(
        _summarize(runs, sigma=sigma, threshold=threshold_percent)
        for sigma in sigmas
    )
    return InfraredPrecisionShadowReport(
        duration_seconds=float(duration), dt_seconds=float(dt),
        threshold_percent=float(threshold_percent), baseline_sigma_deg=0.05,
        baseline_unchanged=True, runs=tuple(runs), groups=groups,
    )


def save_infrared_precision_robustness_shadow(report, output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary, groups = output / "summary.json", output / "groups.csv"
    summary.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with groups.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=tuple(InfraredPrecisionGroup.__dataclass_fields__)
        )
        writer.writeheader()
        writer.writerows(asdict(group) for group in report.groups)
    return summary, groups


def _mean_weight(history, modality):
    values = [
        float(weights.get(modality, 0.0))
        for weights in history or () if weights
    ]
    return float(np.mean(values)) if values else 0.0


def _summarize(runs, *, sigma, threshold):
    selected = [run for run in runs if run.infrared_angle_sigma_deg == sigma]
    increases = np.asarray([
        run.position_rmse_increase_percent for run in selected
    ])
    half_width = (
        0.0 if len(selected) == 1 else
        1.96 * float(np.std(increases, ddof=1)) / np.sqrt(len(selected))
    )
    mean = float(np.mean(increases))
    return InfraredPrecisionGroup(
        infrared_angle_sigma_deg=sigma, run_count=len(selected),
        mean_position_rmse_increase_percent=mean,
        position_increase_95_half_width_percent=half_width,
        worst_position_rmse_increase_percent=float(np.max(increases)),
        threshold_met=bool(mean <= threshold),
    )
