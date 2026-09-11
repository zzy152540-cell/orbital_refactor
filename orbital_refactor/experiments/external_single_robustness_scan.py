"""P08 sensitivity scan over outage duration and existing CI policies."""

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
class RobustnessScanRecord:
    seed: int
    missing_modality: str
    outage_duration_seconds: float
    ci_objective: str
    reset_feedback: bool
    reference_position_rmse_m: float
    outage_position_rmse_m: float
    position_rmse_increase_percent: float


@dataclass(frozen=True)
class RobustnessScanGroup:
    missing_modality: str
    outage_duration_seconds: float
    ci_objective: str
    reset_feedback: bool
    run_count: int
    mean_position_rmse_increase_percent: float
    worst_position_rmse_increase_percent: float
    threshold_met: bool


@dataclass(frozen=True)
class RobustnessScanReport:
    duration_seconds: float
    dt_seconds: float
    threshold_percent: float
    records: tuple[RobustnessScanRecord, ...]
    groups: tuple[RobustnessScanGroup, ...]


def run_external_single_robustness_scan(
    *, seeds=(0, 1, 2), duration=120.0, dt=2.0,
    outage_durations=(10.0, 20.0, 40.0, 60.0),
    ci_objectives=("trace", "logdet"), reset_feedback_modes=(True, False),
    threshold_percent=20.0,
):
    seeds = tuple(int(value) for value in seeds)
    outages = tuple(float(value) for value in outage_durations)
    objectives = tuple(str(value) for value in ci_objectives)
    feedback_modes = tuple(bool(value) for value in reset_feedback_modes)
    if not seeds or not outages or not objectives or not feedback_modes:
        raise ValueError("scan axes must be nonempty.")
    if any(value <= 0.0 or value > duration for value in outages):
        raise ValueError("outage durations must lie in (0, duration].")
    if set(objectives) - {"trace", "logdet"}:
        raise ValueError("CI objectives must be trace or logdet.")

    records = []
    for objective in objectives:
        for reset_feedback in feedback_modes:
            for seed in seeds:
                common = {
                    "duration": duration, "dt": dt, "seed": seed,
                    "outage_modalities": (), "enable_cann": False,
                    "filter_architecture": "federated_ci",
                    "observer_raan_deg": 15.5,
                    "ci_objective": objective,
                    "reset_feedback": reset_feedback,
                }
                reference = run_single_satellite_cann_comparison(**common)
                reference_rmse = float(reference["summary"]["position_rmse_m"])
                for outage_duration in outages:
                    start = 0.5 * (duration - outage_duration)
                    end = start + outage_duration
                    for modality in ("opt", "ir", "rad"):
                        degraded = run_single_satellite_cann_comparison(
                            **common,
                            outage_windows={modality: (start, end)},
                        )
                        degraded_rmse = float(
                            degraded["summary"]["position_rmse_m"]
                        )
                        records.append(RobustnessScanRecord(
                            seed=seed, missing_modality=modality,
                            outage_duration_seconds=outage_duration,
                            ci_objective=objective,
                            reset_feedback=reset_feedback,
                            reference_position_rmse_m=reference_rmse,
                            outage_position_rmse_m=degraded_rmse,
                            position_rmse_increase_percent=float(
                                100.0 * (degraded_rmse - reference_rmse)
                                / reference_rmse
                            ),
                        ))
    groups = tuple(_summarize(
        records, modality=modality, outage_duration=outage_duration,
        objective=objective, reset_feedback=reset_feedback,
        threshold=threshold_percent,
    ) for objective in objectives for reset_feedback in feedback_modes
      for outage_duration in outages for modality in ("opt", "ir", "rad"))
    return RobustnessScanReport(
        duration_seconds=float(duration), dt_seconds=float(dt),
        threshold_percent=float(threshold_percent),
        records=tuple(records), groups=groups,
    )


def save_external_single_robustness_scan(report, output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = output / "summary.json", output / "groups.csv"
    json_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=tuple(RobustnessScanGroup.__dataclass_fields__),
        )
        writer.writeheader()
        for group in report.groups:
            writer.writerow(asdict(group))
    return json_path, csv_path


def _summarize(
    records, *, modality, outage_duration, objective, reset_feedback, threshold,
):
    selected = [record for record in records if (
        record.missing_modality == modality
        and record.outage_duration_seconds == outage_duration
        and record.ci_objective == objective
        and record.reset_feedback == reset_feedback
    )]
    increases = np.asarray([
        record.position_rmse_increase_percent for record in selected
    ])
    mean = float(np.mean(increases))
    return RobustnessScanGroup(
        missing_modality=modality,
        outage_duration_seconds=outage_duration,
        ci_objective=objective, reset_feedback=reset_feedback,
        run_count=len(selected),
        mean_position_rmse_increase_percent=mean,
        worst_position_rmse_increase_percent=float(np.max(increases)),
        threshold_met=bool(mean <= threshold),
    )
