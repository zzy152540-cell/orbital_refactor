"""Audit where measurement-side CANN helps the P08 scenario."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from experiments.single_satellite_three_modal_cann_feedback import (
    run_single_satellite_three_modal_cann_feedback,
    run_staggered_recovery_fault_comparison,
)


@dataclass(frozen=True)
class CANNRobustnessAuditRecord:
    seed: int
    missing_modality: str
    outage_duration_seconds: float
    pure_outage_error_change_with_cann_m: float
    pure_outage_recovery_change_with_cann_m: float
    recovery_fault_cost_without_cann_m: float
    recovery_fault_cost_with_cann_m: float
    recovery_fault_mitigation_m: float


@dataclass(frozen=True)
class CANNRobustnessAuditReport:
    duration_seconds: float
    dt_seconds: float
    records: tuple[CANNRobustnessAuditRecord, ...]
    conclusion: str


def run_external_single_cann_robustness_audit(
    *, seeds=(0, 1, 2), modalities=("opt", "ir", "rad"),
    duration=120.0, dt=2.0, outage_window=(30.0, 90.0),
    recovery_fault_samples=2, recovery_horizon=28.0,
):
    records = []
    outage_duration = float(outage_window[1] - outage_window[0])
    for seed in seeds:
        for modality in modalities:
            windows = {str(modality): tuple(map(float, outage_window))}
            pure = run_single_satellite_three_modal_cann_feedback(
                inject_faults=False, duration=duration, dt=dt, seed=int(seed),
                outage_windows=windows, observer_raan_deg=15.5,
            )
            faulty = run_staggered_recovery_fault_comparison(
                duration=duration, dt=dt, seed=int(seed),
                outage_windows=windows,
                recovery_fault_samples=recovery_fault_samples,
                recovery_horizon=recovery_horizon, observer_raan_deg=15.5,
            )
            impact = faulty["summary"]["recovery_fault_impact"][str(modality)]
            without = float(impact["baseline"]["net_position_rmse_change_m"])
            with_cann = float(impact["processed"]["net_position_rmse_change_m"])
            records.append(CANNRobustnessAuditRecord(
                seed=int(seed), missing_modality=str(modality),
                outage_duration_seconds=outage_duration,
                pure_outage_error_change_with_cann_m=float(
                    pure["summary"]["outage_position_rmse_change_m"]
                ),
                pure_outage_recovery_change_with_cann_m=float(
                    pure["summary"]["recovery_position_rmse_change_m"]
                ),
                recovery_fault_cost_without_cann_m=without,
                recovery_fault_cost_with_cann_m=with_cann,
                recovery_fault_mitigation_m=without - with_cann,
            ))
    return CANNRobustnessAuditReport(
        duration_seconds=float(duration), dt_seconds=float(dt),
        records=tuple(records),
        conclusion=(
            "Measurement-side CANN does not synthesize unavailable observations. "
            "Its tested benefit is causal rejection/reacquisition support when "
            "the first post-outage samples are impulsively corrupted."
        ),
    )


def save_external_single_cann_robustness_audit(report, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
