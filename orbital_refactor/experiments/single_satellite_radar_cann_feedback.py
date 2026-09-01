from __future__ import annotations

import json
from pathlib import Path

from experiments.single_satellite_cann_comparison import (
    run_single_satellite_cann_comparison,
)


def run_single_satellite_radar_cann_feedback(
    *, radar_fault_mode="impulsive", **kwargs,
):
    common = dict(kwargs)
    common["enable_cann"] = False
    baseline = run_single_satellite_cann_comparison(
        **common, radar_cann_preprocess=False,
        radar_fault_mode=radar_fault_mode,
    )
    processed = run_single_satellite_cann_comparison(
        **common, radar_cann_preprocess=True,
        radar_fault_mode=radar_fault_mode,
    )
    baseline_summary = baseline["summary"]
    processed_summary = processed["summary"]
    summary = {
        "baseline": baseline_summary,
        "radar_range_rate_line_cann": processed_summary,
        "position_rmse_change_m": (
            processed_summary["position_rmse_m"]
            - baseline_summary["position_rmse_m"]
        ),
        "outage_position_rmse_change_m": (
            processed_summary["position_rmse_outage_m"]
            - baseline_summary["position_rmse_outage_m"]
        ),
        "recovery_position_rmse_change_m": (
            processed_summary["position_rmse_recovery_m"]
            - baseline_summary["position_rmse_recovery_m"]
        ),
    }
    return {"baseline": baseline, "processed": processed, "summary": summary}


def write_single_satellite_radar_cann_feedback(result, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "summary.json"
    path.write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")
    return path
