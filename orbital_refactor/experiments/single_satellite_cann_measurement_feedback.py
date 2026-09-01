from __future__ import annotations

import json
from pathlib import Path

from experiments.single_satellite_cann_comparison import (
    run_single_satellite_cann_comparison,
)


def run_single_satellite_cann_measurement_feedback(**kwargs):
    common = dict(kwargs)
    common["enable_cann"] = False
    baseline = run_single_satellite_cann_comparison(
        **common, adaptive_cann_preprocess_ir=False,
    )
    processed = run_single_satellite_cann_comparison(
        **common, adaptive_cann_preprocess_ir=True,
    )
    hybrid = run_single_satellite_cann_comparison(
        **common, hybrid_cann_preprocess_ir=True,
    )
    baseline_summary = baseline["summary"]
    processed_summary = processed["summary"]
    hybrid_summary = hybrid["summary"]
    summary = {
        "baseline": baseline_summary,
        "adaptive_cann_ir": processed_summary,
        "hybrid_ring_line_cann_ir": hybrid_summary,
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
        "hybrid_position_rmse_change_m": (
            hybrid_summary["position_rmse_m"]
            - baseline_summary["position_rmse_m"]
        ),
        "hybrid_outage_position_rmse_change_m": (
            hybrid_summary["position_rmse_outage_m"]
            - baseline_summary["position_rmse_outage_m"]
        ),
        "hybrid_recovery_position_rmse_change_m": (
            hybrid_summary["position_rmse_recovery_m"]
            - baseline_summary["position_rmse_recovery_m"]
        ),
    }
    return {
        "baseline": baseline, "processed": processed, "hybrid": hybrid,
        "summary": summary,
    }


def write_single_satellite_cann_measurement_feedback(result, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "summary.json"
    path.write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")
    return path
