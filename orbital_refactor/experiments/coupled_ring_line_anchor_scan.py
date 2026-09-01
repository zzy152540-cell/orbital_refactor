from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from experiments.coupled_ring_line_robustness import (
    evaluate_coupled_ring_line_robustness,
)


def scan_rolling_anchor_parameters(
    *, seeds=(0, 4), baselines=(60.0, 120.0), gains=(0.1, 0.2),
    duration=1800.0, dt=2.0,
):
    rows = []
    for baseline in baselines:
        for gain in gains:
            result = evaluate_coupled_ring_line_robustness(
                seeds=seeds, duration=duration, dt=dt,
                bias_anchor_mode="rolling_cue",
                initial_offsets_deg=(2.0,),
                minimum_bias_baseline=baseline,
                line_cue_gain=gain,
                included_cases=(
                    "initial_phase_offset", "time_varying_rate_bias",
                ),
            )
            for case, summary in result["summary"].items():
                rows.append({
                    "minimum_bias_baseline_s": float(baseline),
                    "line_cue_gain": float(gain), "case": case,
                    **summary,
                })
    grouped = {}
    for baseline in baselines:
        for gain in gains:
            selected = [
                row for row in rows
                if row["minimum_bias_baseline_s"] == baseline
                and row["line_cue_gain"] == gain
            ]
            grouped[(baseline, gain)] = float(np.mean([
                row["mean_outage_rmse_deg"] for row in selected
            ]))
    best = min(grouped, key=grouped.get)
    return {
        "seeds": [int(seed) for seed in seeds], "rows": rows,
        "best": {
            "minimum_bias_baseline_s": float(best[0]),
            "line_cue_gain": float(best[1]),
            "mean_across_case_outage_rmse_deg": grouped[best],
        },
    }


def write_rolling_anchor_parameter_scan(result, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "grid.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=result["rows"][0])
        writer.writeheader()
        writer.writerows(result["rows"])
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps({
        "seeds": result["seeds"], "best": result["best"],
    }, indent=2), encoding="utf-8")
    return {"csv": csv_path, "summary": summary_path}
