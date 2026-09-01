from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from experiments.cann_inter_satellite_azimuth import (
    run_inter_satellite_azimuth_benchmark,
)


def calibrate_adaptive_cann(
    *, seeds=range(10, 15), gains=(0.03, 0.1, 0.3),
    duration=600.0, dt=2.0, outage_window=(200.0, 400.0),
):
    rows = []
    for gain in gains:
        for seed in seeds:
            result = run_inter_satellite_azimuth_benchmark(
                seed=int(seed), duration=duration, dt=dt,
                outage_window=outage_window,
                adaptive_cann_rate_bias_gain=float(gain),
            )
            rows.append({
                "gain": float(gain), "seed": int(seed),
                "rmse_deg": result.rmse_deg_by_mode["bias_adaptive_cann"],
                "outage_rmse_deg": result.outage_rmse_deg_by_mode[
                    "bias_adaptive_cann"
                ],
                "original_cann_outage_rmse_deg": result.outage_rmse_deg_by_mode[
                    "gated_cann"
                ],
            })
    summaries = []
    for gain in gains:
        selected = [row for row in rows if row["gain"] == float(gain)]
        outage = np.array([row["outage_rmse_deg"] for row in selected])
        original = np.array([
            row["original_cann_outage_rmse_deg"] for row in selected
        ])
        summaries.append({
            "gain": float(gain),
            "mean_outage_rmse_deg": float(outage.mean()),
            "outage_rmse_std_deg": float(outage.std()),
            "wins_vs_original_cann": int(np.sum(outage < original)),
        })
    best = min(summaries, key=lambda item: item["mean_outage_rmse_deg"])
    return {
        "calibration_seeds": [int(seed) for seed in seeds],
        "duration_s": float(duration), "rows": rows,
        "summaries": summaries, "best": best,
    }


def write_adaptive_cann_calibration(result, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "per_seed.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=result["rows"][0])
        writer.writeheader()
        writer.writerows(result["rows"])
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps({
        key: result[key] for key in (
            "calibration_seeds", "duration_s", "summaries", "best",
        )
    }, indent=2), encoding="utf-8")
    return {"csv": csv_path, "summary": summary_path}
