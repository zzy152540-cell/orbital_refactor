from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from experiments.cann_inter_satellite_azimuth import (
    run_inter_satellite_azimuth_benchmark,
)


FROZEN_PLL = {"pll_kp": 1.0, "pll_ki": 0.01}
FROZEN_KALMAN = {
    "kalman_phase_process_std_deg": 0.01,
    "kalman_bias_process_std_deg_s": 0.0001,
}


def evaluate_frozen_angle_trackers(
    *, seeds=range(10), duration=1800.0, dt=2.0,
    outage_window=(600.0, 1200.0),
):
    rows = []
    for seed in seeds:
        result = run_inter_satellite_azimuth_benchmark(
            seed=int(seed), duration=duration, dt=dt,
            outage_window=outage_window, **FROZEN_PLL, **FROZEN_KALMAN,
        )
        for method, rmse in result.rmse_deg_by_mode.items():
            rows.append({
                "seed": int(seed), "method": method,
                "rmse_deg": rmse,
                "outage_rmse_deg": result.outage_rmse_deg_by_mode[method],
                "reacquisition_time_s": result.reacquisition_time_s_by_mode[method],
            })
    methods = sorted({row["method"] for row in rows})
    summary = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        overall = np.array([row["rmse_deg"] for row in selected])
        outage = np.array([row["outage_rmse_deg"] for row in selected])
        complement = {
            row["seed"]: row for row in rows
            if row["method"] == "gated_complementary"
        }
        wins = sum(
            row["outage_rmse_deg"]
            < complement[row["seed"]]["outage_rmse_deg"]
            for row in selected
        )
        summary[method] = {
            "mean_rmse_deg": float(overall.mean()),
            "rmse_std_deg": float(overall.std()),
            "mean_outage_rmse_deg": float(outage.mean()),
            "outage_rmse_std_deg": float(outage.std()),
            "outage_wins_vs_complementary": int(wins),
            "seed_count": len(selected),
        }
    return {
        "evaluation_seeds": [int(seed) for seed in seeds],
        "frozen_pll": FROZEN_PLL,
        "frozen_kalman": FROZEN_KALMAN,
        "rows": rows,
        "summary": summary,
    }


def write_frozen_angle_tracker_evaluation(result, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "per_seed.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "seed", "method", "rmse_deg", "outage_rmse_deg",
            "reacquisition_time_s",
        ))
        writer.writeheader()
        writer.writerows(result["rows"])
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps({
        key: result[key] for key in (
            "evaluation_seeds", "frozen_pll", "frozen_kalman", "summary",
        )
    }, indent=2), encoding="utf-8")
    return {"csv": csv_path, "summary": summary_path}
