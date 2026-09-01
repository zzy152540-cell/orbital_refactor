from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from brain_inspired.line_cann import (
    LineCANN, LineCANNConfig, decode_line_activity,
    decode_line_activity_boundary_corrected, decode_line_activity_peak_fit,
)


def run_line_cann_decoder_benchmark(
    *, config=LineCANNConfig(), center_count=101,
):
    centers = np.linspace(config.minimum_value, config.maximum_value, center_count)
    rows = []
    cann = LineCANN(config)
    for center in centers:
        output = cann.reset(float(center))
        centroid, _, _ = decode_line_activity(
            output.neural_activity, cann.preferred_value,
            background_firing_rate=config.background_firing_rate,
        )
        peak_fit = decode_line_activity_peak_fit(
            output.neural_activity, cann.preferred_value,
            background_firing_rate=config.background_firing_rate,
        )
        hybrid = decode_line_activity_boundary_corrected(
            output.neural_activity, cann.preferred_value,
            background_firing_rate=config.background_firing_rate,
            centroid=centroid,
        )
        rows.append({
            "truth": float(center), "centroid": centroid,
            "peak_fit": peak_fit, "hybrid": hybrid,
            "centroid_error": centroid - center,
            "peak_fit_error": peak_fit - center,
            "hybrid_error": hybrid - center,
        })
    edge_count = max(2, center_count // 10)
    edge = rows[:edge_count] + rows[-edge_count:]
    summary = {}
    for method in ("centroid", "peak_fit", "hybrid"):
        errors = np.array([row[f"{method}_error"] for row in rows])
        edge_errors = np.array([row[f"{method}_error"] for row in edge])
        summary[method] = {
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "maximum_absolute_error": float(np.max(np.abs(errors))),
            "edge_rmse": float(np.sqrt(np.mean(edge_errors ** 2))),
        }
    return {"rows": rows, "summary": summary}


def write_line_cann_decoder_benchmark(result, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "decoder_scan.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=result["rows"][0])
        writer.writeheader()
        writer.writerows(result["rows"])
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")
    return {"csv": csv_path, "summary": summary_path}
