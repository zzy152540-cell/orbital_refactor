from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from brain_inspired.coupled_ring_line_cann import CoupledRingLineCANNConfig
from brain_inspired.line_cann import LineCANNConfig
from experiments.angle_tracker_calibration import _sample_inputs
from experiments.cann_inter_satellite_azimuth import (
    _coupled_ring_line_tracker, _difference, _select_link,
)
from scenarios.walker_scenario import WalkerDeltaConfig, generate_walker_delta_scenario


def evaluate_coupled_ring_line(
    *, seeds=range(10), duration=1800.0, dt=2.0,
    outage_window=(600.0, 1200.0),
    bias_anchor_mode="fixed_initial", minimum_bias_baseline=120.0,
    line_cue_gain=0.1, anchor_agreement_scale_deg_s=0.002,
):
    times = np.arange(0.0, duration + 0.5 * dt, dt)
    scenario = generate_walker_delta_scenario(
        timestamps=times,
        config=WalkerDeltaConfig(
            total_satellites=20, plane_count=10, phasing=1,
            semi_major_axis=6_978_137.0, eccentricity=0.001,
            inclination=np.deg2rad(53.0),
        ),
    )
    observer, target, truth, geometry_visible = _select_link(scenario, times)
    truth_rate = np.gradient(np.unwrap(truth), times)
    available = geometry_visible & (np.arange(times.size) % 5 == 0) & ~(
        (times >= outage_window[0]) & (times <= outage_window[1])
    )
    outage = (times >= outage_window[0]) & (times <= outage_window[1])
    rows = []
    tracker_config = CoupledRingLineCANNConfig(
        bias_anchor_mode=bias_anchor_mode,
        minimum_bias_baseline=minimum_bias_baseline,
        anchor_agreement_scale=np.deg2rad(anchor_agreement_scale_deg_s),
        line=LineCANNConfig(
            minimum_value=np.deg2rad(-0.05),
            maximum_value=np.deg2rad(0.05),
            tuning_width=np.deg2rad(0.003), cue_gain=line_cue_gain,
        ),
    )
    for seed in seeds:
        rate, hint, sampled_available = _sample_inputs(
            int(seed), times, truth, truth_rate, available,
        )
        phase, rate_bias = _coupled_ring_line_tracker(
            times, truth[0], rate, hint, sampled_available,
            config=tracker_config,
        )
        error = _difference(phase, truth)
        rows.append({
            "seed": int(seed),
            "rmse_deg": float(np.rad2deg(np.sqrt(np.mean(error ** 2)))),
            "outage_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(error[outage] ** 2)))),
            "final_rate_bias_deg_s": float(np.rad2deg(rate_bias[-1])),
            "pre_outage_rate_bias_deg_s": float(np.rad2deg(
                rate_bias[np.flatnonzero(times < outage_window[0])[-1]]
            )),
        })
    outage_values = np.array([row["outage_rmse_deg"] for row in rows])
    return {
        "observer_id": observer, "target_id": target,
        "bias_anchor_mode": bias_anchor_mode,
        "minimum_bias_baseline": float(minimum_bias_baseline),
        "line_cue_gain": float(line_cue_gain),
        "anchor_agreement_scale_deg_s": float(anchor_agreement_scale_deg_s),
        "evaluation_seeds": [int(seed) for seed in seeds], "rows": rows,
        "summary": {
            "mean_rmse_deg": float(np.mean([row["rmse_deg"] for row in rows])),
            "mean_outage_rmse_deg": float(outage_values.mean()),
            "outage_rmse_std_deg": float(outage_values.std()),
            "maximum_outage_rmse_deg": float(outage_values.max()),
        },
    }


def write_coupled_ring_line_evaluation(result, output_dir):
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
            "observer_id", "target_id", "bias_anchor_mode",
            "minimum_bias_baseline", "line_cue_gain",
            "anchor_agreement_scale_deg_s",
            "evaluation_seeds", "summary",
        )
    }, indent=2), encoding="utf-8")
    return {"csv": csv_path, "summary": summary_path}
