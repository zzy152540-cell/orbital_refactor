from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from experiments.cann_inter_satellite_azimuth import (
    _circular_kalman, _difference, _gated_pll, _select_link,
    _weighted_circular_mean,
)
from scenarios.walker_scenario import WalkerDeltaConfig, generate_walker_delta_scenario


def calibrate_angle_trackers(
    *, seeds=range(10, 20), duration=1800.0, dt=2.0,
    outage_window=(600.0, 1200.0),
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
    scheduled = np.arange(times.size) % 5 == 0
    base_available = geometry_visible & scheduled & ~(
        (times >= outage_window[0]) & (times <= outage_window[1])
    )
    outage = (times >= outage_window[0]) & (times <= outage_window[1])
    samples = [
        _sample_inputs(seed, times, truth, truth_rate, base_available)
        for seed in seeds
    ]
    pll_rows = []
    for kp in (0.2, 0.4, 0.6, 0.8, 1.0):
        for ki in (0.0, 0.01, 0.03, 0.05, 0.1):
            pll_rows.append(_score(
                "pll", {"kp": kp, "ki": ki}, samples, truth, outage,
                lambda rate, hint, available: _gated_pll(
                    truth[0], rate, dt, hint, available, np.deg2rad(3.0), kp, ki,
                ),
            ))
    kalman_rows = []
    for phase_std in (0.0005, 0.001, 0.002, 0.005, 0.01):
        for bias_std in (0.0001, 0.0002, 0.0005, 0.001, 0.002):
            kalman_rows.append(_score(
                "circular_kalman",
                {"phase_process_std_deg": phase_std,
                 "bias_process_std_deg_s": bias_std},
                samples, truth, outage,
                lambda rate, hint, available, ps=phase_std, bs=bias_std:
                _circular_kalman(
                    truth[0], rate, dt, hint, available, np.deg2rad(3.0),
                    phase_process_std_deg=ps, bias_process_std_deg_s=bs,
                ),
            ))
    best_pll = min(pll_rows, key=lambda row: row["mean_outage_rmse_deg"])
    best_kalman = min(kalman_rows, key=lambda row: row["mean_outage_rmse_deg"])
    return {
        "observer_id": observer, "target_id": target,
        "calibration_seeds": list(seeds), "pll_rows": pll_rows,
        "kalman_rows": kalman_rows, "best_pll": best_pll,
        "best_circular_kalman": best_kalman,
    }


def _sample_inputs(seed, times, truth, truth_rate, available):
    rng = np.random.default_rng(seed)
    rate = truth_rate + np.deg2rad(0.003) + rng.normal(
        0.0, np.deg2rad(0.01), times.size,
    )
    ir = rng.normal(0.0, np.deg2rad(0.05), times.size)
    optical = rng.normal(0.0, np.deg2rad(0.02), times.size)
    hint = np.full(times.size, np.nan)
    hint[available] = _weighted_circular_mean(
        (truth[available] + ir[available]) % (2*np.pi),
        (truth[available] + optical[available]) % (2*np.pi),
        1/np.deg2rad(0.05)**2, 1/np.deg2rad(0.02)**2,
    )
    indices = np.flatnonzero(available)
    if indices.size >= 4:
        hint[indices[len(indices)//3]] += np.deg2rad(5.0)
        hint[indices[-2]] -= np.deg2rad(5.0)
        hint %= 2*np.pi
    return rate, hint, available


def _score(method, parameters, samples, truth, outage, tracker):
    overall, outage_values = [], []
    for rate, hint, available in samples:
        error = _difference(tracker(rate, hint, available), truth)
        overall.append(float(np.rad2deg(np.sqrt(np.mean(error**2)))))
        outage_values.append(float(np.rad2deg(np.sqrt(np.mean(error[outage]**2)))))
    return {
        "method": method, **parameters,
        "mean_rmse_deg": float(np.mean(overall)),
        "mean_outage_rmse_deg": float(np.mean(outage_values)),
        "outage_rmse_std_deg": float(np.std(outage_values)),
    }


def write_angle_tracker_calibration(result, output_dir):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    rows = result["pll_rows"] + result["kalman_rows"]
    csv_path = output / "grid.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    summary = {
        key: result[key] for key in (
            "observer_id", "target_id", "calibration_seeds",
            "best_pll", "best_circular_kalman",
        )
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"csv": csv_path, "summary": summary_path}
