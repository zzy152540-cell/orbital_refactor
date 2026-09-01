from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from brain_inspired.coupled_ring_line_cann import (
    CoupledRingLineCANN, CoupledRingLineCANNConfig,
)
from brain_inspired.line_cann import LineCANNConfig
from experiments.cann_inter_satellite_azimuth import (
    _difference, _select_link, _weighted_circular_mean,
)
from scenarios.walker_scenario import WalkerDeltaConfig, generate_walker_delta_scenario


def evaluate_coupled_ring_line_robustness(
    *, seeds=range(5), duration=1800.0, dt=2.0,
    bias_anchor_mode="rolling_cue",
    initial_offsets_deg=(-2.0, -1.0, 1.0, 2.0),
    minimum_bias_baseline=120.0, line_cue_gain=0.1,
    anchor_agreement_scale_deg_s=0.002,
    included_cases=(
        "initial_phase_offset", "time_varying_rate_bias",
        "insufficient_pre_outage_baseline",
    ),
):
    """Exercise the coupled CANN outside its nominal fixed-bias assumptions."""
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
    rows = []

    nominal_outage = (600.0, 1200.0)
    nominal_available = _availability(times, geometry_visible, nominal_outage)
    for offset_deg in initial_offsets_deg if "initial_phase_offset" in included_cases else ():
        for seed in seeds:
            rate, hint = _sample_measurements(
                seed, times, truth, truth_rate, nominal_available,
                np.full(times.size, np.deg2rad(0.003)),
            )
            rows.append(_run_case(
                case="initial_phase_offset", seed=seed, times=times,
                truth=truth, rate=rate, hint=hint,
                available=nominal_available, outage_window=nominal_outage,
                initial_phase=truth[0] + np.deg2rad(offset_deg),
                parameter=offset_deg, bias_anchor_mode=bias_anchor_mode,
                minimum_bias_baseline=minimum_bias_baseline,
                line_cue_gain=line_cue_gain,
                anchor_agreement_scale_deg_s=anchor_agreement_scale_deg_s,
            ))

    varying_bias = np.deg2rad(
        0.003 + 0.002 * np.sin(2.0 * np.pi * times / 900.0)
    )
    for seed in seeds if "time_varying_rate_bias" in included_cases else ():
        rate, hint = _sample_measurements(
            seed, times, truth, truth_rate, nominal_available, varying_bias,
        )
        rows.append(_run_case(
            case="time_varying_rate_bias", seed=seed, times=times,
            truth=truth, rate=rate, hint=hint,
            available=nominal_available, outage_window=nominal_outage,
            initial_phase=truth[0], parameter=0.002,
            bias_anchor_mode=bias_anchor_mode,
            minimum_bias_baseline=minimum_bias_baseline,
            line_cue_gain=line_cue_gain,
            anchor_agreement_scale_deg_s=anchor_agreement_scale_deg_s,
        ))

    short_outage = (60.0, 660.0)
    short_available = _availability(times, geometry_visible, short_outage)
    for seed in seeds if "insufficient_pre_outage_baseline" in included_cases else ():
        rate, hint = _sample_measurements(
            seed, times, truth, truth_rate, short_available,
            np.full(times.size, np.deg2rad(0.003)),
        )
        rows.append(_run_case(
            case="insufficient_pre_outage_baseline", seed=seed, times=times,
            truth=truth, rate=rate, hint=hint, available=short_available,
            outage_window=short_outage, initial_phase=truth[0], parameter=60.0,
            bias_anchor_mode=bias_anchor_mode,
            minimum_bias_baseline=minimum_bias_baseline,
            line_cue_gain=line_cue_gain,
            anchor_agreement_scale_deg_s=anchor_agreement_scale_deg_s,
        ))

    summaries = {}
    for case in sorted({row["case"] for row in rows}):
        selected = [row for row in rows if row["case"] == case]
        outage_rmse = np.asarray([row["outage_rmse_deg"] for row in selected])
        summaries[case] = {
            "run_count": len(selected),
            "mean_outage_rmse_deg": float(outage_rmse.mean()),
            "maximum_outage_rmse_deg": float(outage_rmse.max()),
            "mean_final_rate_bias_deg_s": float(np.mean([
                row["final_rate_bias_deg_s"] for row in selected
            ])),
            "maximum_abs_error_deg": float(max(
                row["maximum_abs_error_deg"] for row in selected
            )),
        }
    return {
        "observer_id": observer, "target_id": target,
        "bias_anchor_mode": bias_anchor_mode,
        "initial_offsets_deg": [float(value) for value in initial_offsets_deg],
        "minimum_bias_baseline": float(minimum_bias_baseline),
        "line_cue_gain": float(line_cue_gain),
        "anchor_agreement_scale_deg_s": float(anchor_agreement_scale_deg_s),
        "evaluation_seeds": [int(seed) for seed in seeds],
        "rows": rows, "summary": summaries,
    }


def write_coupled_ring_line_robustness(result, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "per_case_seed.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=result["rows"][0])
        writer.writeheader()
        writer.writerows(result["rows"])
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps({
        key: result[key] for key in (
            "observer_id", "target_id", "bias_anchor_mode", "initial_offsets_deg",
            "minimum_bias_baseline", "line_cue_gain",
            "anchor_agreement_scale_deg_s",
            "evaluation_seeds", "summary",
        )
    }, indent=2), encoding="utf-8")
    return {"csv": csv_path, "summary": summary_path}


def _availability(times, geometry_visible, outage_window):
    return geometry_visible & (np.arange(times.size) % 5 == 0) & ~(
        (times >= outage_window[0]) & (times <= outage_window[1])
    )


def _sample_measurements(seed, times, truth, truth_rate, available, rate_bias):
    rng = np.random.default_rng(int(seed))
    rate = truth_rate + rate_bias + rng.normal(
        0.0, np.deg2rad(0.01), times.size,
    )
    ir = rng.normal(0.0, np.deg2rad(0.05), times.size)
    optical = rng.normal(0.0, np.deg2rad(0.02), times.size)
    hint = np.full(times.size, np.nan)
    hint[available] = _weighted_circular_mean(
        (truth[available] + ir[available]) % (2.0 * np.pi),
        (truth[available] + optical[available]) % (2.0 * np.pi),
        1.0 / np.deg2rad(0.05) ** 2,
        1.0 / np.deg2rad(0.02) ** 2,
    )
    return rate, hint


def _run_case(
    *, case, seed, times, truth, rate, hint, available, outage_window,
    initial_phase, parameter,
    bias_anchor_mode,
    minimum_bias_baseline,
    line_cue_gain,
    anchor_agreement_scale_deg_s,
):
    observer = CoupledRingLineCANN(CoupledRingLineCANNConfig(
        bias_anchor_mode=bias_anchor_mode,
        minimum_bias_baseline=minimum_bias_baseline,
        anchor_agreement_scale=np.deg2rad(anchor_agreement_scale_deg_s),
        line=LineCANNConfig(
            minimum_value=np.deg2rad(-0.05),
            maximum_value=np.deg2rad(0.05),
            tuning_width=np.deg2rad(0.003),
            cue_gain=line_cue_gain,
        ),
    ))
    output = observer.initialize(phase=initial_phase, timestamp=times[0])
    phase = [output.decoded_phase]
    bias = [output.decoded_rate_bias]
    counts = [output.bias_observation_count]
    for index in range(1, times.size):
        use = bool(available[index])
        output = observer.update(
            timestamp=times[index], measured_phase_rate=rate[index - 1],
            phase_hint=hint[index] if use else None,
            phase_hint_valid=use,
        )
        phase.append(output.decoded_phase)
        bias.append(output.decoded_rate_bias)
        counts.append(output.bias_observation_count)
    phase = np.asarray(phase)
    bias = np.asarray(bias)
    counts = np.asarray(counts)
    error = _difference(phase, truth)
    outage = (times >= outage_window[0]) & (times <= outage_window[1])
    pre_outage = np.flatnonzero(times < outage_window[0])[-1]
    return {
        "case": case, "seed": int(seed), "parameter": float(parameter),
        "rmse_deg": float(np.rad2deg(np.sqrt(np.mean(error ** 2)))),
        "outage_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(error[outage] ** 2)))),
        "maximum_abs_error_deg": float(np.rad2deg(np.max(np.abs(error)))),
        "pre_outage_rate_bias_deg_s": float(np.rad2deg(bias[pre_outage])),
        "final_rate_bias_deg_s": float(np.rad2deg(bias[-1])),
        "pre_outage_bias_observation_count": int(counts[pre_outage]),
        "final_bias_observation_count": int(counts[-1]),
    }
