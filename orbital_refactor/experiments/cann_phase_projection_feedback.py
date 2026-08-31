from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from brain_inspired.orbital_phase_adapter import OrbitalPlaneFrame
from brain_inspired.orbital_phase_sidecar import run_orbital_phase_sidecar
from experiments.single_satellite_cann_comparison import (
    run_single_satellite_cann_comparison,
)


def run_cann_phase_projection_feedback(
    *, duration: float = 1800.0, dt: float = 2.0, seed: int = 0,
    outage_start: float = 600.0, outage_end: float = 1200.0,
    projection_gain: float = 1.0,
):
    if not 0.0 <= projection_gain <= 1.0:
        raise ValueError("projection_gain must be in [0, 1].")
    baseline = run_single_satellite_cann_comparison(
        duration=duration, dt=dt, seed=seed,
        outage_start=outage_start, outage_end=outage_end,
        outage_modalities=("opt", "ir", "rad"), enable_cann=False,
    )
    times = baseline["timestamps"]
    estimate = baseline["estimated_state_history_eci"]
    truth = baseline["truth_state_history_eci"]
    outage = (times >= outage_start) & (times <= outage_end)
    frame = OrbitalPlaneFrame.from_state_eci(estimate[0])
    trace = run_orbital_phase_sidecar(
        timestamps=times, state_history_eci=estimate, frame=frame,
        cue_interval_samples=5, cue_valid_mask=~outage,
        source_id="single-satellite:trusted-cue-cann",
    )
    projected = estimate.copy()
    for index in np.flatnonzero(outage):
        projected[index] = _project_state_phase(
            estimate[index], trace.decoded_phase[index], frame, projection_gain,
        )
    baseline_error = np.linalg.norm(estimate[:, :3] - truth[:, :3], axis=1)
    projected_error = np.linalg.norm(projected[:, :3] - truth[:, :3], axis=1)
    summary = {
        "projection_gain": float(projection_gain),
        "baseline_position_rmse_m": _rmse(baseline_error),
        "projected_position_rmse_m": _rmse(projected_error),
        "baseline_outage_rmse_m": _rmse(baseline_error[outage]),
        "projected_outage_rmse_m": _rmse(projected_error[outage]),
        "outage_improvement_m": (
            _rmse(baseline_error[outage]) - _rmse(projected_error[outage])
        ),
        "maximum_projection_change_m": float(np.max(np.linalg.norm(
            projected[:, :3] - estimate[:, :3], axis=1,
        ))),
        "cue_count": int(np.count_nonzero(trace.cue_applied)),
    }
    return {
        "timestamps": times, "outage": outage, "baseline_error_m": baseline_error,
        "projected_error_m": projected_error, "cann": trace, "summary": summary,
        "estimated_state_history_eci": estimate,
        "truth_state_history_eci": truth, "frame": frame,
    }


def run_cann_phase_projection_gain_sweep(
    *, gains=(0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0),
    **kwargs,
):
    base = run_cann_phase_projection_feedback(projection_gain=0.0, **kwargs)
    estimate = base["estimated_state_history_eci"]
    truth = base["truth_state_history_eci"]
    outage = base["outage"]
    rows = []
    for gain in gains:
        projected = estimate.copy()
        for index in np.flatnonzero(outage):
            projected[index] = _project_state_phase(
                estimate[index], base["cann"].decoded_phase[index],
                base["frame"], float(gain),
            )
        error = np.linalg.norm(projected[:, :3] - truth[:, :3], axis=1)
        rows.append({
            "projection_gain": float(gain),
            "position_rmse_m": _rmse(error),
            "outage_rmse_m": _rmse(error[outage]),
            "outage_improvement_m": (
                base["summary"]["baseline_outage_rmse_m"] - _rmse(error[outage])
            ),
            "maximum_projection_change_m": float(np.max(np.linalg.norm(
                projected[:, :3] - estimate[:, :3], axis=1,
            ))),
        })
    return {"base": base, "rows": rows}


def write_gain_sweep_results(result, output_dir: str | Path) -> dict[str, Path]:
    import csv
    import matplotlib.pyplot as plt

    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "gain_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=result["rows"][0].keys())
        writer.writeheader(); writer.writerows(result["rows"])
    gains = [row["projection_gain"] for row in result["rows"]]
    rmse = [row["outage_rmse_m"] for row in result["rows"]]
    fig, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(gains, rmse, marker="o", label="CANN phase projection")
    axis.axhline(
        result["base"]["summary"]["baseline_outage_rmse_m"],
        color="tab:blue", linestyle="--", label="Original Federated-CI",
    )
    axis.set_xlabel("Projection gain"); axis.set_ylabel("Outage position RMSE (m)")
    axis.set_title("CANN phase-projection gain sweep")
    axis.grid(True); axis.legend(); fig.tight_layout()
    figure_path = output / "gain_sweep.png"
    fig.savefig(figure_path, dpi=180); plt.close(fig)
    return {"csv": csv_path, "figure": figure_path}


def _project_state_phase(state, decoded_phase, frame, gain):
    state = np.asarray(state, dtype=float).copy()
    axes = (frame.ascending_node_axis, frame.in_plane_quadrature_axis, frame.normal_axis)
    x, y, z = (float(state[:3] @ axis) for axis in axes)
    vx, vy, vz = (float(state[3:] @ axis) for axis in axes)
    source_phase = np.arctan2(y, x)
    delta = (decoded_phase - source_phase + np.pi) % (2 * np.pi) - np.pi
    phase = source_phase + gain * delta
    radius = np.hypot(x, y)
    radial_speed = vx * np.cos(source_phase) + vy * np.sin(source_phase)
    tangential_speed = -vx * np.sin(source_phase) + vy * np.cos(source_phase)
    position = (
        radius * np.cos(phase) * axes[0]
        + radius * np.sin(phase) * axes[1] + z * axes[2]
    )
    velocity = (
        (radial_speed * np.cos(phase) - tangential_speed * np.sin(phase)) * axes[0]
        + (radial_speed * np.sin(phase) + tangential_speed * np.cos(phase)) * axes[1]
        + vz * axes[2]
    )
    return np.hstack((position, velocity))


def _rmse(values):
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def write_feedback_results(result, output_dir: str | Path) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")
    fig, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(result["timestamps"], result["baseline_error_m"], label="Original Federated-CI")
    axis.plot(result["timestamps"], result["projected_error_m"],
              label="CANN phase projection", alpha=.85)
    axis.fill_between(result["timestamps"], 0, 1, where=result["outage"],
                      color="gray", alpha=.15, transform=axis.get_xaxis_transform())
    axis.set_xlabel("Time (s)"); axis.set_ylabel("Position error (m)")
    axis.set_title("Controlled CANN phase-projection feedback during all-modal outage")
    axis.grid(True); axis.legend(); fig.tight_layout()
    figure_path = output / "overview.png"
    fig.savefig(figure_path, dpi=180); plt.close(fig)
    return {"summary": summary_path, "figure": figure_path}
