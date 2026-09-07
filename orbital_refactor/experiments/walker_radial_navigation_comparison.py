from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from brain_inspired.orbital_phase_adapter import OrbitalPlaneFrame, extract_orbital_phase_state
from brain_inspired.orbital_radial_runner import OrbitalRadialHistory, run_orbital_radial_states
from brain_inspired.orbital_radial_state import OrbitalRadialConfig
from experiments.walker_direction_navigation_comparison import (
    _periodic_anchor_mask,
    _posterior_anchor_confidence,
)
from experiments.walker_raw_sensor_comparison import _run, _walker_case

Array = np.ndarray


@dataclass(frozen=True)
class WalkerRadialComparison:
    timestamps: Array
    truth_displacement_by_node: dict[str, Array]
    free_running: dict[str, OrbitalRadialHistory]
    periodic_anchor: dict[str, OrbitalRadialHistory]
    interrupted_anchor: dict[str, OrbitalRadialHistory]
    metrics: dict[str, dict[str, float]]
    representative_node: str
    anchor_outage_window: tuple[float, float]
    radial_rate_bias_mps: float


def run_walker_radial_navigation_comparison(
    *, duration=120.0, dt=5.0, seed=0, anchor_interval_samples=2,
    anchor_outage_window=(30.0, 90.0), radial_rate_bias_mps=0.2,
    radial_config=None,
):
    """Compare causal per-node radial states on a Walker-20 filter run."""
    if anchor_interval_samples < 1:
        raise ValueError("anchor_interval_samples must be positive.")
    outage_start, outage_end = map(float, anchor_outage_window)
    if outage_end < outage_start:
        raise ValueError("Anchor outage end cannot precede its start.")
    if not np.isfinite(radial_rate_bias_mps):
        raise ValueError("radial_rate_bias_mps must be finite.")
    case = _walker_case(seed=seed, duration=duration, dt=dt)
    filtered = _run(case, case["observations"])
    times = np.asarray(case["timestamps"], dtype=float)
    nodes = tuple(filtered.node_ids)
    frames = {
        node: OrbitalPlaneFrame.from_state_eci(case["initial_states"][node])
        for node in nodes
    }
    reference_radius = {
        node: extract_orbital_phase_state(
            timestamp=times[0], state_eci=case["initial_states"][node],
            frame=frames[node], source_id=node,
        ).in_plane_radius
        for node in nodes
    }
    truth = {
        node: np.asarray([
            extract_orbital_phase_state(
                timestamp=time, state_eci=case["truth"][node][index],
                frame=frames[node], source_id=node,
            ).in_plane_radius - reference_radius[node]
            for index, time in enumerate(times)
        ])
        for node in nodes
    }
    periodic = {
        node: _periodic_anchor_mask(times.size, anchor_interval_samples)
        for node in nodes
    }
    interrupted = {
        node: periodic[node] & ~(
            (times >= outage_start) & (times <= outage_end)
        )
        for node in nodes
    }
    confidence = {
        node: _posterior_anchor_confidence(
            filtered.active_covariance_history_by_node[node],
        )
        for node in nodes
    }
    common = dict(
        timestamps=times,
        posterior_state_history_by_node=filtered.active_state_history_by_node,
        frame_by_node=frames, anchor_confidence_by_node=confidence,
        radial_rate_bias_by_node={node: radial_rate_bias_mps for node in nodes},
        config=radial_config or OrbitalRadialConfig(),
    )
    histories = {
        "free_running": run_orbital_radial_states(
            anchor_mask_by_node={node: np.zeros(times.size, dtype=bool)
                                 for node in nodes}, **common,
        ),
        "periodic_anchor": run_orbital_radial_states(
            anchor_mask_by_node=periodic, **common,
        ),
        "interrupted_anchor": run_orbital_radial_states(
            anchor_mask_by_node=interrupted, **common,
        ),
    }
    metrics = {
        name: _radial_metrics(history, truth)
        for name, history in histories.items()
    }
    return WalkerRadialComparison(
        timestamps=times, truth_displacement_by_node=truth,
        free_running=histories["free_running"],
        periodic_anchor=histories["periodic_anchor"],
        interrupted_anchor=histories["interrupted_anchor"], metrics=metrics,
        representative_node=nodes[0],
        anchor_outage_window=(outage_start, outage_end),
        radial_rate_bias_mps=float(radial_rate_bias_mps),
    )


def generate_walker_radial_comparison_figure(
    result: WalkerRadialComparison,
    output_path="results/cann/walker_radial_navigation.png",
):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    node = result.representative_node
    histories = (
        ("free running", result.free_running[node]),
        ("periodic anchor", result.periodic_anchor[node]),
        ("anchor outage/recovery", result.interrupted_anchor[node]),
    )
    figure, axes = plt.subplots(3, 1, figsize=(10.5, 9.0), sharex=True,
                                constrained_layout=True)
    truth = result.truth_displacement_by_node[node]
    axes[0].plot(result.timestamps, truth, color="black", linewidth=2,
                 label="truth radial displacement")
    for label, history in histories:
        axes[0].plot(result.timestamps, history.anchored_displacement, label=label)
        axes[1].plot(result.timestamps,
                     history.anchored_displacement - truth, label=label)
    interrupted = result.interrupted_anchor[node]
    axes[2].plot(result.timestamps, interrupted.anchor_age, color="tab:purple",
                 label="anchor age")
    cue_times = result.timestamps[interrupted.cue_applied]
    axes[2].scatter(cue_times, np.zeros(cue_times.size), marker="|", s=100,
                    color="tab:green", label="soft anchor")
    start, end = result.anchor_outage_window
    for axis in axes:
        axis.axvspan(start, end, color="tab:red", alpha=0.1,
                     label="anchor outage" if axis is axes[2] else None)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("radial displacement (m)")
    axes[1].set_ylabel("radial error (m)")
    axes[2].set_ylabel("anchor age (s)")
    axes[2].set_xlabel("time (s)")
    for axis in axes:
        axis.legend()
    figure.suptitle(
        f"Walker radial Line CANN: {node} "
        f"(rate bias {result.radial_rate_bias_mps:g} m/s)"
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _radial_metrics(histories, truth):
    errors = np.concatenate([
        history.anchored_displacement - truth[node]
        for node, history in histories.items()
    ])
    saturated = np.concatenate([
        history.saturated_at_boundary for history in histories.values()
    ])
    valid = np.concatenate([history.valid for history in histories.values()])
    posterior_residual = np.concatenate([
        history.anchored_residual for history in histories.values()
    ])
    return {
        "radial_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "maximum_radial_error_m": float(np.max(np.abs(errors))),
        "posterior_tracking_rmse_m": float(
            np.sqrt(np.mean(posterior_residual**2))
        ),
        "valid_fraction": float(np.mean(valid)),
        "boundary_saturation_fraction": float(np.mean(saturated)),
        "cue_count": int(sum(np.count_nonzero(history.cue_applied)
                             for history in histories.values())),
    }


if __name__ == "__main__":
    comparison = run_walker_radial_navigation_comparison()
    print(comparison.metrics)
    print(generate_walker_radial_comparison_figure(comparison))
