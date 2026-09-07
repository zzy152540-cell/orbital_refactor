from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from brain_inspired.orbital_direction_runner import (
    OrbitalDirectionHistory,
    run_orbital_direction_states,
)
from brain_inspired.orbital_direction_state import OrbitalDirectionConfig
from brain_inspired.orbital_phase_adapter import (
    OrbitalPlaneFrame,
    extract_orbital_phase_state,
)
from brain_inspired.ring_cann import RingCANNConfig
from experiments.walker_raw_sensor_comparison import _run, _walker_case

Array = np.ndarray


@dataclass(frozen=True)
class WalkerDirectionComparison:
    timestamps: Array
    truth_phase_by_node: dict[str, Array]
    free_running: dict[str, OrbitalDirectionHistory]
    periodic_anchor: dict[str, OrbitalDirectionHistory]
    interrupted_anchor: dict[str, OrbitalDirectionHistory]
    metrics: dict[str, dict[str, float]]
    runtime_seconds: dict[str, float]
    representative_node: str
    anchor_outage_window: tuple[float, float]
    phase_rate_bias_deg_per_hour: float


def run_walker_direction_navigation_comparison(
    *, duration=20.0, dt=2.0, seed=0, anchor_interval_samples=2,
    anchor_outage_window=(6.0, 14.0), direction_config=None,
    phase_rate_bias_deg_per_hour=0.0,
):
    """Compare causal per-node direction states on a Walker-20 filter run."""
    if anchor_interval_samples < 1:
        raise ValueError("anchor_interval_samples must be positive.")
    outage_start, outage_end = map(float, anchor_outage_window)
    if outage_end < outage_start:
        raise ValueError("Anchor outage end cannot precede its start.")
    case = _walker_case(seed=seed, duration=duration, dt=dt)
    filter_history = _run(case, case["observations"])
    times = case["timestamps"]
    nodes = tuple(filter_history.node_ids)
    frames = {
        node: OrbitalPlaneFrame.from_state_eci(case["initial_states"][node])
        for node in nodes
    }
    truth_phase = {
        node: np.asarray([
            extract_orbital_phase_state(
                timestamp=timestamp, state_eci=case["truth"][node][index],
                frame=frames[node], source_id=node,
            ).argument_of_latitude
            for index, timestamp in enumerate(times)
        ])
        for node in nodes
    }
    periodic_masks = {
        node: _periodic_anchor_mask(times.size, anchor_interval_samples)
        for node in nodes
    }
    interrupted_masks = {
        node: periodic_masks[node] & ~(
            (times >= outage_start) & (times <= outage_end)
        )
        for node in nodes
    }
    confidence = {
        node: _posterior_anchor_confidence(
            filter_history.active_covariance_history_by_node[node],
        )
        for node in nodes
    }
    config = direction_config or OrbitalDirectionConfig(
        ring=replace(RingCANNConfig(), internal_dt=0.002),
    )
    rate_bias = np.deg2rad(float(phase_rate_bias_deg_per_hour)) / 3600.0
    if not np.isfinite(rate_bias):
        raise ValueError("phase_rate_bias_deg_per_hour must be finite.")
    common = dict(
        timestamps=times,
        posterior_state_history_by_node=(
            filter_history.active_state_history_by_node
        ),
        frame_by_node=frames,
        anchor_confidence_by_node=confidence,
        phase_rate_bias_by_node={node: rate_bias for node in nodes},
        config=config,
    )
    histories = {}
    runtimes = {}
    for name, masks in (
        ("free_running", {node: np.zeros(times.size, dtype=bool)
                          for node in nodes}),
        ("periodic_anchor", periodic_masks),
        ("interrupted_anchor", interrupted_masks),
    ):
        started = perf_counter()
        histories[name] = run_orbital_direction_states(
            anchor_mask_by_node=masks, **common,
        )
        runtimes[name] = float(perf_counter() - started)
    metrics = {
        name: _direction_metrics(values, truth_phase)
        for name, values in histories.items()
    }
    return WalkerDirectionComparison(
        timestamps=times.copy(), truth_phase_by_node=truth_phase,
        free_running=histories["free_running"],
        periodic_anchor=histories["periodic_anchor"],
        interrupted_anchor=histories["interrupted_anchor"],
        metrics=metrics, runtime_seconds=runtimes,
        representative_node=nodes[0],
        anchor_outage_window=(outage_start, outage_end),
        phase_rate_bias_deg_per_hour=float(phase_rate_bias_deg_per_hour),
    )


def generate_walker_direction_comparison_figure(
    result: WalkerDirectionComparison,
    output_path="results/cann/walker_direction_navigation.png",
):
    """Plot phase, circular error, and anchor age for one Walker node."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    node = result.representative_node
    histories = (
        ("free running", result.free_running[node]),
        ("periodic anchor", result.periodic_anchor[node]),
        ("anchor outage/recovery", result.interrupted_anchor[node]),
    )
    truth = np.unwrap(result.truth_phase_by_node[node])
    figure, axes = plt.subplots(
        3, 1, figsize=(10.5, 9.0), sharex=True, constrained_layout=True,
    )
    axes[0].plot(result.timestamps, np.rad2deg(truth),
                 color="black", linewidth=2.0, label="truth phase")
    for label, history in histories:
        decoded = np.unwrap(history.anchored_phase)
        decoded += truth[0] - decoded[0]
        axes[0].plot(result.timestamps, np.rad2deg(decoded), label=label)
        error = _circular_error(history.anchored_phase,
                                result.truth_phase_by_node[node])
        axes[1].plot(result.timestamps, np.rad2deg(error), label=label)
    interrupted = result.interrupted_anchor[node]
    axes[2].plot(result.timestamps, interrupted.anchor_age,
                 color="tab:purple", label="anchor age")
    cue_times = result.timestamps[interrupted.cue_applied]
    axes[2].scatter(cue_times, np.zeros(cue_times.size), marker="|", s=100,
                    color="tab:green", label="soft anchor")
    start, end = result.anchor_outage_window
    for axis in axes:
        axis.axvspan(start, end, color="tab:red", alpha=0.10,
                     label="anchor outage" if axis is axes[2] else None)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("unwrapped phase (deg)")
    axes[1].set_ylabel("circular error (deg)")
    axes[2].set_ylabel("anchor age (s)")
    axes[2].set_xlabel("time (s)")
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    figure.suptitle(
        f"Walker direction-navigation CANN: {node} "
        f"(rate bias {result.phase_rate_bias_deg_per_hour:g} deg/h)"
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _periodic_anchor_mask(size, interval):
    mask = np.zeros(size, dtype=bool)
    mask[interval::interval] = True
    return mask


def _posterior_anchor_confidence(covariance_history):
    covariance = np.asarray(covariance_history, dtype=float)
    position_sigma = np.sqrt(
        np.maximum(0.0, np.trace(covariance[:, :3, :3], axis1=1, axis2=2) / 3.0)
    )
    return np.clip(10.0 / (10.0 + position_sigma), 0.0, 1.0)


def _direction_metrics(histories, truth_phase):
    errors = np.concatenate([
        _circular_error(history.anchored_phase, truth_phase[node])
        for node, history in histories.items()
    ])
    return {
        "phase_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(errors**2)))),
        "maximum_phase_error_deg": float(np.rad2deg(np.max(np.abs(errors)))),
        "valid_fraction": float(np.mean([
            value for history in histories.values() for value in history.valid
        ])),
        "cue_count": int(sum(
            np.count_nonzero(history.cue_applied)
            for history in histories.values()
        )),
        "maximum_anchor_age_s": float(max(
            np.max(history.anchor_age) for history in histories.values()
        )),
    }


def _circular_error(actual, expected):
    return (actual - expected + np.pi) % (2.0 * np.pi) - np.pi


if __name__ == "__main__":
    comparison = run_walker_direction_navigation_comparison()
    print(comparison.metrics)
    print(comparison.runtime_seconds)
    print(generate_walker_direction_comparison_figure(comparison))
