from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from brain_inspired.orbital_rt_adapter import extract_orbital_rt_offset
from brain_inspired.orbital_rt_grid_runner import (
    OrbitalRTGridHistory,
    run_orbital_rt_grid_states,
)
from brain_inspired.orbital_rt_grid_state import OrbitalRTGridConfig
from experiments.walker_direction_navigation_comparison import (
    _periodic_anchor_mask,
    _posterior_anchor_confidence,
)
from experiments.walker_raw_sensor_comparison import _run, _walker_case
from orbital_core.dynamics import rk4_step_absolute

Array = np.ndarray


@dataclass(frozen=True)
class WalkerRTGridComparison:
    timestamps: Array
    truth_rt_by_node: dict[str, Array]
    reference_state_history_by_node: dict[str, Array]
    free_running: dict[str, OrbitalRTGridHistory]
    periodic_anchor: dict[str, OrbitalRTGridHistory]
    interrupted_anchor: dict[str, OrbitalRTGridHistory]
    adaptive_anchor: dict[str, OrbitalRTGridHistory]
    adaptive_interrupted_anchor: dict[str, OrbitalRTGridHistory]
    metrics: dict[str, dict[str, float]]
    representative_node: str
    anchor_outage_window: tuple[float, float]
    rate_bias_rt_mps: Array


def run_walker_rt_grid_comparison(
    *, duration=120.0, dt=5.0, seed=0, anchor_interval_samples=2,
    anchor_outage_window=(30.0, 90.0), rate_bias_rt_mps=(0.1, 0.2),
    grid_config=None,
):
    case = _walker_case(seed=seed, duration=duration, dt=dt)
    filtered = _run(case, case["observations"])
    times = np.asarray(case["timestamps"], dtype=float)
    nodes = tuple(filtered.node_ids)
    references = {
        node: _propagate_reference(case["initial_states"][node], times)
        for node in nodes
    }
    truth_rt = {
        node: np.asarray([
            _rt_position(
                time, case["truth"][node][index], references[node][index], node,
            )
            for index, time in enumerate(times)
        ])
        for node in nodes
    }
    periodic_mask = _periodic_anchor_mask(times.size, anchor_interval_samples)
    periodic = {node: periodic_mask for node in nodes}
    start, end = map(float, anchor_outage_window)
    if end < start:
        raise ValueError("Anchor outage end cannot precede its start.")
    interrupted_mask = periodic_mask & ~((times >= start) & (times <= end))
    interrupted = {node: interrupted_mask for node in nodes}
    confidence = {
        node: _posterior_anchor_confidence(
            filtered.active_covariance_history_by_node[node],
        )
        for node in nodes
    }
    bias = np.asarray(rate_bias_rt_mps, dtype=float).reshape(-1)
    if bias.shape != (2,) or np.any(~np.isfinite(bias)):
        raise ValueError("rate_bias_rt_mps must be a finite 2-vector.")
    base_config = grid_config or OrbitalRTGridConfig()
    common = dict(
        timestamps=times,
        posterior_state_history_by_node=filtered.active_state_history_by_node,
        reference_state_history_by_node=references,
        anchor_confidence_by_node=confidence,
        rate_bias_rt_by_node={node: bias for node in nodes},
        config=base_config,
    )
    histories = {
        "free_running": run_orbital_rt_grid_states(
            anchor_mask_by_node={node: np.zeros(times.size, dtype=bool)
                                 for node in nodes}, **common,
        ),
        "periodic_anchor": run_orbital_rt_grid_states(
            anchor_mask_by_node=periodic, **common,
        ),
        "interrupted_anchor": run_orbital_rt_grid_states(
            anchor_mask_by_node=interrupted, **common,
        ),
        "adaptive_anchor": run_orbital_rt_grid_states(
            anchor_mask_by_node=periodic,
            **{**common, "config": replace(
                base_config, rolling_bias_enabled=True,
            )},
        ),
        "adaptive_interrupted_anchor": run_orbital_rt_grid_states(
            anchor_mask_by_node=interrupted,
            **{**common, "config": replace(
                base_config, rolling_bias_enabled=True,
            )},
        ),
    }
    return WalkerRTGridComparison(
        timestamps=times, truth_rt_by_node=truth_rt,
        reference_state_history_by_node=references,
        free_running=histories["free_running"],
        periodic_anchor=histories["periodic_anchor"],
        interrupted_anchor=histories["interrupted_anchor"],
        adaptive_anchor=histories["adaptive_anchor"],
        adaptive_interrupted_anchor=histories["adaptive_interrupted_anchor"],
        metrics={name: _metrics(values, truth_rt)
                 for name, values in histories.items()},
        representative_node=nodes[0], anchor_outage_window=(start, end),
        rate_bias_rt_mps=bias.copy(),
    )


def generate_walker_rt_grid_figure(
    result: WalkerRTGridComparison,
    output_path="results/cann/walker_rt_grid_navigation.png",
):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    node = result.representative_node
    histories = (
        ("free running", result.free_running[node]),
        ("periodic anchor", result.periodic_anchor[node]),
        ("anchor outage/recovery", result.interrupted_anchor[node]),
        ("adaptive rolling anchor", result.adaptive_anchor[node]),
        ("adaptive anchor outage", result.adaptive_interrupted_anchor[node]),
    )
    figure, axes = plt.subplots(3, 1, figsize=(10.5, 9.0), sharex=True,
                                constrained_layout=True)
    truth = result.truth_rt_by_node[node]
    for axis_index, coordinate in enumerate((0, 1)):
        axes[axis_index].plot(result.timestamps, truth[:, coordinate],
                              color="black", linewidth=2, label="truth offset")
        for label, history in histories:
            axes[axis_index].plot(
                result.timestamps, history.anchored_rt[:, coordinate], label=label,
            )
        axes[axis_index].set_ylabel(("radial R" if coordinate == 0 else
                                     "along-track T") + " (m)")
    interrupted = result.interrupted_anchor[node]
    axes[2].plot(result.timestamps, interrupted.anchor_age, color="tab:purple",
                 label="anchor age")
    axes[2].scatter(result.timestamps[interrupted.cue_applied],
                    np.zeros(np.count_nonzero(interrupted.cue_applied)),
                    marker="|", s=100, color="tab:green", label="soft anchor")
    start, end = result.anchor_outage_window
    for axis in axes:
        axis.axvspan(start, end, color="tab:red", alpha=0.1,
                     label="anchor outage" if axis is axes[2] else None)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[2].set_ylabel("anchor age (s)")
    axes[2].set_xlabel("time (s)")
    figure.suptitle(f"Walker RT grid sidecar: {node}")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _propagate_reference(initial, times):
    history = [np.asarray(initial, dtype=float).copy()]
    for index in range(1, times.size):
        history.append(rk4_step_absolute(
            history[-1], float(times[index] - times[index - 1]),
        ))
    return np.asarray(history)


def _rt_position(timestamp, state, reference, node):
    offset = extract_orbital_rt_offset(
        timestamp=timestamp, state_eci=state, reference_state_eci=reference,
        source_id=node,
    )
    return np.array([offset.radial_position, offset.along_track_position])


def _metrics(histories, truth):
    errors = np.concatenate([
        history.anchored_rt - truth[node]
        for node, history in histories.items()
    ])
    posterior_errors = np.concatenate([
        history.anchored_residual_rt for history in histories.values()
    ])
    saturated = np.concatenate([
        history.saturated_at_boundary for history in histories.values()
    ])
    return {
        "radial_rmse_m": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
        "along_track_rmse_m": float(np.sqrt(np.mean(errors[:, 1] ** 2))),
        "rt_norm_rmse_m": float(np.sqrt(np.mean(np.sum(errors**2, axis=1)))),
        "maximum_rt_error_m": float(np.max(np.linalg.norm(errors, axis=1))),
        "posterior_tracking_radial_rmse_m": float(np.sqrt(np.mean(
            posterior_errors[:, 0] ** 2,
        ))),
        "posterior_tracking_along_track_rmse_m": float(np.sqrt(np.mean(
            posterior_errors[:, 1] ** 2,
        ))),
        "posterior_tracking_rt_norm_rmse_m": float(np.sqrt(np.mean(
            np.sum(posterior_errors**2, axis=1),
        ))),
        "valid_fraction": float(np.mean(np.concatenate([
            history.valid for history in histories.values()
        ]))),
        "boundary_saturation_fraction": float(np.mean(saturated)),
        "bias_update_count": float(np.sum([
            np.count_nonzero(history.bias_update_applied)
            for history in histories.values()
        ])),
        "anchor_rejection_count": float(np.sum([
            np.count_nonzero(history.anchor_rejected)
            for history in histories.values()
        ])),
        "final_rate_correction_norm_mps": float(np.mean([
            np.linalg.norm(history.rate_correction_rt[-1])
            for history in histories.values()
        ])),
        "reference_rebase_event_count": float(np.sum([
            np.count_nonzero(history.reference_rebased)
            for history in histories.values()
        ])),
    }


if __name__ == "__main__":
    comparison = run_walker_rt_grid_comparison()
    print(comparison.metrics)
    print(generate_walker_rt_grid_figure(comparison))
