from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from brain_inspired.orbital_direction_runner import run_orbital_direction_states
from brain_inspired.orbital_direction_state import OrbitalDirectionConfig
from brain_inspired.orbital_phase_adapter import OrbitalPlaneFrame, extract_orbital_phase_state
from brain_inspired.ring_cann import RingCANNConfig
from experiments.walker_direction_navigation_comparison import (
    _circular_error,
    _periodic_anchor_mask,
    _posterior_anchor_confidence,
)
from experiments.walker_raw_sensor_comparison import _run, _walker_case


@dataclass(frozen=True)
class AnchorScanRecord:
    phase_rate_bias_deg_per_hour: float
    maximum_anchor_gain: float
    anchor_interval_seconds: float
    phase_rmse_deg: float
    maximum_phase_error_deg: float
    cue_count: int


def run_walker_direction_anchor_scan(
    *, duration=120.0, dt=5.0, seed=0,
    phase_rate_biases_deg_per_hour=(2.0, 10.0),
    maximum_anchor_gains=(0.05, 0.15, 0.25, 0.4),
    anchor_intervals_seconds=(10.0, 20.0, 40.0),
    node_limit=5, ring_config=None,
):
    """Coarsely screen direction-anchor parameters on shared filter histories."""
    case = _walker_case(seed=seed, duration=duration, dt=dt)
    filtered = _run(case, case["observations"])
    nodes = tuple(filtered.node_ids[:node_limit])
    times = np.asarray(case["timestamps"], dtype=float)
    posterior = {
        node: filtered.active_state_history_by_node[node] for node in nodes
    }
    frames = {
        node: OrbitalPlaneFrame.from_state_eci(case["initial_states"][node])
        for node in nodes
    }
    truth = {
        node: np.asarray([
            extract_orbital_phase_state(
                timestamp=time, state_eci=case["truth"][node][index],
                frame=frames[node], source_id=node,
            ).argument_of_latitude
            for index, time in enumerate(times)
        ])
        for node in nodes
    }
    confidence = {
        node: _posterior_anchor_confidence(
            filtered.active_covariance_history_by_node[node],
        )
        for node in nodes
    }
    base_ring = ring_config or replace(
        RingCANNConfig(), num_neurons=72, initialization_duration=0.05,
        internal_dt=0.002,
    )
    records = []
    for bias_deg_h in phase_rate_biases_deg_per_hour:
        bias = np.deg2rad(float(bias_deg_h)) / 3600.0
        for gain in maximum_anchor_gains:
            config = OrbitalDirectionConfig(
                ring=base_ring, maximum_anchor_gain=float(gain),
            )
            for interval_seconds in anchor_intervals_seconds:
                interval_samples = max(1, int(round(interval_seconds / dt)))
                mask = _periodic_anchor_mask(times.size, interval_samples)
                histories = run_orbital_direction_states(
                    timestamps=times,
                    posterior_state_history_by_node=posterior,
                    frame_by_node=frames,
                    anchor_mask_by_node={node: mask for node in nodes},
                    anchor_confidence_by_node=confidence,
                    phase_rate_bias_by_node={node: bias for node in nodes},
                    config=config,
                )
                errors = np.concatenate([
                    _circular_error(history.anchored_phase, truth[node])
                    for node, history in histories.items()
                ])
                records.append(AnchorScanRecord(
                    phase_rate_bias_deg_per_hour=float(bias_deg_h),
                    maximum_anchor_gain=float(gain),
                    anchor_interval_seconds=float(interval_samples * dt),
                    phase_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(errors**2)))),
                    maximum_phase_error_deg=float(
                        np.rad2deg(np.max(np.abs(errors))),
                    ),
                    cue_count=int(sum(
                        np.count_nonzero(history.cue_applied)
                        for history in histories.values()
                    )),
                ))
    return records


def save_anchor_scan(records, output_dir="results/cann"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "walker_direction_anchor_scan.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=AnchorScanRecord.__annotations__)
        writer.writeheader()
        writer.writerows(record.__dict__ for record in records)

    biases = sorted({record.phase_rate_bias_deg_per_hour for record in records})
    gains = sorted({record.maximum_anchor_gain for record in records})
    intervals = sorted({record.anchor_interval_seconds for record in records})
    figure, axes = plt.subplots(1, len(biases), figsize=(5.0 * len(biases), 4.2),
                                squeeze=False, constrained_layout=True)
    for axis, bias in zip(axes[0], biases):
        matrix = np.asarray([
            [next(record.phase_rmse_deg for record in records
                  if record.phase_rate_bias_deg_per_hour == bias
                  and record.maximum_anchor_gain == gain
                  and record.anchor_interval_seconds == interval)
             for interval in intervals]
            for gain in gains
        ])
        image = axis.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
        axis.set_xticks(range(len(intervals)), labels=[f"{v:g}" for v in intervals])
        axis.set_yticks(range(len(gains)), labels=[f"{v:g}" for v in gains])
        axis.set_xlabel("anchor interval (s)")
        axis.set_ylabel("maximum anchor gain")
        axis.set_title(f"rate bias {bias:g} deg/h")
        figure.colorbar(image, ax=axis, label="phase RMSE (deg)")
    figure_path = output / "walker_direction_anchor_scan.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return csv_path, figure_path


if __name__ == "__main__":
    scan = run_walker_direction_anchor_scan()
    print(min(scan, key=lambda record: record.phase_rmse_deg))
    print(save_anchor_scan(scan))
