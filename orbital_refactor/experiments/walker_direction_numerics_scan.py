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
from experiments.walker_direction_navigation_comparison import _circular_error
from experiments.walker_raw_sensor_comparison import _run, _walker_case


@dataclass(frozen=True)
class DirectionNumericsRecord:
    num_neurons: int
    internal_dt: float
    phase_rmse_deg: float
    maximum_phase_error_deg: float
    drift_rate_deg_per_hour: float


def run_walker_direction_numerics_scan(
    *, duration=60.0, dt=5.0, seed=0, node_limit=3,
    neuron_counts=(36, 72, 180), internal_steps=(0.001, 0.002),
):
    """Measure unanchored CANN drift without injecting a rate bias."""
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
    records = []
    no_anchor = {node: np.zeros(times.size, dtype=bool) for node in nodes}
    for neurons in neuron_counts:
        for internal_dt in internal_steps:
            ring = replace(
                RingCANNConfig(), num_neurons=int(neurons),
                internal_dt=float(internal_dt), initialization_duration=0.05,
            )
            histories = run_orbital_direction_states(
                timestamps=times,
                posterior_state_history_by_node=posterior,
                frame_by_node=frames,
                anchor_mask_by_node=no_anchor,
                config=OrbitalDirectionConfig(ring=ring),
            )
            errors = np.stack([
                _circular_error(histories[node].anchored_phase, truth[node])
                for node in nodes
            ])
            mean_error = np.mean(np.unwrap(errors, axis=1), axis=0)
            slope_rad_s = np.polyfit(times, mean_error, deg=1)[0]
            records.append(DirectionNumericsRecord(
                num_neurons=int(neurons), internal_dt=float(internal_dt),
                phase_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(errors**2)))),
                maximum_phase_error_deg=float(
                    np.rad2deg(np.max(np.abs(errors))),
                ),
                drift_rate_deg_per_hour=float(np.rad2deg(slope_rad_s) * 3600.0),
            ))
    return records


def save_direction_numerics_scan(records, output_dir="results/cann"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "walker_direction_numerics_scan.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=DirectionNumericsRecord.__annotations__,
        )
        writer.writeheader()
        writer.writerows(record.__dict__ for record in records)

    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), constrained_layout=True)
    steps = sorted({record.internal_dt for record in records})
    for internal_dt in steps:
        subset = sorted(
            (record for record in records if record.internal_dt == internal_dt),
            key=lambda record: record.num_neurons,
        )
        label = f"internal dt={internal_dt:g} s"
        axes[0].plot([r.num_neurons for r in subset],
                     [r.phase_rmse_deg for r in subset], marker="o", label=label)
        axes[1].plot([r.num_neurons for r in subset],
                     [r.drift_rate_deg_per_hour for r in subset], marker="o",
                     label=label)
    axes[0].set_ylabel("phase RMSE (deg)")
    axes[1].set_ylabel("fitted drift rate (deg/h)")
    for axis in axes:
        axis.set_xlabel("ring neurons")
        axis.grid(alpha=0.25)
        axis.legend()
    figure_path = output / "walker_direction_numerics_scan.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return csv_path, figure_path


if __name__ == "__main__":
    scan = run_walker_direction_numerics_scan()
    for record in scan:
        print(record)
    print(save_direction_numerics_scan(scan))
