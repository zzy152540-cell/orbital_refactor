from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

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
class DirectionRobustnessRecord:
    seed: int
    policy: str
    maximum_anchor_gain: float
    phase_rmse_deg: float
    maximum_phase_error_deg: float
    valid_fraction: float
    cue_count: int
    maximum_anchor_age_s: float


def run_walker_direction_robustness(
    *, seeds=(0, 1, 2), duration=120.0, dt=5.0,
    anchor_interval_seconds=10.0, anchor_outage_window=(30.0, 90.0),
    anchor_outage_windows=None,
    phase_rate_bias_deg_per_hour=2.0, anchor_gains=(0.25, 0.4),
    node_limit=None, ring_config=None,
):
    """Compare safe direction-anchor gains over shared per-seed filter runs."""
    records = []
    ring = ring_config or replace(
        RingCANNConfig(), num_neurons=72, internal_dt=0.002,
        initialization_duration=0.1,
    )
    for seed in seeds:
        case = _walker_case(seed=int(seed), duration=duration, dt=dt)
        filtered = _run(case, case["observations"])
        all_nodes = tuple(filtered.node_ids)
        nodes = all_nodes if node_limit is None else all_nodes[:node_limit]
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
        interval = max(1, int(round(anchor_interval_seconds / dt)))
        periodic = _periodic_anchor_mask(times.size, interval)
        outage_windows = (
            (anchor_outage_window,)
            if anchor_outage_windows is None else tuple(anchor_outage_windows)
        )
        policies = [("free_running", 0.0, np.zeros(times.size, dtype=bool))]
        for start, end in outage_windows:
            if end < start:
                raise ValueError("Anchor outage end cannot precede its start.")
            interrupted = periodic & ~((times >= start) & (times <= end))
            duration_label = float(end - start)
            for gain in anchor_gains:
                policy = (
                    f"outage_gain_{gain:g}"
                    if anchor_outage_windows is None
                    else f"outage_{duration_label:g}s_gain_{gain:g}"
                )
                policies.append((policy, float(gain), interrupted))
        rate_bias = np.deg2rad(phase_rate_bias_deg_per_hour) / 3600.0
        for policy, gain, mask in policies:
            histories = run_orbital_direction_states(
                timestamps=times,
                posterior_state_history_by_node=posterior,
                frame_by_node=frames,
                anchor_mask_by_node={node: mask for node in nodes},
                anchor_confidence_by_node=confidence,
                phase_rate_bias_by_node={node: rate_bias for node in nodes},
                config=OrbitalDirectionConfig(
                    ring=ring, maximum_anchor_gain=gain,
                ),
            )
            errors = np.concatenate([
                _circular_error(history.anchored_phase, truth[node])
                for node, history in histories.items()
            ])
            records.append(DirectionRobustnessRecord(
                seed=int(seed), policy=policy, maximum_anchor_gain=gain,
                phase_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(errors**2)))),
                maximum_phase_error_deg=float(
                    np.rad2deg(np.max(np.abs(errors))),
                ),
                valid_fraction=float(np.mean(np.concatenate([
                    history.valid for history in histories.values()
                ]))),
                cue_count=int(sum(np.count_nonzero(history.cue_applied)
                                  for history in histories.values())),
                maximum_anchor_age_s=float(max(
                    np.max(history.anchor_age) for history in histories.values()
                )),
            ))
    return records


def summarize_direction_robustness(records):
    policies = sorted({record.policy for record in records})
    return {
        policy: {
            "mean_phase_rmse_deg": float(np.mean([
                record.phase_rmse_deg for record in records
                if record.policy == policy
            ])),
            "worst_phase_rmse_deg": float(np.max([
                record.phase_rmse_deg for record in records
                if record.policy == policy
            ])),
            "worst_maximum_phase_error_deg": float(np.max([
                record.maximum_phase_error_deg for record in records
                if record.policy == policy
            ])),
        }
        for policy in policies
    }


def save_direction_robustness(records, output_path="results/cann/walker_direction_robustness.csv"):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=DirectionRobustnessRecord.__annotations__)
        writer.writeheader()
        writer.writerows(record.__dict__ for record in records)
    return path


if __name__ == "__main__":
    result = run_walker_direction_robustness()
    print(summarize_direction_robustness(result))
    print(save_direction_robustness(result))
