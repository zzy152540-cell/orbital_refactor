from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from brain_inspired.orbital_rt_grid_runner import run_orbital_rt_grid_states
from brain_inspired.orbital_rt_grid_state import OrbitalRTGridConfig
from experiments.walker_direction_navigation_comparison import (
    _periodic_anchor_mask,
    _posterior_anchor_confidence,
)
from experiments.walker_raw_sensor_comparison import _run, _walker_case
from experiments.walker_rt_grid_comparison import (
    _metrics,
    _propagate_reference,
    _rt_position,
)


@dataclass(frozen=True)
class WalkerRTGridStressResult:
    timestamps: np.ndarray
    metrics: dict[str, dict[str, dict[str, float]]]
    injected_bias_rt_by_case: dict[str, np.ndarray]
    outlier_count_per_node: int


def run_walker_rt_grid_stress_matrix(
    *, duration=300.0, dt=5.0, seed=0, anchor_interval_samples=2,
    step_fraction=0.5, outlier_interval_samples=10,
    outlier_rt_m=(500.0, -500.0), extreme_bias_rt_mps=(40.0, 40.0),
    grid_config=None,
):
    """Compare fixed and adaptive RT sidecars under controlled stressors."""
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
            _rt_position(time, case["truth"][node][index], references[node][index], node)
            for index, time in enumerate(times)
        ])
        for node in nodes
    }
    far_references = {
        node: _propagate_reference(
            _shift_reference_radially(references[node][:1], -15_000.0)[0], times,
        )
        for node in nodes
    }
    far_truth_rt = {
        node: np.asarray([
            _rt_position(
                time, case["truth"][node][index], far_references[node][index], node,
            )
            for index, time in enumerate(times)
        ])
        for node in nodes
    }
    posterior = filtered.active_state_history_by_node
    confidence = {
        node: _posterior_anchor_confidence(
            filtered.active_covariance_history_by_node[node],
        )
        for node in nodes
    }
    periodic = _periodic_anchor_mask(times.size, anchor_interval_samples)
    anchor_masks = {node: periodic.copy() for node in nodes}
    no_anchors = {node: np.zeros(times.size, dtype=bool) for node in nodes}
    normalized_time = ((times - times[0]) / max(times[-1] - times[0], 1.0))[:, None]
    nominal = np.array([0.1, 0.2])
    profiles = {
        "constant_bias": np.broadcast_to(nominal, (times.size, 2)).copy(),
        "step_bias": np.where(
            (normalized_time >= float(step_fraction)), nominal, 0.0,
        ),
        "ramp_bias": normalized_time * nominal,
        "anchor_outliers": np.broadcast_to(nominal, (times.size, 2)).copy(),
        "extreme_unanchored": np.broadcast_to(
            np.asarray(extreme_bias_rt_mps, dtype=float), (times.size, 2),
        ).copy(),
        "large_reference_offset": np.zeros((times.size, 2)),
    }
    outlier_mask = periodic & (
        np.arange(times.size) % max(int(outlier_interval_samples), 1) == 0
    )
    anchors_with_outliers = {
        node: _inject_rt_position_outliers(
            posterior[node], references[node], outlier_mask, outlier_rt_m,
        )
        for node in nodes
    }
    base_config = grid_config or OrbitalRTGridConfig()
    configs = {
        "fixed": base_config,
        "adaptive": replace(base_config, rolling_bias_enabled=True),
    }
    metrics = {}
    for case_name, profile in profiles.items():
        mask = no_anchors if case_name == "extreme_unanchored" else anchor_masks
        anchor_states = (anchors_with_outliers if case_name == "anchor_outliers"
                         else posterior)
        active_references = (far_references if case_name == "large_reference_offset"
                             else references)
        active_truth = (far_truth_rt if case_name == "large_reference_offset"
                        else truth_rt)
        metrics[case_name] = {}
        for policy_name, config in configs.items():
            histories = run_orbital_rt_grid_states(
                timestamps=times,
                posterior_state_history_by_node=posterior,
                reference_state_history_by_node=active_references,
                anchor_state_history_by_node=anchor_states,
                anchor_mask_by_node=mask,
                anchor_confidence_by_node=confidence,
                rate_bias_rt_by_node={node: profile for node in nodes},
                config=config,
            )
            metrics[case_name][policy_name] = _metrics(histories, active_truth)
    return WalkerRTGridStressResult(
        timestamps=times,
        metrics=metrics,
        injected_bias_rt_by_case={name: values.copy()
                                  for name, values in profiles.items()},
        outlier_count_per_node=int(np.count_nonzero(outlier_mask)),
    )


def _inject_rt_position_outliers(states, references, mask, outlier_rt_m):
    corrupted = np.asarray(states, dtype=float).copy()
    delta = np.asarray(outlier_rt_m, dtype=float).reshape(-1)
    if delta.shape != (2,) or np.any(~np.isfinite(delta)):
        raise ValueError("outlier_rt_m must be a finite 2-vector.")
    for index in np.flatnonzero(mask):
        radial = references[index, :3]
        radial = radial / np.linalg.norm(radial)
        normal = np.cross(references[index, :3], references[index, 3:])
        normal = normal / np.linalg.norm(normal)
        along = np.cross(normal, radial)
        corrupted[index, :3] += delta[0] * radial + delta[1] * along
    return corrupted


def _shift_reference_radially(references, radial_shift):
    shifted = np.asarray(references, dtype=float).copy()
    radial_unit = shifted[:, :3] / np.linalg.norm(
        shifted[:, :3], axis=1, keepdims=True,
    )
    shifted[:, :3] += float(radial_shift) * radial_unit
    return shifted


if __name__ == "__main__":
    result = run_walker_rt_grid_stress_matrix()
    for case_name, policies in result.metrics.items():
        print(case_name, {
            name: values["rt_norm_rmse_m"] for name, values in policies.items()
        })
