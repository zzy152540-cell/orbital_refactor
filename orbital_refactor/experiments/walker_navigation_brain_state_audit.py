from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brain_inspired.navigation_brain_state import (
    NavigationBrainStateHistory,
    build_navigation_brain_states,
)
from brain_inspired.navigation_shadow_quality import build_navigation_shadow_quality
from brain_inspired.orbital_direction_runner import run_orbital_direction_states
from brain_inspired.orbital_phase_adapter import OrbitalPlaneFrame
from brain_inspired.orbital_radial_runner import run_orbital_radial_states
from brain_inspired.orbital_rt_grid_runner import run_orbital_rt_grid_states
from brain_inspired.orbital_rt_grid_state import OrbitalRTGridConfig
from experiments.walker_direction_navigation_comparison import (
    _periodic_anchor_mask,
    _posterior_anchor_confidence,
)
from experiments.walker_raw_sensor_comparison import _run, _walker_case
from experiments.walker_rt_grid_comparison import _propagate_reference
from cooperative.topology import fully_connected_topology
from experiments.walker_filter_setup import build_walker_filter_case
from scenarios.walker_scenario import (
    WalkerDeltaConfig,
    generate_walker_delta_scenario,
)


@dataclass(frozen=True)
class WalkerNavigationBrainStateAudit:
    histories: dict[str, NavigationBrainStateHistory]
    summary: dict[str, object]


def run_walker_navigation_brain_state_audit(
    *, duration=300.0, dt=5.0, seed=0, anchor_interval_samples=2,
    rt_config=None, walker_config: WalkerDeltaConfig | None = None,
):
    case = _build_case(
        seed=seed, duration=duration, dt=dt, walker_config=walker_config,
    )
    filtered = _run(case, case["observations"])
    times = np.asarray(case["timestamps"], dtype=float)
    nodes = tuple(filtered.node_ids)
    posterior = filtered.active_state_history_by_node
    frames = {
        node: OrbitalPlaneFrame.from_state_eci(case["initial_states"][node])
        for node in nodes
    }
    mask = _periodic_anchor_mask(times.size, anchor_interval_samples)
    masks = {node: mask for node in nodes}
    confidence = {
        node: _posterior_anchor_confidence(
            filtered.active_covariance_history_by_node[node],
        )
        for node in nodes
    }
    common = dict(
        timestamps=times, posterior_state_history_by_node=posterior,
        frame_by_node=frames, anchor_mask_by_node=masks,
        anchor_confidence_by_node=confidence,
    )
    direction = run_orbital_direction_states(**common)
    radial = run_orbital_radial_states(**common)
    references = {
        node: _propagate_reference(case["initial_states"][node], times)
        for node in nodes
    }
    rt = run_orbital_rt_grid_states(
        timestamps=times, posterior_state_history_by_node=posterior,
        reference_state_history_by_node=references,
        anchor_mask_by_node=masks, anchor_confidence_by_node=confidence,
        config=rt_config or OrbitalRTGridConfig(),
    )
    quality = build_navigation_shadow_quality(
        direction_by_node=direction, radial_by_node=radial,
    )
    histories = build_navigation_brain_states(
        direction_by_node=direction, rt_by_node=rt, quality_by_node=quality,
    )
    return WalkerNavigationBrainStateAudit(
        histories=histories, summary=_summarize(histories),
    )


def _build_case(*, seed, duration, dt, walker_config):
    if walker_config is None:
        return _walker_case(seed=seed, duration=duration, dt=dt)
    times = np.arange(0.0, duration + 0.5 * dt, dt)
    scenario = generate_walker_delta_scenario(
        timestamps=times, config=walker_config,
    )
    return build_walker_filter_case(
        seed=seed, duration=duration, dt=dt, maximum_range=7_000e3,
        topology=fully_connected_topology(scenario.node_ids),
        truth_history_by_node=scenario.truth_state_history_by_node,
        topology_type="navigation_brain_geometry_audit",
    )


def _summarize(histories):
    first = next(iter(histories.values()))
    names = first.feature_names
    stacked = np.concatenate([
        history.feature_matrix for history in histories.values()
    ])
    standard_deviation = np.std(stacked, axis=0)
    varying = standard_deviation > 1.0e-12
    standardized = (
        (stacked[:, varying] - np.mean(stacked[:, varying], axis=0))
        / standard_deviation[varying]
    )
    singular_values = np.linalg.svd(standardized, compute_uv=False)
    energy = singular_values**2
    probabilities = energy / np.sum(energy)
    effective_rank = float(np.exp(-np.sum(
        probabilities * np.log(np.maximum(probabilities, 1.0e-15))
    )))
    varying_names = [name for name, keep in zip(names, varying) if keep]
    correlation = np.corrcoef(standardized, rowvar=False)
    np.fill_diagonal(correlation, 0.0)
    pair_index = np.unravel_index(
        np.argmax(np.abs(correlation)), correlation.shape,
    )
    total_variance = np.var(stacked, axis=0)
    node_means = np.stack([
        np.mean(history.feature_matrix, axis=0) for history in histories.values()
    ])
    between_variance = np.var(node_means, axis=0)
    separation = np.divide(
        between_variance, total_variance,
        out=np.zeros_like(total_variance), where=total_variance > 1.0e-12,
    )
    return {
        "node_count": len(histories),
        "epoch_count": int(first.timestamps.size),
        "feature_count": len(names),
        "varying_feature_count": int(np.count_nonzero(varying)),
        "effective_rank": effective_rank,
        "maximum_absolute_correlation": float(abs(correlation[pair_index])),
        "most_correlated_feature_pair": (
            varying_names[pair_index[0]], varying_names[pair_index[1]],
        ),
        "mean_node_separation_ratio": float(np.mean(separation[varying])),
        "valid_fraction": float(np.mean(np.concatenate([
            history.valid for history in histories.values()
        ]))),
        "boundary_saturation_fraction": float(np.mean(np.concatenate([
            history.boundary_saturated for history in histories.values()
        ]))),
        "feature_standard_deviation": {
            name: float(value) for name, value in zip(names, standard_deviation)
        },
    }


if __name__ == "__main__":
    print(run_walker_navigation_brain_state_audit().summary)
