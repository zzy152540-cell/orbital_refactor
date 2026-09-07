from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from brain_inspired.line_cann import LineCANNConfig
from brain_inspired.orbital_rt_grid_runner import run_orbital_rt_grid_states
from brain_inspired.orbital_rt_grid_state import OrbitalRTGridConfig
from brain_inspired.orbital_rt_multiscale_runner import (
    run_orbital_rt_multiscale_states,
)
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
from experiments.walker_rt_grid_stress_matrix import _shift_reference_radially


@dataclass(frozen=True)
class WalkerRTMultiScaleComparison:
    metrics: dict[str, dict[str, dict[str, float]]]
    duration: float
    dt: float
    node_count: int
    reference_shift_m: float


def run_walker_rt_multiscale_comparison(
    *, duration=300.0, dt=5.0, seed=0, anchor_interval_samples=2,
    reference_shift_m=15_000.0,
):
    case = _walker_case(seed=seed, duration=duration, dt=dt)
    filtered = _run(case, case["observations"])
    times = np.asarray(case["timestamps"], dtype=float)
    nodes = tuple(filtered.node_ids)
    nominal_references = {
        node: _propagate_reference(case["initial_states"][node], times)
        for node in nodes
    }
    shifted_references = {
        node: _propagate_reference(
            _shift_reference_radially(
                nominal_references[node][:1], -float(reference_shift_m),
            )[0],
            times,
        )
        for node in nodes
    }
    posterior = filtered.active_state_history_by_node
    confidence = {
        node: _posterior_anchor_confidence(
            filtered.active_covariance_history_by_node[node],
        )
        for node in nodes
    }
    mask = _periodic_anchor_mask(times.size, anchor_interval_samples)
    masks = {node: mask for node in nodes}
    narrow_axis = LineCANNConfig(
        num_neurons=81, minimum_value=-10_000.0,
        maximum_value=10_000.0, tuning_width=250.0,
    )
    wide_axis = LineCANNConfig(
        num_neurons=81, minimum_value=-50_000.0,
        maximum_value=50_000.0, tuning_width=1_250.0,
    )
    single_configs = {
        "single_10km": OrbitalRTGridConfig(
            radial=narrow_axis, along_track=narrow_axis,
        ),
        "single_50km": OrbitalRTGridConfig(
            radial=wide_axis, along_track=wide_axis,
        ),
    }
    metrics = {}
    for scenario, references in {
        "nominal_reference": nominal_references,
        "large_reference_offset": shifted_references,
    }.items():
        truth = {
            node: np.asarray([
                _rt_position(
                    time, case["truth"][node][index], references[node][index], node,
                )
                for index, time in enumerate(times)
            ])
            for node in nodes
        }
        metrics[scenario] = {}
        for name, config in single_configs.items():
            start = perf_counter()
            histories = run_orbital_rt_grid_states(
                timestamps=times,
                posterior_state_history_by_node=posterior,
                reference_state_history_by_node=references,
                anchor_mask_by_node=masks,
                anchor_confidence_by_node=confidence,
                config=config,
            )
            elapsed = perf_counter() - start
            values = _metrics(histories, truth)
            values.update({
                "sidecar_runtime_s": float(elapsed),
                "neurons_per_node": 162.0,
                "coarse_cell_change_count": 0.0,
                "mean_cross_scale_disagreement_m": 0.0,
            })
            metrics[scenario][name] = values
        start = perf_counter()
        multiscale = run_orbital_rt_multiscale_states(
            timestamps=times,
            posterior_state_history_by_node=posterior,
            reference_state_history_by_node=references,
            anchor_mask_by_node=masks,
            anchor_confidence_by_node=confidence,
        )
        elapsed = perf_counter() - start
        metrics[scenario]["multiscale"] = _multiscale_metrics(
            multiscale, truth, elapsed,
        )
    return WalkerRTMultiScaleComparison(
        metrics=metrics, duration=float(duration), dt=float(dt),
        node_count=len(nodes), reference_shift_m=float(reference_shift_m),
    )


def _multiscale_metrics(histories, truth, elapsed):
    errors = np.concatenate([
        history.decoded_rt - truth[node] for node, history in histories.items()
    ])
    residuals = np.concatenate([
        history.residual_rt for history in histories.values()
    ])
    return {
        "radial_rmse_m": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
        "along_track_rmse_m": float(np.sqrt(np.mean(errors[:, 1] ** 2))),
        "rt_norm_rmse_m": float(np.sqrt(np.mean(np.sum(errors**2, axis=1)))),
        "maximum_rt_error_m": float(np.max(np.linalg.norm(errors, axis=1))),
        "posterior_tracking_rt_norm_rmse_m": float(np.sqrt(np.mean(
            np.sum(residuals**2, axis=1),
        ))),
        "valid_fraction": float(np.mean(np.concatenate([
            history.valid for history in histories.values()
        ]))),
        "boundary_saturation_fraction": float(np.mean(np.concatenate([
            history.saturated_at_boundary for history in histories.values()
        ]))),
        "anchor_rejection_count": float(np.sum([
            np.count_nonzero(history.anchor_rejected)
            for history in histories.values()
        ])),
        "coarse_cell_change_count": float(np.sum([
            np.count_nonzero(history.coarse_cell_changed)
            for history in histories.values()
        ])),
        "mean_cross_scale_disagreement_m": float(np.mean(np.abs(np.concatenate([
            history.cross_scale_disagreement_rt for history in histories.values()
        ])))),
        "sidecar_runtime_s": float(elapsed),
        "neurons_per_node": 324.0,
    }


if __name__ == "__main__":
    result = run_walker_rt_multiscale_comparison()
    for scenario, methods in result.metrics.items():
        print(scenario)
        for method, values in methods.items():
            print(method, values)
