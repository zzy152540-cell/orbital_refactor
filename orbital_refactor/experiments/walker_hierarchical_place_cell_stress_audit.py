from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brain_inspired.hierarchical_navigation_place_cells import (
    HierarchicalNavigationPlaceCellConfig,
    HierarchicalNavigationPlaceCellEncoder,
)
from experiments.walker_rt_grid_stress_matrix import (
    WalkerRTGridStressResult,
    run_walker_rt_grid_stress_matrix,
)


@dataclass(frozen=True)
class WalkerHierarchicalPlaceCellStressAudit:
    rt_stress: WalkerRTGridStressResult
    metrics: dict[str, dict[str, dict[str, float]]]


def run_walker_hierarchical_place_cell_stress_audit(
    *, duration=300.0, dt=5.0, seed=0, anchor_interval_samples=2,
    place_config: HierarchicalNavigationPlaceCellConfig | None = None,
):
    """Encode actual RT-sidecar stress histories without estimator feedback."""
    stress = run_walker_rt_grid_stress_matrix(
        duration=duration, dt=dt, seed=seed,
        anchor_interval_samples=anchor_interval_samples,
    )
    metrics = {}
    for case_name, policies in stress.histories_by_case.items():
        metrics[case_name] = {}
        truth = stress.truth_rt_by_case[case_name]
        for policy_name, histories in policies.items():
            metrics[case_name][policy_name] = _evaluate(
                histories=histories, truth_by_node=truth,
                phase_by_node=stress.phase_by_node, config=place_config,
            )
    return WalkerHierarchicalPlaceCellStressAudit(
        rt_stress=stress, metrics=metrics,
    )


def _evaluate(*, histories, truth_by_node, phase_by_node, config):
    representation_errors = []
    sidecar_errors = []
    entropies = []
    peaks = []
    fine_active = []
    transitions = []
    saturated = []
    for node, history in histories.items():
        encoder = HierarchicalNavigationPlaceCellEncoder(config)
        outputs = [encoder.encode(
            phase=phase_by_node[node][index],
            radial_position=history.anchored_rt[index, 0],
            along_track_position=history.anchored_rt[index, 1],
        ) for index in range(history.timestamps.size)]
        decoded = np.asarray([[
            item.decoded_radial_position,
            item.decoded_along_track_position,
        ] for item in outputs])
        representation_errors.append(np.linalg.norm(
            decoded - history.anchored_rt, axis=1,
        ))
        sidecar_errors.append(np.linalg.norm(
            history.anchored_rt - truth_by_node[node], axis=1,
        ))
        entropies.append(np.asarray([
            item.normalized_entropy for item in outputs
        ]))
        peaks.append(np.asarray([item.peak_activity for item in outputs]))
        fine_active.append(np.asarray([
            item.fine_scale_active for item in outputs
        ], dtype=bool))
        transitions.append(np.asarray([
            item.scale_transition for item in outputs
        ], dtype=bool))
        saturated.append(np.asarray([
            item.boundary_saturated for item in outputs
        ], dtype=bool))
    representation_error = np.concatenate(representation_errors)
    sidecar_error = np.concatenate(sidecar_errors)
    entropy = np.concatenate(entropies)
    peak = np.concatenate(peaks)
    return {
        "representation_rmse_m": float(np.sqrt(np.mean(
            representation_error ** 2
        ))),
        "sidecar_truth_rmse_m": float(np.sqrt(np.mean(sidecar_error ** 2))),
        "fine_scale_active_fraction": float(np.mean(np.concatenate(
            fine_active
        ))),
        "scale_transition_count": int(np.count_nonzero(np.concatenate(
            transitions
        ))),
        "boundary_saturation_fraction": float(np.mean(np.concatenate(
            saturated
        ))),
        "entropy_error_correlation": _safe_correlation(entropy, sidecar_error),
        "peak_error_correlation": _safe_correlation(peak, sidecar_error),
    }


def _safe_correlation(first, second):
    if np.std(first) <= 1.0e-12 or np.std(second) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


if __name__ == "__main__":
    result = run_walker_hierarchical_place_cell_stress_audit()
    for case_name, policies in result.metrics.items():
        print(case_name, policies)
