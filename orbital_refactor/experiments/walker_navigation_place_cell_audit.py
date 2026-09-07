from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brain_inspired.navigation_place_cells import (
    NavigationPlaceCellConfig,
    NavigationPlaceCellHistory,
    build_navigation_place_cell_histories,
)
from experiments.walker_navigation_brain_state_audit import (
    WalkerNavigationBrainStateAudit,
    run_walker_navigation_brain_state_audit,
)
from scenarios.walker_scenario import WalkerDeltaConfig


@dataclass(frozen=True)
class WalkerNavigationPlaceCellAudit:
    navigation: WalkerNavigationBrainStateAudit
    histories: dict[str, NavigationPlaceCellHistory]
    summary: dict[str, object]


def run_walker_navigation_place_cell_audit(
    *, duration=300.0, dt=5.0, seed=0, anchor_interval_samples=2,
    walker_config: WalkerDeltaConfig | None = None,
    place_config: NavigationPlaceCellConfig | None = None,
):
    """Audit the read-only place-cell display over an unchanged Walker run."""
    navigation = run_walker_navigation_brain_state_audit(
        duration=duration, dt=dt, seed=seed,
        anchor_interval_samples=anchor_interval_samples,
        walker_config=walker_config,
    )
    histories = build_navigation_place_cell_histories(
        navigation_by_node=navigation.histories, config=place_config,
    )
    return WalkerNavigationPlaceCellAudit(
        navigation=navigation,
        histories=histories,
        summary=_summarize(navigation.histories, histories),
    )


def _summarize(navigation_by_node, place_by_node):
    phase_errors = []
    rt_errors = []
    activities = []
    node_means = []
    for node, place in place_by_node.items():
        navigation = navigation_by_node[node]
        phase_errors.append(np.arctan2(
            np.sin(place.decoded_phase - navigation.decoded_phase),
            np.cos(place.decoded_phase - navigation.decoded_phase),
        ))
        rt_errors.append(place.decoded_rt - navigation.decoded_rt)
        activities.append(place.activity)
        node_means.append(np.mean(place.activity, axis=0))
    phase_error = np.concatenate(phase_errors)
    rt_error = np.concatenate(rt_errors)
    activity = np.concatenate(activities)
    singular_values = np.linalg.svd(
        activity - np.mean(activity, axis=0), compute_uv=False,
    )
    energy = singular_values ** 2
    if np.sum(energy) > 0.0:
        probability = energy / np.sum(energy)
        effective_rank = float(np.exp(-np.sum(
            probability * np.log(np.maximum(probability, 1.0e-15))
        )))
    else:
        effective_rank = 0.0
    total_variance = np.var(activity, axis=0)
    between_variance = np.var(np.stack(node_means), axis=0)
    separation = np.divide(
        between_variance, total_variance, out=np.zeros_like(total_variance),
        where=total_variance > 1.0e-15,
    )
    return {
        "node_count": len(place_by_node),
        "epoch_count": int(next(iter(place_by_node.values())).timestamps.size),
        "cell_count": int(activity.shape[1]),
        "valid_fraction": float(np.mean(np.concatenate([
            history.valid for history in place_by_node.values()
        ]))),
        "phase_reconstruction_rmse_rad": float(np.sqrt(np.mean(
            phase_error ** 2
        ))),
        "radial_reconstruction_rmse_m": float(np.sqrt(np.mean(
            rt_error[:, 0] ** 2
        ))),
        "along_track_reconstruction_rmse_m": float(np.sqrt(np.mean(
            rt_error[:, 1] ** 2
        ))),
        "mean_peak_activity": float(np.mean(np.concatenate([
            history.peak_activity for history in place_by_node.values()
        ]))),
        "mean_normalized_entropy": float(np.mean(np.concatenate([
            history.normalized_entropy for history in place_by_node.values()
        ]))),
        "activity_effective_rank": effective_rank,
        "mean_node_separation_ratio": float(np.mean(
            separation[total_variance > 1.0e-15]
        )),
        "interpretation": (
            "Place-cell activity is a deterministic nonlinear display of "
            "direction and RT state, not an independent observation."
        ),
    }


if __name__ == "__main__":
    print(run_walker_navigation_place_cell_audit().summary)
