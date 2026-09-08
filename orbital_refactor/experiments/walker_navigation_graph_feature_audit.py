from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brain_inspired.hierarchical_navigation_place_cells import (
    build_hierarchical_navigation_place_cell_histories,
)
from brain_inspired.navigation_graph_features import (
    NavigationGraphFeatureHistory,
    build_navigation_graph_feature_histories,
)
from experiments.walker_navigation_brain_state_audit import (
    run_walker_navigation_brain_state_audit,
)
from scenarios.walker_scenario import WalkerDeltaConfig


@dataclass(frozen=True)
class WalkerNavigationGraphFeatureAudit:
    histories: dict[str, NavigationGraphFeatureHistory]
    summary: dict[str, object]


def run_walker_navigation_graph_feature_audit(
    *, duration=300.0, dt=5.0, seed=0, anchor_interval_samples=2,
    walker_config: WalkerDeltaConfig | None = None,
):
    navigation = run_walker_navigation_brain_state_audit(
        duration=duration, dt=dt, seed=seed,
        anchor_interval_samples=anchor_interval_samples,
        walker_config=walker_config,
    ).histories
    place = build_hierarchical_navigation_place_cell_histories(
        navigation_by_node=navigation,
    )
    histories = build_navigation_graph_feature_histories(
        navigation_by_node=navigation, place_by_node=place,
    )
    return WalkerNavigationGraphFeatureAudit(
        histories=histories, summary=_summarize(histories),
    )


def _summarize(histories):
    first = next(iter(histories.values()))
    matrix = np.concatenate([
        history.metric_matrix for history in histories.values()
    ])
    standard_deviation = np.std(matrix, axis=0)
    varying = standard_deviation > 1.0e-12
    standardized = (
        (matrix[:, varying] - np.mean(matrix[:, varying], axis=0))
        / standard_deviation[varying]
    )
    if standardized.shape[1] > 0:
        singular = np.linalg.svd(standardized, compute_uv=False)
        energy = singular ** 2
        probability = energy / np.sum(energy)
        effective_rank = float(np.exp(-np.sum(
            probability * np.log(np.maximum(probability, 1.0e-15))
        )))
    else:
        effective_rank = 0.0
    correlated_pairs = []
    varying_names = [
        name for name, keep in zip(first.metric_names, varying) if keep
    ]
    if standardized.shape[1] >= 2:
        correlation = np.corrcoef(standardized, rowvar=False)
        for left in range(len(varying_names)):
            for right in range(left + 1, len(varying_names)):
                if abs(correlation[left, right]) >= 0.98:
                    correlated_pairs.append((
                        varying_names[left], varying_names[right],
                        float(correlation[left, right]),
                    ))
    return {
        "node_count": len(histories),
        "epoch_count": int(first.timestamps.size),
        "metric_count": len(first.metric_names),
        "varying_metric_count": int(np.count_nonzero(varying)),
        "effective_rank": effective_rank,
        "constant_metrics": tuple(
            name for name, keep in zip(first.metric_names, varying) if not keep
        ),
        "highly_correlated_pairs": tuple(correlated_pairs),
        "valid_fraction": float(np.mean(np.concatenate([
            history.valid for history in histories.values()
        ]))),
        "feature_standard_deviation": {
            name: float(value)
            for name, value in zip(first.metric_names, standard_deviation)
        },
    }


if __name__ == "__main__":
    print(run_walker_navigation_graph_feature_audit(duration=30.0).summary)
