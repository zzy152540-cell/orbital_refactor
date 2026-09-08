from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brain_inspired.hierarchical_navigation_place_cells import (
    HierarchicalNavigationPlaceCellHistory,
)
from brain_inspired.navigation_brain_state import NavigationBrainStateHistory

Array = np.ndarray

NAVIGATION_GRAPH_NODE_METRIC_NAMES = (
    "cann_direction_concentration",
    "cann_abs_direction_residual_rad",
    "cann_rt_residual_norm_m",
    "cann_anchor_age_s",
    "cann_shadow_quality",
    "cann_delayed_feedback_quality",
    "cann_boundary_saturated",
    "cann_anchor_rejected",
    "cann_reference_rebased",
    "place_fine_scale_active",
    "place_boundary_saturated",
    "place_scale_transition",
)


@dataclass(frozen=True)
class NavigationGraphFeatureHistory:
    node_id: str
    timestamps: Array
    metric_names: tuple[str, ...]
    metric_matrix: Array
    valid: Array

    def metrics_at(self, index: int) -> dict[str, float]:
        if not 0 <= int(index) < self.timestamps.size:
            raise IndexError("Navigation graph feature index is out of range.")
        return {
            name: float(value)
            for name, value in zip(self.metric_names, self.metric_matrix[index])
        }


def build_navigation_graph_feature_histories(
    *, navigation_by_node: dict[str, NavigationBrainStateHistory],
    place_by_node: dict[str, HierarchicalNavigationPlaceCellHistory],
):
    """Build deployment-available compact metrics; no truth or future data."""
    if not navigation_by_node or set(navigation_by_node) != set(place_by_node):
        raise ValueError(
            "Navigation and place histories must cover identical nonempty nodes."
        )
    result = {}
    for node, navigation in navigation_by_node.items():
        place = place_by_node[node]
        if not np.array_equal(navigation.timestamps, place.timestamps):
            raise ValueError("Navigation and place timestamps must align.")
        feature_index = {
            name: index for index, name in enumerate(navigation.feature_names)
        }
        required = {
            "direction_bump_concentration", "direction_residual_rad",
            "radial_residual_m", "along_residual_m", "anchor_age_s",
        }
        if required - set(feature_index):
            raise ValueError("Navigation history lacks required compact features.")
        features = navigation.feature_matrix
        direction_residual = features[
            :, feature_index["direction_residual_rad"]
        ]
        rt_residual = features[:, [
            feature_index["radial_residual_m"],
            feature_index["along_residual_m"],
        ]]
        matrix = np.column_stack([
            features[:, feature_index["direction_bump_concentration"]],
            np.abs(direction_residual), np.linalg.norm(rt_residual, axis=1),
            features[:, feature_index["anchor_age_s"]],
            navigation.shadow_quality, navigation.delayed_feedback_quality,
            navigation.boundary_saturated.astype(float),
            navigation.anchor_rejected.astype(float),
            np.any(navigation.reference_rebased, axis=1).astype(float),
            place.fine_scale_active.astype(float),
            place.boundary_saturated.astype(float),
            place.scale_transition.astype(float),
        ])
        if np.any(~np.isfinite(matrix)) or np.any(matrix < 0.0):
            raise ValueError(
                "Navigation graph metrics must be finite and nonnegative."
            )
        result[node] = NavigationGraphFeatureHistory(
            node_id=node, timestamps=navigation.timestamps.copy(),
            metric_names=NAVIGATION_GRAPH_NODE_METRIC_NAMES,
            metric_matrix=matrix, valid=navigation.valid & place.valid,
        )
    return result


def navigation_graph_metrics_at_timestamp(
    histories: dict[str, NavigationGraphFeatureHistory], *, timestamp: float,
):
    """Extract one aligned per-node metric mapping for GraphObservation."""
    if not histories or not np.isfinite(timestamp):
        raise ValueError("Feature histories and timestamp must be valid.")
    result = {}
    for node, history in histories.items():
        matches = np.flatnonzero(np.isclose(
            history.timestamps, float(timestamp), rtol=0.0, atol=1.0e-9,
        ))
        if matches.size != 1:
            raise ValueError(
                f"Node {node} has no unique feature sample at the timestamp."
            )
        index = int(matches[0])
        if not history.valid[index]:
            raise ValueError(f"Node {node} has invalid CANN features.")
        result[node] = history.metrics_at(index)
    return result
