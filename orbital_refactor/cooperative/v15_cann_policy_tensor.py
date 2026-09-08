from __future__ import annotations

from dataclasses import replace

import numpy as np

from brain_inspired.navigation_graph_features import (
    NAVIGATION_GRAPH_NODE_METRIC_NAMES,
)
from cooperative.topology_policy import GraphObservation
from cooperative.v15_policy_tensor import (
    V15PolicyTensor,
    tensorize_v15_policy_observation,
)


CANN_NODE_FEATURE_NAMES = (
    "cann_direction_concentration",
    "normalized_cann_abs_direction_residual",
    "log1p_cann_rt_residual_norm",
    "log1p_cann_anchor_age",
    "cann_shadow_quality",
    "cann_delayed_feedback_quality",
    "cann_boundary_saturated",
    "cann_anchor_rejected",
    "cann_reference_rebased",
    "place_fine_scale_active",
    "place_boundary_saturated",
    "place_scale_transition",
)
CANN_AVAILABILITY_FEATURE_NAMES = tuple(
    f"available_{name}" for name in NAVIGATION_GRAPH_NODE_METRIC_NAMES
)

_BOUNDED_METRICS = {
    "cann_direction_concentration", "cann_shadow_quality",
    "cann_delayed_feedback_quality", "cann_boundary_saturated",
    "cann_anchor_rejected", "cann_reference_rebased",
    "place_fine_scale_active", "place_boundary_saturated",
    "place_scale_transition",
}


def tensorize_v15_cann_policy_observation(
    observation: GraphObservation, *, direction_residual_scale=np.pi,
    rt_residual_scale=100.0, anchor_age_scale=10.0, **base_scales,
) -> V15PolicyTensor:
    """Append masked CANN diagnostics without modifying the v15.0 schema."""
    scales = (direction_residual_scale, rt_residual_scale, anchor_age_scale)
    if any(not np.isfinite(value) or value <= 0.0 for value in scales):
        raise ValueError("CANN policy tensor scales must be finite and positive.")
    base = tensorize_v15_policy_observation(observation, **base_scales)
    nodes = {node.node_id: node for node in observation.nodes}
    rows = []
    masks = []
    for node_id in base.node_ids:
        metrics = dict(nodes[node_id].estimator_metrics)
        raw = {}
        available = {}
        for name in NAVIGATION_GRAPH_NODE_METRIC_NAMES:
            available[name] = name in metrics
            value = float(metrics.get(name, 0.0))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"CANN policy metric {name} must be nonnegative.")
            if name in _BOUNDED_METRICS and value > 1.0:
                raise ValueError(f"CANN policy metric {name} must lie in [0, 1].")
            raw[name] = value
        rows.append((
            raw["cann_direction_concentration"],
            raw["cann_abs_direction_residual_rad"]
            / direction_residual_scale,
            np.log1p(raw["cann_rt_residual_norm_m"] / rt_residual_scale),
            np.log1p(raw["cann_anchor_age_s"] / anchor_age_scale),
            raw["cann_shadow_quality"],
            raw["cann_delayed_feedback_quality"],
            raw["cann_boundary_saturated"],
            raw["cann_anchor_rejected"],
            raw["cann_reference_rebased"],
            raw["place_fine_scale_active"],
            raw["place_boundary_saturated"],
            raw["place_scale_transition"],
        ))
        masks.append(tuple(
            float(available[name])
            for name in NAVIGATION_GRAPH_NODE_METRIC_NAMES
        ))
    appended = np.column_stack((np.asarray(rows), np.asarray(masks)))
    node_features = np.column_stack((base.node_features, appended))
    node_features.setflags(write=False)
    return replace(
        base,
        schema_version="v15.1-cann-policy-normalized",
        node_feature_names=(
            *base.node_feature_names, *CANN_NODE_FEATURE_NAMES,
            *CANN_AVAILABILITY_FEATURE_NAMES,
        ),
        node_features=node_features,
    )
