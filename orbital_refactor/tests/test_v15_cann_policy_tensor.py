import numpy as np
import pytest

from brain_inspired.navigation_graph_features import (
    NAVIGATION_GRAPH_NODE_METRIC_NAMES,
)
from cooperative.network_schmidt_orchestrator import NetworkSchmidtOrchestrator
from cooperative.online_graph_observation import build_online_graph_observation
from cooperative.topology import fully_connected_topology
from cooperative.v15_cann_policy_tensor import (
    CANN_AVAILABILITY_FEATURE_NAMES,
    CANN_NODE_FEATURE_NAMES,
    tensorize_v15_cann_policy_observation,
)
from cooperative.v15_policy_tensor import tensorize_v15_policy_observation


def _observation(additional=None):
    nodes = ("a", "b", "c")
    orchestrator = NetworkSchmidtOrchestrator(
        initial_state_by_node={
            "a": np.array([7e6, 0, 0, 0, 7e3, 0]),
            "b": np.array([7e6 + 3e3, 4e3, 0, 0, 7e3, 0]),
            "c": np.array([7e6, 0, 12e3, 0, 7e3, 0]),
        },
        initial_covariance_by_node={node: np.eye(6) for node in nodes},
        topology=fully_connected_topology(nodes),
    )
    return build_online_graph_observation(
        orchestrator, additional_node_metrics_by_node=additional,
    )


def _complete_metrics():
    return {
        "cann_direction_concentration": 0.8,
        "cann_abs_direction_residual_rad": 0.2,
        "cann_rt_residual_norm_m": 50.0,
        "cann_anchor_age_s": 20.0,
        "cann_shadow_quality": 0.7,
        "cann_delayed_feedback_quality": 0.6,
        "cann_boundary_saturated": 0.0,
        "cann_anchor_rejected": 1.0,
        "cann_reference_rebased": 0.0,
        "place_fine_scale_active": 1.0,
        "place_boundary_saturated": 0.0,
        "place_scale_transition": 1.0,
    }


def test_v151_appends_cann_values_and_masks_without_changing_v150():
    observation = _observation({"a": _complete_metrics()})
    base = tensorize_v15_policy_observation(observation)
    cann = tensorize_v15_cann_policy_observation(observation)
    assert base.schema_version == "v15.0-policy-normalized"
    assert cann.schema_version == "v15.1-cann-policy-normalized"
    assert cann.node_features.shape[1] == base.node_features.shape[1] + 24
    np.testing.assert_allclose(
        cann.node_features[:, :base.node_features.shape[1]], base.node_features,
    )
    node_a = cann.node_ids.index("a")
    node_b = cann.node_ids.index("b")
    mask_start = len(base.node_feature_names) + len(CANN_NODE_FEATURE_NAMES)
    assert np.all(cann.node_features[node_a, mask_start:] == 1.0)
    assert np.all(cann.node_features[node_b, mask_start:] == 0.0)
    assert cann.node_feature_names[-12:] == CANN_AVAILABILITY_FEATURE_NAMES
    assert len(NAVIGATION_GRAPH_NODE_METRIC_NAMES) == 12
    with pytest.raises(ValueError):
        cann.node_features[0, 0] = 0.0


def test_v151_rejects_invalid_cann_metric_and_scale():
    metrics = _complete_metrics()
    metrics["cann_shadow_quality"] = 1.1
    with pytest.raises(ValueError, match="lie in"):
        tensorize_v15_cann_policy_observation(_observation({"a": metrics}))
    with pytest.raises(ValueError, match="scales"):
        tensorize_v15_cann_policy_observation(
            _observation(), rt_residual_scale=0.0,
        )


def test_v151_all_missing_cann_features_have_zero_values_and_masks():
    observation = _observation()
    base = tensorize_v15_policy_observation(observation)
    cann = tensorize_v15_cann_policy_observation(observation)
    appended = cann.node_features[:, base.node_features.shape[1]:]
    assert np.all(appended == 0.0)
