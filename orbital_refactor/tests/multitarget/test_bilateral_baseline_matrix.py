import json

import pytest

from experiments.bilateral_multitarget_frontend import (
    BilateralFrontendConfig,
    run_bilateral_baseline_matrix,
)
from experiments.run_bilateral_baseline_matrix import _arm_payload, _jsonable


EXPECTED_ARMS = {
    "centralized_radar_only",
    "centralized_infrared_only",
    "centralized_optical_only",
    "centralized_multimodal_no_cann",
    "centralized_multimodal_cann",
    "fixed_topology_multimodal_no_cann",
    "fixed_topology_multimodal_cann",
}

EXPECTED_LOCAL_NODE_BASELINES = {
    "single_node_multimodal_no_cann",
    "single_node_multimodal_cann",
    "independent_15_node_multimodal_no_cann",
    "independent_15_node_multimodal_cann",
}


@pytest.fixture(scope="module")
def baseline_matrix():
    return run_bilateral_baseline_matrix(BilateralFrontendConfig(
        duration_seconds=3.0,
        random_seed=13,
    ))


def test_shared_frame_baseline_matrix_runs_all_declared_arms(baseline_matrix):
    result = baseline_matrix

    assert set(result.arms) == EXPECTED_ARMS
    assert set(result.tracking_modalities_by_arm) == EXPECTED_ARMS
    assert set(result.local_node_baselines) == EXPECTED_LOCAL_NODE_BASELINES
    assert result.uses_common_multimodal_initialization
    assert len(result.message_count_by_timestamp) == 4
    for name, arm in result.arms.items():
        assert arm.initialized_track_count == 15
        assert arm.final_active_track_count == 15
        assert arm.identity_switch_count == 0
        assert len(arm.position_rmse_m_by_target) == 15
        if name.endswith(("radar_only", "infrared_only", "optical_only")):
            assert arm.tracking_modality_restriction_start_seconds is not None
        else:
            assert arm.tracking_modality_restriction_start_seconds is None
    assert result.arms[
        "centralized_multimodal_no_cann"
    ].cann_preprocessed_observation_count == 0
    assert result.arms[
        "centralized_multimodal_cann"
    ].cann_preprocessed_observation_count > 0
    assert result.arms[
        "fixed_topology_multimodal_no_cann"
    ].consensus_phase_metrics
    single = result.local_node_baselines["single_node_multimodal_no_cann"]
    independent = result.local_node_baselines[
        "independent_15_node_multimodal_no_cann"
    ]
    assert single.node_count == 1
    assert independent.node_count == 15
    assert len(single.node_ids) == 1
    assert len(independent.node_ids) == 15
    assert single.identity_switch_count == 0
    assert independent.identity_switch_count == 0


def test_baseline_arm_payload_is_json_serializable(baseline_matrix):
    result = baseline_matrix
    payload = {
        name: _arm_payload(arm) for name, arm in result.arms.items()
    }

    encoded = json.dumps(_jsonable(payload), allow_nan=False)

    assert "centralized_radar_only" in encoded
    assert "consensus_phase_metrics" in encoded
