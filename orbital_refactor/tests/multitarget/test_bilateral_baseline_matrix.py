import json
from types import SimpleNamespace

import pytest

from experiments.bilateral_multitarget_frontend import (
    BilateralFrontendConfig,
    run_bilateral_baseline_matrix,
    run_bilateral_cann_acceptance_arm,
    _matched_view_metrics,
)
from experiments.run_bilateral_baseline_matrix import _arm_payload, _jsonable


EXPECTED_ARMS = {
    "centralized_radar_only",
    "centralized_infrared_only",
    "centralized_optical_only",
    "centralized_multimodal_no_cann",
    "centralized_multimodal_cann",
    "single_observer_joint_multimodal_no_cann",
    "single_observer_joint_multimodal_cann",
    "fixed_topology_multimodal_no_cann",
    "fixed_topology_multimodal_cann",
    "fixed_topology_joint_multimodal_cann",
}

EXPECTED_LOCAL_NODE_BASELINES = {
    "single_node_optical_only",
    "single_node_multimodal_no_cann",
    "single_node_multimodal_cann",
    "independent_15_node_multimodal_no_cann",
    "independent_15_node_multimodal_cann",
    "independent_15_node_joint_multimodal_cann",
}


@pytest.fixture(scope="module")
def baseline_matrix():
    return run_bilateral_baseline_matrix(BilateralFrontendConfig(
        duration_seconds=3.0,
        random_seed=13,
        backup_radial_range_sigma_m=1.0,
        backup_radial_range_rate_sigma_mps=0.0125,
        independent_radial_backup_channel=True,
    ))


def test_shared_frame_baseline_matrix_runs_all_declared_arms(baseline_matrix):
    result = baseline_matrix

    assert set(result.arms) == EXPECTED_ARMS
    assert set(result.tracking_modalities_by_arm) == EXPECTED_ARMS
    assert set(result.local_node_baselines) == EXPECTED_LOCAL_NODE_BASELINES
    assert result.uses_common_multimodal_initialization
    assert result.matched_single_observer_cann_metrics.common_sample_count > 0
    assert result.matched_single_observer_joint_cann_metrics.common_sample_count > 0
    assert set(result.matched_single_observer_ablation_metrics) == {
        "federatedCannIncrement",
        "jointFusionWithoutCann",
        "jointCannIncrement",
        "jointFusionWithCannCombined",
    }
    assert set(result.matched_single_observer_cann_vs_modality) == {
        "radar", "infrared", "optical",
    }
    assert set(result.matched_single_observer_joint_cann_vs_modality) == {
        "radar", "infrared", "optical",
    }
    assert all(
        metrics.common_sample_count > 0
        for metrics in result.matched_single_observer_cann_vs_modality.values()
    )
    assert set(result.matched_independent_consensus_metrics_by_cann) == {
        "withoutCann", "incrementalCann",
    }
    assert result.matched_joint_independent_consensus_metrics.common_sample_count > 0
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
    central_bound = result.arms[
        "centralized_multimodal_no_cann"
    ].matched_local_central_metrics
    assert central_bound.common_sample_count > 0
    assert central_bound.common_sample_count == central_bound.local_sample_count
    single = result.local_node_baselines["single_node_multimodal_no_cann"]
    independent = result.local_node_baselines[
        "independent_15_node_multimodal_no_cann"
    ]
    assert single.node_count == 1
    assert independent.node_count == 15
    assert single.scored_sample_count > 0
    assert len(single.scored_sample_key_sha256) == 64
    assert single.scored_truth_ids
    assert len(single.node_ids) == 1
    assert len(independent.node_ids) == 15
    assert single.identity_switch_count == 0
    assert independent.identity_switch_count == 0
    assert (
        single.scored_sample_key_sha256
        == result.local_node_baselines[
            "single_node_multimodal_cann"
        ].scored_sample_key_sha256
    )


def test_matched_view_metrics_uses_only_common_acceleration_intervals():
    state_key = ("node-0", "target-0", 1.0)
    local_states = {
        state_key: SimpleNamespace(
            position_error_m=2.0, velocity_error_mps=0.2,
        ),
    }
    candidate_states = {
        state_key: SimpleNamespace(
            position_error_m=1.0, velocity_error_mps=0.1,
        ),
    }
    common = ("node-0", "target-0", 0.0, 1.0)
    local_only = ("node-0", "target-0", 1.0, 2.0)
    candidate_only = ("node-1", "target-0", 0.0, 1.0)

    metrics = _matched_view_metrics(
        local_states,
        candidate_states,
        {common: 0.20, local_only: 99.0},
        {common: 0.10, candidate_only: 99.0},
    )

    assert metrics.common_acceleration_sample_count == 1
    assert metrics.local_only_acceleration_sample_count == 1
    assert metrics.consensus_only_acceleration_sample_count == 1
    assert metrics.local_acceleration_rmse_mps2 == pytest.approx(0.20)
    assert metrics.consensus_acceleration_rmse_mps2 == pytest.approx(0.10)
    assert metrics.acceleration_rmse_improvement_percent == pytest.approx(50.0)


def test_baseline_arm_payload_is_json_serializable(baseline_matrix):
    result = baseline_matrix
    payload = {
        name: _arm_payload(arm) for name, arm in result.arms.items()
    }

    encoded = json.dumps(_jsonable(payload), allow_nan=False)

    assert "centralized_radar_only" in encoded
    assert "consensus_phase_metrics" in encoded


def test_single_acceptance_arm_matches_the_matrix_cann_arm(baseline_matrix):
    arm = run_bilateral_cann_acceptance_arm(BilateralFrontendConfig(
        duration_seconds=3.0,
        random_seed=13,
        backup_radial_range_sigma_m=1.0,
        backup_radial_range_rate_sigma_mps=0.0125,
        independent_radial_backup_channel=True,
    ))
    expected = baseline_matrix.arms["centralized_multimodal_cann"]

    assert arm.final_position_rmse_m == expected.final_position_rmse_m
    assert arm.final_velocity_rmse_mps == expected.final_velocity_rmse_mps
    assert arm.position_rmse_m_by_target == expected.position_rmse_m_by_target


@pytest.mark.parametrize(
    ("scope", "expected_arms", "expected_local_baselines"),
    (
        (
            "p05-control",
            {"centralized_optical_only"},
            {"single_node_optical_only"},
        ),
        (
            "p05-candidate",
            {
                "centralized_optical_only",
                "single_observer_joint_multimodal_cann",
            },
            {
                "single_node_optical_only",
                "single_node_multimodal_cann",
                "independent_15_node_joint_multimodal_cann",
            },
        ),
        (
            "p07-control",
            {"single_observer_joint_multimodal_cann"},
            {"independent_15_node_joint_multimodal_cann"},
        ),
        (
            "p05-optical-pilot",
            {
                "centralized_optical_only",
                "single_observer_joint_multimodal_cann",
            },
            {
                "single_node_optical_only",
                "independent_15_node_joint_multimodal_cann",
            },
        ),
        (
            "p05-fixed-weight-pilot",
            {
                "single_observer_fixed_weight_federated_no_cann",
                "single_observer_joint_multimodal_cann",
            },
            {
                "single_node_fixed_weight_federated_no_cann",
                "independent_15_node_joint_multimodal_cann",
            },
        ),
        (
            "p07-candidate",
            {
                "single_observer_joint_multimodal_cann",
                "fixed_topology_joint_multimodal_cann",
            },
            {"independent_15_node_joint_multimodal_cann"},
        ),
    ),
)
def test_acceptance_scope_runs_only_required_arms(
    scope, expected_arms, expected_local_baselines,
):
    result = run_bilateral_baseline_matrix(
        BilateralFrontendConfig(duration_seconds=3.0, random_seed=17),
        acceptance_scope=scope,
    )

    assert set(result.arms) == expected_arms
    assert set(result.tracking_modalities_by_arm) == expected_arms
    assert set(result.local_node_baselines) == expected_local_baselines
