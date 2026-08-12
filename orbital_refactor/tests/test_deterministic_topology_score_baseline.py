import pytest

from experiments.deterministic_topology_score_baseline import (
    cross_validate_deterministic_score_with_abstention,
)
from experiments.short_horizon_counterfactual_study import (
    run_short_horizon_counterfactual_study,
)


def test_deterministic_score_abstention_is_seed_disjoint_and_conservative():
    study = run_short_horizon_counterfactual_study(
        seeds=(0, 1, 2), decision_epochs=(2,), horizon_epochs=(2,),
        relative_modalities=("RANGE", "RANGE_RATE", "AZ_EL"),
        backend="online_orchestrator",
        future_batch_relative_observations=True,
    )
    result = cross_validate_deterministic_score_with_abstention(
        study, feature="endpoint_uncertainty_score_gain",
    )

    assert result.fold_count == 3
    assert result.group_count == 3
    assert len(result.selections) == result.group_count
    assert result.worst_nees_calibration_improvement <= (
        result.mean_nees_calibration_improvement
    )
    assert result.position_rmse_reduction_confidence_interval[0] <= (
        result.mean_position_rmse_reduction
    ) <= result.position_rmse_reduction_confidence_interval[1]
    assert result.nees_calibration_violation_confidence_interval[0] <= (
        result.nees_calibration_violation_rate
    ) <= result.nees_calibration_violation_confidence_interval[1]
    assert 0.0 <= result.keep_rate <= 1.0
    assert result.keep_rate + result.add_rate + result.swap_rate + (
        result.remove_rate
    ) == pytest.approx(1.0)


def test_deterministic_score_keeps_decision_and_horizon_groups_separate():
    study = run_short_horizon_counterfactual_study(
        seeds=(0, 1), decision_epochs=(1, 2), horizon_epochs=(1, 2)
    )
    result = cross_validate_deterministic_score_with_abstention(
        study, feature="endpoint_uncertainty_score_gain",
    )

    assert result.fold_count == 2
    assert result.group_count == 8


def test_deterministic_score_rejects_unknown_feature():
    study = run_short_horizon_counterfactual_study(
        seeds=(0, 1), decision_epochs=(1,), horizon_epochs=(1,)
    )
    with pytest.raises(ValueError, match="ShortHorizonActionRecord"):
        cross_validate_deterministic_score_with_abstention(
            study, feature="future_truth"
        )


def test_deterministic_score_rejects_negative_risk_weight():
    study = run_short_horizon_counterfactual_study(
        seeds=(0, 1), decision_epochs=(1,), horizon_epochs=(1,)
    )
    with pytest.raises(ValueError, match="cannot be negative"):
        cross_validate_deterministic_score_with_abstention(
            study, feature="endpoint_uncertainty_score_gain",
            communication_risk_weight=-1.0,
        )

    with pytest.raises(ValueError, match="tolerance cannot be negative"):
        cross_validate_deterministic_score_with_abstention(
            study, feature="endpoint_uncertainty_score_gain",
            nees_calibration_degradation_tolerance=-0.1,
        )


def test_added_link_risk_gate_can_force_keep():
    study = run_short_horizon_counterfactual_study(
        seeds=(0, 1), decision_epochs=(1,), horizon_epochs=(1,),
        packet_loss_by_edge={("sat_01", "sat_03"): 0.5},
    )
    result = cross_validate_deterministic_score_with_abstention(
        study, feature="endpoint_uncertainty_score_gain",
        maximum_added_communication_risk=0.0,
    )

    assert result.add_rate == 0.0
    assert result.remove_rate == 0.0


def test_five_node_heterogeneous_risk_score_is_finite():
    study = run_short_horizon_counterfactual_study(
        node_counts=(5,), seeds=(0, 1),
        decision_epochs=(1,), horizon_epochs=(1,),
        backend="online_orchestrator",
        packet_loss_by_edge={("sat_01", "sat_05"): 0.4},
        communication_delay_by_edge={("sat_01", "sat_05"): 2.0},
    )
    result = cross_validate_deterministic_score_with_abstention(
        study, feature="endpoint_uncertainty_score_gain",
        communication_risk_weight=1.0,
        maximum_added_communication_risk=1.0,
    )

    assert result.group_count == 2
    assert result.remove_rate == 0.0
    assert result.mean_position_rmse_reduction == pytest.approx(
        result.mean_position_rmse_reduction
    )
