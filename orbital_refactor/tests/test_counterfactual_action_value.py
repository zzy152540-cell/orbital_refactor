import numpy as np
import pytest

from experiments.counterfactual_action_value import (
    build_counterfactual_action_value_dataset,
    cross_validate_seed_holdout_linear_action_value,
    cross_validate_seed_holdout_tree_action_value,
    cross_validate_linear_action_value_with_abstention,
    fit_seed_holdout_tree_action_value,
    evaluate_frozen_linear_action_value_policy,
    fit_frozen_linear_action_value_policy,
    fit_seed_holdout_linear_action_value,
)
from experiments.short_horizon_counterfactual_study import (
    run_short_horizon_counterfactual_study,
)


def _study():
    return run_short_horizon_counterfactual_study(
        seeds=(0, 1), decision_epochs=(1,), horizon_epochs=(1,)
    )


def test_action_value_dataset_keeps_causal_features_and_all_labels():
    dataset = build_counterfactual_action_value_dataset(_study())

    assert dataset.features.shape == (12, len(dataset.feature_names))
    assert np.all(np.isfinite(dataset.features))
    assert len(dataset.position_rmse_reduction) == 12
    assert set(dataset.seed_by_row) == {0, 1}
    assert len(set(dataset.group_by_row)) == 2
    assert len(dataset.observation_by_group) == 2
    assert all(
        observation.candidate_edges
        for _, observation in dataset.observation_by_group
    )
    assert not any("reduction" in name for name in dataset.feature_names)


def test_linear_action_value_uses_disjoint_seed_holdout():
    dataset = build_counterfactual_action_value_dataset(_study())
    result = fit_seed_holdout_linear_action_value(
        dataset, training_seeds=(0,), holdout_seeds=(1,), ridge=1.0,
    )

    assert result.holdout.group_count == 1
    assert 0.0 <= result.holdout.action_match_rate <= 1.0
    assert 0.0 <= result.holdout.action_kind_match_rate <= 1.0
    assert 0.0 <= result.holdout.keep_rate <= 1.0
    assert result.holdout.keep_rate + result.holdout.add_rate + (
        result.holdout.swap_rate + result.holdout.remove_rate
    ) == pytest.approx(1.0)
    assert len(result.model.coefficients) == len(dataset.feature_names)
    assert np.all(np.isfinite(result.model.predict(dataset.features)))


def test_linear_action_value_rejects_seed_leakage():
    dataset = build_counterfactual_action_value_dataset(_study())

    with pytest.raises(ValueError, match="disjoint"):
        fit_seed_holdout_linear_action_value(
            dataset, training_seeds=(0,), holdout_seeds=(0,)
        )


def test_linear_action_value_cross_validation_holds_out_every_seed():
    dataset = build_counterfactual_action_value_dataset(_study())
    summary = cross_validate_seed_holdout_linear_action_value(dataset)

    assert summary.fold_count == 2
    assert summary.group_count == 2
    assert 0.0 <= summary.keep_rate <= 1.0


def test_shallow_tree_action_value_is_deterministic_and_seed_disjoint():
    dataset = build_counterfactual_action_value_dataset(_study())
    options = dict(
        training_seeds=(0,), holdout_seeds=(1,),
        maximum_depth=2, minimum_leaf_size=2,
    )
    left = fit_seed_holdout_tree_action_value(dataset, **options)
    right = fit_seed_holdout_tree_action_value(dataset, **options)

    assert left == right
    assert left.holdout.group_count == 1
    assert np.all(np.isfinite(left.model.predict(dataset.features)))


def test_shallow_tree_cross_validation_covers_every_seed():
    dataset = build_counterfactual_action_value_dataset(_study())
    summary = cross_validate_seed_holdout_tree_action_value(
        dataset, maximum_depth=2, minimum_leaf_size=2,
    )

    assert summary.fold_count == 2
    assert summary.group_count == 2


def test_linear_abstention_threshold_is_calibrated_without_seed_leakage():
    dataset = build_counterfactual_action_value_dataset(_study())
    summary = cross_validate_linear_action_value_with_abstention(
        dataset, maximum_training_nees_violation_rate=0.5,
    )

    assert summary.fold_count == 2
    assert summary.group_count == 2
    assert 0.0 <= summary.keep_rate <= 1.0


def test_frozen_policy_evaluates_only_disjoint_validation_dataset():
    training = build_counterfactual_action_value_dataset(_study())
    policy = fit_frozen_linear_action_value_policy(
        training, training_seeds=(0,),
    )
    validation_study = run_short_horizon_counterfactual_study(
        seeds=(2,), decision_epochs=(1,), horizon_epochs=(1,)
    )
    validation = build_counterfactual_action_value_dataset(validation_study)
    result = evaluate_frozen_linear_action_value_policy(policy, validation)

    assert result.summary.holdout_seeds == (2,)
    assert result.summary.group_count == 1
    assert 0.0 <= result.positive_mean_gain_seed_rate <= 1.0

    with pytest.raises(ValueError, match="overlap"):
        evaluate_frozen_linear_action_value_policy(policy, training)
