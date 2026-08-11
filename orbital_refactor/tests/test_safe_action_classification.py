import pytest

from experiments.counterfactual_action_value import (
    build_counterfactual_action_value_dataset,
)
from experiments.safe_action_classification import (
    fit_safe_action_classifier,
)
from experiments.short_horizon_counterfactual_study import (
    run_short_horizon_counterfactual_study,
)


def test_safe_action_classifiers_keep_train_validation_test_seeds_disjoint():
    study = run_short_horizon_counterfactual_study(
        seeds=range(6), decision_epochs=(1, 3), horizon_epochs=(1,),
        relative_modalities=("RANGE", "RANGE_RATE", "AZ_EL"),
    )
    dataset = build_counterfactual_action_value_dataset(study)

    for model_kind in ("linear", "tree"):
        result = fit_safe_action_classifier(
            dataset,
            action_kind="remove",
            model_kind=model_kind,
            training_seeds=(0, 1, 2, 3),
            validation_seeds=(4,),
            test_seeds=(5,),
            minimum_leaf_size=2,
        )
        assert result.validation.sample_count > 0
        assert result.test.sample_count > 0
        assert 0.0 <= result.test.balanced_accuracy <= 1.0

    with pytest.raises(ValueError, match="disjoint"):
        fit_safe_action_classifier(
            dataset,
            action_kind="swap",
            model_kind="linear",
            training_seeds=(0, 1),
            validation_seeds=(1, 2),
            test_seeds=(3,),
        )
