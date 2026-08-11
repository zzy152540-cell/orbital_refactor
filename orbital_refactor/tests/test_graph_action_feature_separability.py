import pytest

from experiments.counterfactual_action_value import (
    build_counterfactual_action_value_dataset,
)
from experiments.graph_action_feature_separability import (
    binary_auc,
    analyze_action_feature_separability,
)
from experiments.short_horizon_counterfactual_study import (
    run_short_horizon_counterfactual_study,
)


def test_binary_auc_handles_direction_ties_and_missing_class():
    assert binary_auc([0.0, 1.0, 2.0, 3.0], [False, False, True, True]) == 1.0
    assert binary_auc([3.0, 2.0, 1.0, 0.0], [False, False, True, True]) == 0.0
    assert binary_auc([1.0, 1.0], [False, True]) is None
    assert binary_auc([0.0, 1.0], [True, True]) is None


def test_feature_separability_uses_only_requested_causal_action_rows():
    study = run_short_horizon_counterfactual_study(
        seeds=(0, 1, 2), decision_epochs=(1, 3), horizon_epochs=(1,),
        relative_modalities=("RANGE", "RANGE_RATE", "AZ_EL"),
    )
    dataset = build_counterfactual_action_value_dataset(study)
    report = analyze_action_feature_separability(
        dataset, action_kinds=("remove", "swap"), decision_epochs=(3,),
    )

    expected = sum(
        record.action_kind in {"remove", "swap"}
        and record.decision_epoch == 3
        for record in dataset.records
    )
    assert report.sample_count == expected
    assert len(report.features) == len(dataset.feature_names)
    assert 0.0 <= report.safe_positive_rate <= 1.0
    assert all(
        value.direction_free_auc is None
        or 0.5 <= value.direction_free_auc <= 1.0
        for value in report.features
    )
    with pytest.raises(ValueError, match="No action rows"):
        analyze_action_feature_separability(
            dataset, action_kinds=("unknown",)
        )
