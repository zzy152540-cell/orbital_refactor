from dataclasses import replace

import pytest

from experiments.graph_action_label_stability import (
    analyze_graph_action_label_stability,
)
from experiments.short_horizon_counterfactual_study import (
    run_short_horizon_counterfactual_study,
)


def test_label_stability_computes_exact_cross_seed_agreement():
    study = run_short_horizon_counterfactual_study(
        seeds=(0,), decision_epochs=(1,), horizon_epochs=(1,),
    )
    base = next(
        record for record in study.records if record.action_kind == "swap"
    )
    records = tuple(
        replace(
            base,
            seed=seed,
            position_rmse_reduction=0.1 if seed < 3 else -0.1,
            nees_calibration_improvement=0.1,
            nees_coverage_calibration_improvement=0.1,
        )
        for seed in range(4)
    )
    report = analyze_graph_action_label_stability(
        replace(study, seeds=(0, 1, 2, 3), records=records)
    )

    assert report.overall.cell_count == 1
    assert report.cells[0].safe_positive_rate == 0.75
    assert report.cells[0].pairwise_label_agreement == 0.5
    assert report.overall.ambiguous_cell_rate == 1.0


def test_label_stability_aligns_real_action_replicas_and_rejects_singletons():
    study = run_short_horizon_counterfactual_study(
        seeds=(0, 1, 2), decision_epochs=(1,), horizon_epochs=(1,),
        relative_modalities=("RANGE", "RANGE_RATE", "AZ_EL"),
    )
    report = analyze_graph_action_label_stability(study)

    assert report.seed_count == 3
    assert report.overall.minimum_replica_count == 3
    assert {value.action_kind for value in report.by_action_kind} == {
        "add", "remove", "swap"
    }
    with pytest.raises(ValueError, match="replicated"):
        analyze_graph_action_label_stability(
            replace(study, seeds=(0,), records=study.records[:6])
        )
