import pytest

from experiments.edge_metric_correlation_study import (
    EdgeMetricRecord,
    edge_score_selection_summaries,
    metric_correlations,
    run_edge_metric_correlation_study,
)


def _record(index, outcome):
    return EdgeMetricRecord(
        node_count=3,
        seed=index,
        edge=("a", "b"),
        covariance_trace_reduction=float(index),
        covariance_logdet_reduction=float(4 - index),
        position_rmse_reduction=float(outcome),
        worst_node_position_rmse_reduction=float(outcome),
        nees_calibration_improvement=float(outcome),
        nees_coverage_calibration_improvement=float(outcome),
        transmitted_message_cost=1,
        replay_cost=0,
        resynchronization_cost=0,
    )


def test_metric_correlations_report_linear_and_rank_direction():
    correlations = metric_correlations(tuple(
        _record(index, 2 * index) for index in range(1, 4)
    ))
    by_pair = {
        (value.predictor, value.outcome): value for value in correlations
    }

    trace = by_pair[(
        "covariance_trace_reduction", "position_rmse_reduction"
    )]
    logdet = by_pair[(
        "covariance_logdet_reduction", "position_rmse_reduction"
    )]
    assert trace.pearson == pytest.approx(1.0)
    assert trace.spearman == pytest.approx(1.0)
    assert logdet.pearson == pytest.approx(-1.0)
    assert logdet.spearman == pytest.approx(-1.0)


def test_three_and_five_node_correlation_study_collects_all_edge_samples():
    study = run_edge_metric_correlation_study(
        node_counts=(3, 5), seeds=(0,), duration=4.0, dt=2.0,
        relative_modalities=("RANGE",),
    )

    assert len(study.records) == 3 + 10
    assert len(study.pooled_correlations) == 8
    assert set(study.correlations_by_node_count) == {3, 5}
    assert len(study.selection_summaries) == 2
    assert set(study.selection_summaries_by_node_count) == {3, 5}
    assert all(value.sample_count == 3
               for value in study.correlations_by_node_count[3])
    assert all(value.sample_count == 10
               for value in study.correlations_by_node_count[5])


def test_correlation_study_rejects_duplicate_seeds():
    with pytest.raises(ValueError, match="unique"):
        run_edge_metric_correlation_study(seeds=(0, 0))


def test_selection_summary_reports_best_edge_hits_and_regret():
    records = (
        _record(0, 1.0),
        EdgeMetricRecord(
            **{
                **_record(0, 2.0).__dict__,
                "edge": ("a", "c"),
                "covariance_trace_reduction": 3.0,
                "covariance_logdet_reduction": 1.0,
            }
        ),
    )

    summaries = {
        value.predictor: value
        for value in edge_score_selection_summaries(records)
    }
    trace = summaries["covariance_trace_reduction"]
    logdet = summaries["covariance_logdet_reduction"]
    assert trace.best_edge_hit_rate == 1.0
    assert trace.mean_position_rmse_regret == 0.0
    assert logdet.best_edge_hit_rate == 0.0
    assert logdet.mean_position_rmse_regret == 1.0
