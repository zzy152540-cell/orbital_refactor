import numpy as np
import pytest

from cooperative.topology_policy import (
    GraphEdgeFeature,
    GraphNodeFeature,
    GraphObservation,
    TopologyAction,
)
from experiments.decision_time_edge_scoring import (
    assess_edge_nis_safety,
    range_edge_scores,
    range_topology_score,
)


def test_range_edge_score_matches_symmetric_diagonal_approximation():
    observation = GraphObservation(
        0.0,
        (
            GraphNodeFeature("a", (0.0, 0.0, 0.0), (2.0, 2.0, 2.0)),
            GraphNodeFeature("b", (3.0, 4.0, 0.0), (2.0, 2.0, 2.0)),
        ),
        (GraphEdgeFeature(("a", "b"), 5.0),),
    )

    score = range_edge_scores(observation, range_sigma=2.0)[0]

    assert score.endpoint_position_uncertainty == 12.0
    assert score.projected_position_uncertainty == pytest.approx(4.0)
    assert score.approximate_trace_reduction == pytest.approx(1.0)
    assert score.approximate_logdet_reduction == pytest.approx(
        2.0 * np.log1p(2.0 / 6.0)
    )
    assert score.negative_distance == -5.0
    assert score.observation_age == 0.0
    assert score.recent_mean_nis == 0.0
    assert score.negative_recent_mean_nis == 0.0
    assert score.nis_calibration_quality == 0.0
    assert score.nis_sample_count == 0.0


def test_range_topology_score_adds_only_selected_candidate_edges():
    observation = GraphObservation(
        0.0,
        tuple(
            GraphNodeFeature(
                node, state, (1.0, 1.0, 1.0)
            )
            for node, state in (
                ("a", (0.0, 0.0, 0.0)),
                ("b", (1.0, 0.0, 0.0)),
                ("c", (2.0, 0.0, 0.0)),
            )
        ),
        (
            GraphEdgeFeature(("a", "b"), 1.0),
            GraphEdgeFeature(("b", "c"), 1.0),
        ),
    )
    one = range_topology_score(
        observation, TopologyAction("one", (("a", "b"),))
    )
    two = range_topology_score(
        observation,
        TopologyAction("two", (("a", "b"), ("b", "c"))),
    )

    assert two.approximate_trace_reduction == pytest.approx(
        2.0 * one.approximate_trace_reduction
    )
    with pytest.raises(ValueError, match="non-candidate"):
        range_topology_score(
            observation, TopologyAction("bad", (("a", "c"),))
        )


def test_range_edge_score_rejects_missing_covariance():
    observation = GraphObservation(
        0.0,
        (
            GraphNodeFeature("a", (0.0, 0.0, 0.0)),
            GraphNodeFeature("b", (1.0, 0.0, 0.0)),
        ),
        (GraphEdgeFeature(("a", "b"), 1.0),),
    )

    with pytest.raises(ValueError, match="covariance"):
        range_edge_scores(observation)


def test_range_edge_score_exports_recent_innovation_and_freshness():
    observation = GraphObservation(
        0.0,
        (
            GraphNodeFeature("a", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            GraphNodeFeature("b", (1.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ),
        (
            GraphEdgeFeature(
                ("a", "b"), 1.0,
                nis_by_modality=(("RANGE", 2.5),),
                nis_sample_count_by_modality=(("RANGE", 3),),
                consecutive_anomaly_count_by_modality=(("RANGE", 1),),
                observation_age=3.0,
            ),
        ),
    )

    score = range_edge_scores(observation)[0]

    assert score.observation_age == 3.0
    assert score.recent_mean_nis == 2.5
    assert score.negative_recent_mean_nis == -2.5
    assert score.nis_calibration_quality == pytest.approx(-abs(np.log(2.5)))
    assert score.nis_sample_count == 3.0
    assert score.negative_consecutive_anomaly_count == -1.0


def test_nis_safety_gate_requires_history_calibration_and_no_anomaly():
    safe = GraphEdgeFeature(
        ("a", "b"), 1.0,
        nis_by_modality=(("RADAR", 2.0),),
        nis_sample_count_by_modality=(("RADAR", 3),),
        consecutive_anomaly_count_by_modality=(("RADAR", 0),),
    )
    unknown = GraphEdgeFeature(("a", "b"), 1.0)

    assert assess_edge_nis_safety(safe).passes_safety_gate
    assert not assess_edge_nis_safety(unknown).passes_safety_gate
