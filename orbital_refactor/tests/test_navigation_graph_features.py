import numpy as np
import pytest

from brain_inspired.hierarchical_navigation_place_cells import (
    build_hierarchical_navigation_place_cell_histories,
)
from brain_inspired.navigation_graph_features import (
    NAVIGATION_GRAPH_NODE_METRIC_NAMES,
    build_navigation_graph_feature_histories,
    navigation_graph_metrics_at_timestamp,
)
from experiments.walker_navigation_brain_state_audit import (
    run_walker_navigation_brain_state_audit,
)


def test_navigation_graph_features_are_compact_nonnegative_and_aligned():
    navigation = run_walker_navigation_brain_state_audit(
        duration=4.0, dt=2.0, seed=2, anchor_interval_samples=1,
    ).histories
    place = build_hierarchical_navigation_place_cell_histories(
        navigation_by_node=navigation,
    )
    features = build_navigation_graph_feature_histories(
        navigation_by_node=navigation, place_by_node=place,
    )
    assert set(features) == set(navigation)
    for node, history in features.items():
        assert history.metric_names == NAVIGATION_GRAPH_NODE_METRIC_NAMES
        assert history.metric_matrix.shape == (
            navigation[node].timestamps.size,
            len(NAVIGATION_GRAPH_NODE_METRIC_NAMES),
        )
        assert np.all(np.isfinite(history.metric_matrix))
        assert np.all(history.metric_matrix >= 0.0)
        assert set(history.metrics_at(0)) == set(NAVIGATION_GRAPH_NODE_METRIC_NAMES)
    snapshot = navigation_graph_metrics_at_timestamp(
        features, timestamp=next(iter(features.values())).timestamps[1],
    )
    assert set(snapshot) == set(features)
    assert all(set(metrics) == set(NAVIGATION_GRAPH_NODE_METRIC_NAMES)
               for metrics in snapshot.values())


def test_navigation_graph_features_reject_node_mismatch():
    with pytest.raises(ValueError, match="identical nonempty"):
        build_navigation_graph_feature_histories(
            navigation_by_node={}, place_by_node={},
        )


def test_navigation_graph_snapshot_rejects_unaligned_timestamp():
    with pytest.raises(ValueError, match="no unique"):
        navigation_graph_metrics_at_timestamp({
            "a": type("History", (), {
                "timestamps": np.array([0.0]), "valid": np.array([True]),
                "metrics_at": lambda self, index: {},
            })(),
        }, timestamp=1.0)
