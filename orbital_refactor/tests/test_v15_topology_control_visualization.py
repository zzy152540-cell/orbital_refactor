import numpy as np

from experiments.topology_control_baselines import AlwaysKeepPolicy
from experiments.topology_control_environment import TopologyControlEnvironment
from experiments.v15_topology_control_visualization import (
    collect_topology_control_trace,
    load_topology_control_trace_bundle,
    save_topology_control_trace_bundle,
)


def test_topology_control_trace_aligns_filter_and_resource_timelines():
    trace = collect_topology_control_trace(
        TopologyControlEnvironment(
            node_count=3, episode_epochs=4, decision_interval_epochs=1,
            relative_modalities=("RANGE",),
        ),
        AlwaysKeepPolicy(), condition_seed=0, noise_seed=0,
    )

    assert len(trace.times) == 5
    assert len(trace.action_kinds) == 4
    assert trace.position_rmse.shape == trace.times.shape
    assert trace.position_three_sigma.shape == trace.times.shape
    assert trace.active_edge_count.shape == trace.times.shape
    assert np.all(np.diff(trace.cumulative_transmitted) >= 0.0)
    assert np.all(np.diff(trace.cumulative_dropped) >= 0.0)
    assert trace.initial_edges == trace.final_edges


def test_topology_control_trace_bundle_round_trips_without_pickle(tmp_path):
    trace = collect_topology_control_trace(
        TopologyControlEnvironment(
            node_count=3, episode_epochs=2, relative_modalities=("RANGE",),
        ), AlwaysKeepPolicy(), condition_seed=2, noise_seed=1,
    )
    path = save_topology_control_trace_bundle(
        tmp_path / "trace.npz", trace, trace,
        condition_seed=2, noise_seed=1,
    )
    keep, reference, metadata = load_topology_control_trace_bundle(path)

    assert metadata["schema_version"] == "v15.0-topology-control-visualization"
    assert metadata["condition_seed"] == 2
    np.testing.assert_allclose(keep.position_rmse, trace.position_rmse)
    np.testing.assert_allclose(reference.position_three_sigma,
                               trace.position_three_sigma)
    assert keep.truth_by_node.keys() == trace.truth_by_node.keys()
