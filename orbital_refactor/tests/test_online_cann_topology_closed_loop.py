from experiments.online_cann_topology_closed_loop import (
    run_online_cann_topology_closed_loop,
)


def test_paired_online_cann_loop_preserves_filter_and_base_policy_path():
    result = run_online_cann_topology_closed_loop(
        seed=4, node_count=3, episode_epochs=3, dt=2.0,
    )
    assert result.epoch_count == 4
    assert result.base_filter_identical
    assert result.base_tensor_prefix_identical
    assert result.rewards_identical
    assert result.costs_identical
    assert result.all_actions_legal
    assert result.cann_features_changed
    assert result.disabled_runtime_seconds > 0.0
    assert result.enabled_runtime_seconds > 0.0
