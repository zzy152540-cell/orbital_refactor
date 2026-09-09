from experiments.online_cann_update_mode_ablation import (
    run_online_cann_update_mode_ablation,
)


def test_decision_epoch_cann_mode_preserves_filter_and_reduces_calls():
    result = run_online_cann_update_mode_ablation(
        seed=2, node_count=3, episode_epochs=6,
        decision_interval_epochs=2, dt=2.0,
    )
    assert result.base_filter_identical
    assert result.base_tensor_prefix_identical
    assert result.every_epoch_runtime_seconds > 0.0
    assert result.decision_epoch_runtime_seconds > 0.0
    assert result.every_epoch_reset_seconds > 0.0
    assert result.every_epoch_step_seconds > 0.0
    assert result.mean_cann_feature_difference > 0.0


def test_invalid_cann_update_mode_is_rejected():
    from experiments.topology_control_environment import TopologyControlEnvironment

    try:
        TopologyControlEnvironment(cann_update_mode="invalid")
    except ValueError as error:
        assert "update mode" in str(error)
    else:
        raise AssertionError("Invalid CANN update mode was accepted.")
