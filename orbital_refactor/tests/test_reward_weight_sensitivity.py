import pytest

from experiments.reward_weight_sensitivity import scan_reward_weight_sensitivity


def test_difference_reward_removes_keep_resource_baseline():
    common = {
        "condition_seed": 1, "noise_seed": 0, "node_count": 5,
        "trajectory": "policy", "decision_index": 0,
        "resynchronization_penalty": 0.0, "topology_switch_penalty": 0.0,
    }
    records = (
        {**common, "action_id": 0, "action_kind": "keep",
         "counterfactual_task_gain": 0.0, "communication_penalty": 0.4},
        {**common, "action_id": 1, "action_kind": "remove",
         "counterfactual_task_gain": -0.1, "communication_penalty": 0.1},
    )
    result = scan_reward_weight_sensitivity(
        records, communication_multipliers=(1.0,),
        resynchronization_multipliers=(1.0,), switch_multipliers=(1.0,),
    )
    assert result["configuration_count"] == 2
    for configuration in result["configurations"]:
        assert configuration["best_action_kind_counts"] == {"remove": 1}
        assert configuration["mean_best_gain_over_keep"] == pytest.approx(0.2)
