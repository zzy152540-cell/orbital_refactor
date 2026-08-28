from experiments.variable_scale_policy_diagnostics import (
    _action_summary,
    _advantage_summary,
    _baseline_summary,
)


def test_policy_diagnostic_summaries_preserve_scale_and_action_kind():
    baselines = [{
        "node_count": 5,
        "keep_final_position_rmse": 2.0,
        "keep_final_worst_node_position_rmse": 3.0,
    }]
    state_records = [{
        "node_count": 5,
        "trajectory": "policy",
        "outcomes": [{
            "action_kind": "remove",
            "action_probability": 0.2,
            "counterfactual_task_gain": 0.4,
            "communication_penalty": 0.1,
            "resynchronization_penalty": 0.0,
            "topology_switch_penalty": 0.1,
            "training_objective": 0.2,
            "objective_gain_over_keep": 0.3,
        }],
    }]
    advantages = [{
        "node_count": 5,
        "action_kind": "remove",
        "advantage": 0.3,
        "reward": 0.2,
    }]

    assert _baseline_summary(baselines)["5"]["mean_keep_final_position_rmse"] == 2.0
    remove = _action_summary(state_records)["5"]["remove"]
    assert remove["positive_training_objective_fraction"] == 1.0
    assert remove["mean_communication_penalty"] == 0.1
    assert remove["mean_objective_gain_over_keep"] == 0.3
    assert _advantage_summary(advantages)["5"]["remove"]["mean_advantage"] == 0.3
