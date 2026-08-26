import json

from experiments.variable_scale_ppo_multiseed import (
    summarize_variable_scale_ppo_seeds,
)


def _summary(path, seed, improvement):
    scales = {}
    for node_count in (5, 10, 20):
        scales[str(node_count)] = {
            "episode_count": 1,
            "improved_episode_count": int(improvement > 0.0),
            "mean_rmse_improvement": improvement,
            "worst_rmse_improvement": improvement,
            "action_kind_counts": {"keep": 1},
        }
    branch = {
        "evaluation_by_node_count": scales,
        "batch_diagnostics": [{
            "update": {"explained_variance_before_update": seed / 10.0},
        }],
    }
    path.write_text(json.dumps({
        "configuration": {"policy_seed": seed, "training_episodes": 2},
        "evaluation_conditions": [10, 11, 12],
        "random_init": branch,
        "warm_start": branch,
    }), encoding="utf-8")


def test_multiseed_summary_aggregates_frozen_runs(tmp_path):
    first, second = tmp_path / "seed0.json", tmp_path / "seed1.json"
    _summary(first, 0, 0.1)
    _summary(second, 1, 0.3)
    result = summarize_variable_scale_ppo_seeds((first, second))
    warm = result["aggregate"]["warm_start"]
    assert [item["policy_seed"] for item in result["runs"]] == [0, 1]
    assert warm["episode_count"] == 6
    assert warm["improved_episode_count"] == 6
    assert abs(warm["mean_rmse_improvement"] - 0.2) < 1.0e-12
    assert warm["action_kind_counts"] == {
        "keep": 6, "add": 0, "swap": 0, "remove": 0,
    }


def test_multiseed_summary_rejects_nonseed_configuration_change(tmp_path):
    first, second = tmp_path / "seed0.json", tmp_path / "seed1.json"
    _summary(first, 0, 0.1)
    _summary(second, 1, 0.1)
    payload = json.loads(second.read_text(encoding="utf-8"))
    payload["configuration"]["training_episodes"] = 3
    second.write_text(json.dumps(payload), encoding="utf-8")
    try:
        summarize_variable_scale_ppo_seeds((first, second))
    except ValueError as error:
        assert "policy seed" in str(error)
    else:
        raise AssertionError("A changed frozen configuration was accepted.")
