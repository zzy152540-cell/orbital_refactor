import numpy as np

from experiments.topology_ppo import collect_topology_rollout
from experiments.topology_ppo_stage1 import (
    Stage1Configuration,
    apply_stage1_penalties,
    build_stage1_environment,
    compare_stage1_training_seeds,
    evaluate_stage1_policies,
    scan_stage1_penalty_sensitivity,
    train_stage1_ppo,
)


def test_stage1_penalties_are_normalized_and_do_not_double_count_replay():
    configuration = Stage1Configuration(training_episodes=1)
    environment = build_stage1_environment(configuration)
    trained = train_stage1_ppo(configuration)
    rollout = collect_topology_rollout(
        environment, trained.model, seed=0, deterministic=True,
    )
    penalized = apply_stage1_penalties(rollout, configuration)
    for original, adjusted in zip(rollout.transitions, penalized.transitions):
        expected = (
            original.reward
            - 0.0025 * original.costs[0] / 4.0
            - 0.001 * original.costs[4]
            - 0.001 * original.costs[3] / 2.0
        )
        np.testing.assert_allclose(adjusted.reward, expected)
        assert adjusted.costs == original.costs


def test_stage1_training_cycles_randomized_seeds_and_records_two_level_entropy():
    configuration = Stage1Configuration(
        training_episodes=3, episode_epochs=4,
        decision_interval_epochs=2, environment_seed_count=2,
        update_epochs=1, policy_seed=3,
    )
    result = train_stage1_ppo(configuration)
    assert tuple(item.environment_seed for item in result.diagnostics) == (0, 1, 0)
    assert all(
        np.isfinite(item.task_return)
        and np.isfinite(item.penalized_return)
        and np.isfinite(item.final_position_rmse)
        and item.type_entropy >= 0.0
        and item.conditional_entropy >= 0.0
        and len(item.initial_type_probabilities) == 4
        and np.isclose(sum(item.initial_type_probabilities), 1.0)
        and item.update.actor_transition_count
        <= item.update.transition_count
        and item.update.epochs_run == 1
        for item in result.diagnostics
    )
    assert all(
        item.penalized_return <= item.task_return + 1e-7
        for item in result.diagnostics
    )


def test_stage1_evaluation_is_paired_deterministic_and_rejects_training_seeds():
    configuration = Stage1Configuration(
        training_episodes=1, episode_epochs=4, decision_interval_epochs=2,
        environment_seed_count=2, update_epochs=1,
    )
    random_result = train_stage1_ppo(configuration)
    warm_result = train_stage1_ppo(
        configuration,
        warm_start_checkpoint=(
            "results/v15_walker_gnn_hierarchical_seed00_05_val06_07.pt"
        ),
    )
    arguments = dict(
        configuration=configuration, test_seeds=(10, 11),
        random_model=random_result.model, warm_start_model=warm_result.model,
    )
    left = evaluate_stage1_policies(**arguments)
    right = evaluate_stage1_policies(**arguments)
    assert left == right
    assert len(left.records) == 8
    assert {record.environment_seed for record in left.records} == {10, 11}
    assert {record.policy_name for record in left.records} == {
        "always_keep", "cost_aware_information_greedy",
        "ppo_random_init", "ppo_warm_start",
    }
    assert all(record.fallback_count == 0.0 for record in left.records)
    try:
        evaluate_stage1_policies(**{**arguments, "test_seeds": (1,)})
    except ValueError:
        pass
    else:
        raise AssertionError("Training seed leaked into Stage 1 evaluation.")


def test_penalty_sensitivity_rescores_fixed_records_without_mutation():
    configuration = Stage1Configuration(
        training_episodes=1, episode_epochs=4, decision_interval_epochs=2,
        environment_seed_count=1, update_epochs=1,
    )
    random_result = train_stage1_ppo(configuration)
    warm_result = train_stage1_ppo(configuration)
    evaluation = evaluate_stage1_policies(
        configuration, test_seeds=(10,), random_model=random_result.model,
        warm_start_model=warm_result.model,
    )
    sensitivity = scan_stage1_penalty_sensitivity(
        evaluation, configuration, scales=(0.0, 1.0),
    )
    assert len(sensitivity) == 2
    zero = dict(sensitivity[0].mean_penalized_return_by_policy)
    assert all(
        np.isclose(zero[record.policy_name], record.task_return)
        for record in evaluation.records
    )
    assert evaluation == evaluate_stage1_policies(
        configuration, test_seeds=(10,), random_model=random_result.model,
        warm_start_model=warm_result.model,
    )


def test_training_seed_comparison_pairs_initializations():
    records = compare_stage1_training_seeds(
        Stage1Configuration(
            training_episodes=1, episode_epochs=4,
            decision_interval_epochs=2, environment_seed_count=1,
            rollout_batch_episodes=1, minibatch_size=2, update_epochs=1,
        ),
        policy_seeds=(3,), test_seeds=(10,),
        warm_start_checkpoint=(
            "results/v15_walker_gnn_hierarchical_seed00_05_val06_07.pt"
        ),
    )
    assert tuple(record.initialization for record in records) == (
        "random", "warm_start",
    )
    assert {record.policy_seed for record in records} == {3}
    assert all(np.isfinite(record.mean_final_position_rmse) for record in records)
