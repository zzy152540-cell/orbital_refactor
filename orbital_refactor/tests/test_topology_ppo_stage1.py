import numpy as np
from pathlib import Path

from experiments.topology_ppo import collect_topology_rollout
from experiments.topology_ppo_stage1 import (
    Stage1Configuration,
    Stage1SeedSplit,
    apply_stage1_penalties,
    build_stage1_environment,
    compare_stage1_training_seeds,
    evaluate_stage1_policies,
    five_node_heterogeneous_link_configuration,
    five_node_randomized_physical_configuration,
    five_node_stage1_configuration,
    five_node_robust_ppo_configuration,
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


def test_stage1_environment_configuration_selects_fleet_and_candidate_bound():
    configuration = Stage1Configuration(
        node_count=5, top_k_candidate_neighbors=1, training_episodes=1,
    )
    environment = build_stage1_environment(configuration)
    state = environment.reset(seed=0)
    assert environment.node_count == 5
    assert environment.top_k_candidate_neighbors == 1
    assert len(state.observation.nodes) == 5


def test_five_node_baseline_freezes_distribution_and_disjoint_seed_split():
    configuration = five_node_stage1_configuration(training_episodes=1)
    assert configuration.node_count == 5
    assert configuration.top_k_candidate_neighbors == 2
    assert configuration.scenario_distribution.initial_topology_types == (
        "chain", "ring", "star",
    )
    split = Stage1SeedSplit()
    split.validate()
    assert not set(split.training) & set(split.validation)
    assert not set(split.validation) & set(split.test)


def test_five_node_heterogeneous_link_configuration_is_explicit_opt_in():
    baseline = five_node_stage1_configuration()
    expanded = five_node_heterogeneous_link_configuration()

    assert baseline.scenario_distribution.link_condition_mode == "homogeneous"
    assert (
        expanded.scenario_distribution.link_condition_mode
        == "undirected_independent"
    )
    assert expanded.node_count == baseline.node_count == 5
    assert expanded.top_k_candidate_neighbors == baseline.top_k_candidate_neighbors


def test_five_node_randomized_physical_configuration_combines_geometry_and_links():
    configuration = five_node_randomized_physical_configuration()
    distribution = configuration.scenario_distribution

    assert configuration.node_count == 5
    assert distribution.link_condition_mode == "undirected_independent"
    assert distribution.physical_scenario_families == (
        "compact_along_track",
        "differential_along_track",
        "two_plane_cluster",
    )


def test_five_node_robust_ppo_baseline_uses_low_variance_updates():
    configuration = five_node_robust_ppo_configuration()
    assert configuration.node_count == 5
    assert configuration.maximum_topology_switches_per_episode == 1
    assert configuration.condition_seed_offset == 40
    assert configuration.condition_seed_count == 4
    assert configuration.environment_seed_count == 8
    assert configuration.rollout_batch_episodes == 32
    assert configuration.learning_rate == 1.0e-4
    assert configuration.target_kl == 0.02


def test_stage1_configuration_forwards_target_kl_to_ppo_update():
    configuration = Stage1Configuration(
        training_episodes=2, episode_epochs=4, decision_interval_epochs=2,
        rollout_batch_episodes=2, minibatch_size=2, update_epochs=4,
        learning_rate=1.0e-2, target_kl=1.0e-12,
    )
    result = train_stage1_ppo(configuration)
    assert result.diagnostics[-1].update.stopped_early


def test_stage1_warm_start_can_preserve_same_task_type_head():
    configuration = Stage1Configuration(
        training_episodes=1, episode_epochs=2, update_epochs=1,
    )
    checkpoint = Path(
        "results/v15_walker_gnn_hierarchical_seed00_05_val06_07.pt"
    )
    result = train_stage1_ppo(
        configuration, warm_start_checkpoint=str(checkpoint),
        reset_warm_start_type_head=False,
    )
    assert len(result.diagnostics) == 1


def test_stage1_training_cycles_randomized_seeds_and_records_two_level_entropy():
    configuration = Stage1Configuration(
        training_episodes=3, episode_epochs=4,
        decision_interval_epochs=2, environment_seed_count=2,
        update_epochs=1, policy_seed=3,
    )
    result = train_stage1_ppo(configuration)
    assert tuple(item.environment_seed for item in result.diagnostics) == (0, 1, 0)
    assert tuple(item.condition_seed for item in result.diagnostics) == (0, 1, 0)
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


def test_stage1_training_cycles_cartesian_condition_and_noise_seeds():
    configuration = Stage1Configuration(
        training_episodes=5, episode_epochs=2, decision_interval_epochs=1,
        environment_seed_count=2, condition_seed_offset=40,
        condition_seed_count=2, rollout_batch_episodes=2, update_epochs=1,
    )
    result = train_stage1_ppo(configuration)
    assert tuple(item.environment_seed for item in result.diagnostics) == (
        0, 1, 0, 1, 0,
    )
    assert tuple(item.condition_seed for item in result.diagnostics) == (
        40, 40, 41, 41, 40,
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
