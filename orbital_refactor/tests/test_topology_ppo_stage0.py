import numpy as np

from experiments.topology_ppo_stage0 import (
    Stage0Configuration,
    build_stage0_environment,
    compare_stage0_initializations,
    train_stage0_ppo,
)


def test_stage0_scenario_has_a_unique_better_than_keep_initial_action():
    result = train_stage0_ppo(Stage0Configuration(training_episodes=1))
    assert result.oracle_initial_action_id == 2
    assert result.oracle_initial_action_id != 0


def test_stage0_training_loop_records_finite_online_diagnostics():
    configuration = Stage0Configuration(
        training_episodes=2, episode_epochs=3, policy_seed=4,
        update_epochs=1,
    )
    left = train_stage0_ppo(configuration)
    right = train_stage0_ppo(configuration)
    assert left.initial_policy_action_id == right.initial_policy_action_id
    assert left.final_policy_action_id == right.final_policy_action_id
    assert len(left.diagnostics) == 2
    assert all(
        np.isfinite(diagnostic.task_return)
        and np.isfinite(diagnostic.penalized_return)
        and np.isfinite(diagnostic.update.final_loss)
        and diagnostic.update.epochs_run == 1
        for diagnostic in left.diagnostics
    )
    assert all(diagnostic.update.epochs_run == 1 for diagnostic in left.diagnostics)
    assert all(
        diagnostic.initial_action_id >= 0
        and diagnostic.transmitted_messages > 0.0
        for diagnostic in left.diagnostics
    )


def test_stage0_environment_definition_remains_range_only_and_three_node():
    environment = build_stage0_environment(episode_epochs=2)
    state = environment.reset(seed=0)
    assert len(state.observation.nodes) == 3
    assert {
        modality
        for edge in state.observation.candidate_edges
        for modality in edge.measurement_modalities
    } == {"RANGE"}


def test_stage0_comparison_uses_matched_seeds_and_budgets():
    result = compare_stage0_initializations(
        policy_seeds=(2,), training_episodes=2, success_window=1,
        warm_start_checkpoint=(
            "results/v15_walker_gnn_hierarchical_seed00_05_val06_07.pt"
        ),
    )
    assert tuple(record.initialization for record in result.records) == (
        "random", "warm_start",
    )
    assert {record.policy_seed for record in result.records} == {2}
    assert len(result.records_for("random")) == 1
    assert all(record.oracle_action_id == 2 for record in result.records)
