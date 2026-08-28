from dataclasses import replace

import torch

from experiments.variable_scale_topology_curriculum import (
    VariableScaleTopologyCurriculum,
)
from experiments.variable_scale_topology_ppo import (
    VariableScalePPOConfiguration,
    train_variable_scale_topology_ppo,
)


def test_one_shared_update_contains_all_three_graph_sizes():
    configuration = VariableScalePPOConfiguration(
        curriculum=VariableScaleTopologyCurriculum(episode_epochs=2),
        training_episodes=3,
        rollout_batch_episodes=3,
        training_condition_seed_offset=400,
        training_condition_seed_count=3,
        environment_seed_count=1,
        update_epochs=1,
        minibatch_size=3,
        target_kl=None,
        return_scale_by_node_count=((5, 2.0), (10, 2.0), (20, 2.0)),
    )
    result = train_variable_scale_topology_ppo(configuration)
    assert [item.node_count for item in result.diagnostics] == [5, 10, 20]
    batch = result.batch_diagnostics[0]
    assert batch.episode_count_by_node_count == ((5, 1), (10, 1), (20, 1))
    assert batch.transition_count_by_node_count == ((5, 1), (10, 1), (20, 1))
    assert batch.update.transition_count == 3
    assert batch.update.gradient_norm > 0.0
    assert {item.node_count for item in batch.action_diagnostics} == {5, 10, 20}
    assert sum(item.transition_count for item in batch.action_diagnostics) == 3
    assert all(torch.isfinite(parameter).all() for parameter in result.model.parameters())
    assert result.model.actor.explicit_action_pairing
    for diagnostic in result.diagnostics:
        expected_penalty = (
            configuration.penalty_weights.communication
            * diagnostic.transmitted_messages_per_node_epoch
            + configuration.penalty_weights.resynchronization
            * diagnostic.resynchronizations_per_node
            + configuration.penalty_weights.topology_switch
            * diagnostic.topology_switches
        )
        assert abs(
            diagnostic.task_return
            - diagnostic.penalized_return * 2.0
            - expected_penalty
        ) < 1.0e-5
        assert abs(
            diagnostic.unnormalized_penalized_return
            - diagnostic.penalized_return * 2.0
        ) < 1.0e-5


def test_warm_and_random_initialization_use_identical_actor_structure():
    checkpoint = "results/v15_stratified_physical_gnn_hierarchical_seed00.pt"
    configuration = VariableScalePPOConfiguration(
        curriculum=VariableScaleTopologyCurriculum(episode_epochs=2),
        training_episodes=1,
        rollout_batch_episodes=1,
        training_condition_seed_offset=400,
        training_condition_seed_count=1,
        environment_seed_count=1,
        update_epochs=1,
        minibatch_size=1,
        target_kl=None,
        critic_timestamp_horizon=4.0,
        critic_scale_calibration_node_counts=(5, 10, 20),
        critic_weight_decay=1.0e-3,
    )
    random_result = train_variable_scale_topology_ppo(configuration)
    warm_result = train_variable_scale_topology_ppo(
        configuration, warm_start_checkpoint=checkpoint,
    )
    random_actor = random_result.model.actor.state_dict()
    warm_actor = warm_result.model.actor.state_dict()
    assert random_actor.keys() == warm_actor.keys()
    assert {
        name: tuple(value.shape) for name, value in random_actor.items()
    } == {
        name: tuple(value.shape) for name, value in warm_actor.items()
    }
    assert random_result.model.critic[0].in_features == (
        warm_result.model.critic[0].in_features
    )
    assert random_result.model.critic_timestamp_horizon == 4.0
    assert tuple(random_result.model.critic_scale_calibration) == ("5", "10", "20")


def test_incompatible_warm_start_pairing_is_rejected():
    configuration = VariableScalePPOConfiguration(
        curriculum=VariableScaleTopologyCurriculum(episode_epochs=2),
        training_episodes=1,
        rollout_batch_episodes=1,
        training_condition_seed_count=1,
        environment_seed_count=1,
        update_epochs=1,
        minibatch_size=1,
        target_kl=None,
        explicit_action_pairing=False,
    )
    try:
        train_variable_scale_topology_ppo(
            configuration,
            warm_start_checkpoint=(
                "results/v15_stratified_physical_gnn_hierarchical_seed00.pt"
            ),
        )
    except ValueError as error:
        assert "explicit-action-pairing" in str(error)
    else:
        raise AssertionError("Incompatible warm-start structure was accepted.")


def test_difference_resource_penalties_require_counterfactual_reward():
    configuration = VariableScalePPOConfiguration(
        curriculum=VariableScaleTopologyCurriculum(episode_epochs=2),
        training_episodes=1, rollout_batch_episodes=1,
        training_condition_seed_count=1, environment_seed_count=1,
        update_epochs=1, minibatch_size=1, target_kl=None,
        difference_resource_penalties_from_keep=True,
    )
    try:
        train_variable_scale_topology_ppo(configuration)
    except ValueError as error:
        assert "counterfactual keep reward" in str(error)
    else:
        raise AssertionError("Difference costs without keep branch were accepted.")


def test_training_checkpoint_resumes_at_next_batch(tmp_path):
    checkpoint = tmp_path / "training.pt"
    configuration = VariableScalePPOConfiguration(
        curriculum=VariableScaleTopologyCurriculum(episode_epochs=2),
        training_episodes=2,
        rollout_batch_episodes=1,
        training_condition_seed_offset=470,
        training_condition_seed_count=2,
        environment_seed_count=1,
        update_epochs=1,
        minibatch_size=1,
        target_kl=None,
    )
    uninterrupted = train_variable_scale_topology_ppo(configuration)
    partial = train_variable_scale_topology_ppo(
        configuration,
        training_checkpoint=checkpoint,
        stop_after_batches=1,
    )
    assert [item.episode for item in partial.diagnostics] == [0]

    resumed = train_variable_scale_topology_ppo(
        configuration,
        training_checkpoint=checkpoint,
        resume_training_checkpoint=checkpoint,
    )
    assert [item.episode for item in resumed.diagnostics] == [0, 1]
    assert [(item.batch_start, item.batch_end) for item in resumed.batch_diagnostics] == [
        (0, 1), (1, 2),
    ]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["next_episode"] == 2
    assert all(
        torch.equal(value, resumed.model.state_dict()[name])
        for name, value in saved["model_state_dict"].items()
    )
    assert resumed.diagnostics == uninterrupted.diagnostics
    assert resumed.batch_diagnostics == uninterrupted.batch_diagnostics
    assert all(
        torch.equal(value, uninterrupted.model.state_dict()[name])
        for name, value in resumed.model.state_dict().items()
    )


def test_training_checkpoint_rejects_configuration_change(tmp_path):
    checkpoint = tmp_path / "training.pt"
    configuration = VariableScalePPOConfiguration(
        curriculum=VariableScaleTopologyCurriculum(episode_epochs=2),
        training_episodes=1,
        rollout_batch_episodes=1,
        training_condition_seed_count=1,
        environment_seed_count=1,
        update_epochs=1,
        minibatch_size=1,
        target_kl=None,
    )
    train_variable_scale_topology_ppo(
        configuration, training_checkpoint=checkpoint,
    )
    try:
        train_variable_scale_topology_ppo(
            replace(configuration, learning_rate=2.0e-4),
            resume_training_checkpoint=checkpoint,
        )
    except ValueError as error:
        assert "configuration" in str(error)
    else:
        raise AssertionError("Changed training configuration was accepted.")
