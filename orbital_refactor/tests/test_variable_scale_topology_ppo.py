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
    )
    result = train_variable_scale_topology_ppo(configuration)
    assert [item.node_count for item in result.diagnostics] == [5, 10, 20]
    batch = result.batch_diagnostics[0]
    assert batch.episode_count_by_node_count == ((5, 1), (10, 1), (20, 1))
    assert batch.transition_count_by_node_count == ((5, 1), (10, 1), (20, 1))
    assert batch.update.transition_count == 3
    assert batch.update.gradient_norm > 0.0
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
            - diagnostic.penalized_return
            - expected_penalty
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
