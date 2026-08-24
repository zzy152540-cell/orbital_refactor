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
