import torch

from experiments.graph_action_gnn import torch_snapshot_action_group
from experiments.topology_ppo import TopologyActorCritic
from experiments.topology_ppo_stage1 import build_stage1_environment
from experiments.topology_snapshot_counterfactual import (
    build_online_snapshot_action_tensor,
)
from experiments.variable_scale_topology_curriculum import (
    VariableScaleTopologyCurriculum,
)


def _group(state):
    return torch_snapshot_action_group(
        build_online_snapshot_action_tensor(state)[0]
    )


def test_variable_scale_cycle_has_frozen_target_proportions():
    curriculum = VariableScaleTopologyCurriculum()
    counts = {
        node_count: curriculum.node_count_cycle.count(node_count)
        for node_count in (5, 10, 20)
    }
    assert counts == {5: 8, 10: 7, 20: 5}


def test_shared_actor_accepts_each_curriculum_graph_size():
    curriculum = VariableScaleTopologyCurriculum(episode_epochs=2)
    states = []
    for condition_seed in (0, 1, 2):
        configuration = curriculum.configuration_for_condition(condition_seed)
        assert configuration.treat_horizon_as_truncation
        environment = build_stage1_environment(configuration)
        state = environment.reset(seed=0, condition_seed=condition_seed)
        assert len(state.observation.nodes) == configuration.node_count
        assert state.action_space.legal_mask.any()
        states.append(state)

    first = _group(states[0])
    model = TopologyActorCritic(
        node_feature_count=first.node_features.shape[1],
        candidate_edge_feature_count=first.candidate_edge_features.shape[1],
        measurement_feature_count=first.measurement_features.shape[1],
        action_feature_count=first.action_features.shape[1],
        global_feature_count=states[0].policy_tensor.global_features.shape[0],
        hidden_size=16,
        message_passing_steps=1,
        explicit_action_pairing=False,
    )
    for state in states:
        group = _group(state)
        with torch.no_grad():
            output = model(group)
        assert output.distribution.action_log_probabilities.shape[0] == len(
            state.action_space.actions
        )
        assert torch.isfinite(output.value)


def test_curriculum_link_events_change_and_restore_channel_conditions():
    curriculum = VariableScaleTopologyCurriculum(episode_epochs=12)
    configuration = curriculum.configuration_for_condition(0)
    environment = build_stage1_environment(configuration)
    environment.reset(seed=0, condition_seed=0)
    events = environment._episode_conditions["dynamic_link_events_by_link"]
    assert len(events) == 4
    edge, event = next(iter(events.items()))
    receiver, source = edge
    channel = environment._orchestrator.channels[edge]
    base_loss = environment._episode_conditions[
        "packet_loss_rate_by_link"
    ][edge]
    base_delay = environment._episode_conditions[
        "communication_delay_by_link"
    ][edge]

    environment._apply_dynamic_link_conditions(event[0])
    assert channel.packet_loss_rate[source] >= event[2]
    assert channel.delay_by_source[source] >= event[3]
    environment._apply_dynamic_link_conditions(event[1] + environment.dt)
    assert channel.packet_loss_rate[source] == base_loss
    assert channel.delay_by_source[source] == base_delay
