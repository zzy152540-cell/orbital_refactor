import numpy as np
import torch
from pathlib import Path

from experiments.topology_ppo import (
    TopologyActorCritic,
    advantage_gated_policy_action_index,
    build_warm_started_actor_critic,
    clipped_ppo_loss,
    collect_topology_rollout,
    combine_prepared_topology_rollouts,
    conservative_policy_action_index,
    generalized_advantage_estimate,
    hierarchical_action_distribution,
    prepare_topology_rollout,
    update_topology_ppo,
)
from experiments.topology_control_environment import TopologyControlEnvironment
from experiments.topology_snapshot_counterfactual import (
    build_online_snapshot_action_tensor,
)
from experiments.graph_action_gnn import torch_snapshot_action_group


def _actor_critic_and_state(
    *, episode_epochs=3, treat_horizon_as_truncation=False,
):
    environment = TopologyControlEnvironment(
        node_count=3, episode_epochs=episode_epochs,
        relative_modalities=("RANGE",),
        treat_horizon_as_truncation=treat_horizon_as_truncation,
    )
    state = environment.reset(seed=0)
    snapshot, _ = build_online_snapshot_action_tensor(state)
    group = torch_snapshot_action_group(snapshot)
    model = TopologyActorCritic(
        node_feature_count=group.node_features.shape[1],
        candidate_edge_feature_count=group.candidate_edge_features.shape[1],
        measurement_feature_count=group.measurement_features.shape[1],
        action_feature_count=group.action_features.shape[1],
        global_feature_count=len(state.policy_tensor.global_feature_names),
        hidden_size=16, explicit_action_pairing=True,
    )
    return environment, state, group, model


def test_hierarchical_distribution_masks_actions_and_normalizes_two_levels():
    distribution = hierarchical_action_distribution(
        type_logits=torch.tensor([0.0, 1.0, 20.0, 30.0]),
        conditional_action_logits=torch.tensor([0.0, 1.0, 2.0, 99.0]),
        action_kind_index=torch.tensor([0, 1, 1, 3]),
        legal_mask=torch.tensor([True, True, True, False]),
    )
    probabilities = distribution.action_log_probabilities.exp()
    assert probabilities[3] == 0.0
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))
    expected_types = torch.softmax(torch.tensor([0.0, 1.0]), dim=0)
    torch.testing.assert_close(distribution.type_probabilities[:2], expected_types)
    torch.testing.assert_close(probabilities[1:].sum(), expected_types[1])
    assert distribution.mode().item() == 2
    assert distribution.type_entropy > 0.0
    assert distribution.conditional_entropy > 0.0


def test_hierarchical_distribution_single_keep_is_deterministic():
    distribution = hierarchical_action_distribution(
        torch.zeros(4), torch.tensor([4.0, 9.0]), torch.tensor([0, 1]),
        torch.tensor([True, False]),
    )
    assert distribution.mode().item() == 0
    assert distribution.sample(generator=torch.Generator().manual_seed(3)).item() == 0
    assert distribution.log_prob(0).item() == 0.0
    assert distribution.entropy.item() == 0.0


def test_hierarchical_mode_selects_type_before_member_without_count_bias():
    distribution = hierarchical_action_distribution(
        type_logits=torch.tensor([0.0, 1.0, -10.0, -10.0]),
        conditional_action_logits=torch.zeros(5),
        action_kind_index=torch.tensor([0, 1, 1, 1, 1]),
        legal_mask=torch.ones(5, dtype=torch.bool),
    )
    probabilities = distribution.action_log_probabilities.exp()

    assert probabilities[0] > probabilities[1]
    assert distribution.type_probabilities[1] > distribution.type_probabilities[0]
    assert distribution.mode().item() in {1, 2, 3, 4}


def test_gae_stops_at_true_terminal_and_returns_value_targets():
    advantages, returns = generalized_advantage_estimate(
        rewards=torch.tensor([1.0, 2.0, 3.0]),
        values=torch.tensor([0.5, 0.5, 0.5]),
        next_values=torch.tensor([0.5, 0.5, 10.0]),
        terminated=torch.tensor([False, True, False]),
        gamma=1.0, gae_lambda=1.0,
    )
    torch.testing.assert_close(advantages, torch.tensor([2.5, 1.5, 12.5]))
    torch.testing.assert_close(returns, torch.tensor([3.0, 2.0, 13.0]))


def test_clipped_ppo_loss_uses_clipped_surrogate_and_reports_diagnostics():
    old = torch.zeros(2)
    new = torch.log(torch.tensor([1.5, 0.5])).requires_grad_()
    result = clipped_ppo_loss(
        new, old, advantages=torch.tensor([1.0, -1.0]),
        predicted_values=torch.tensor([0.0, 2.0]),
        returns=torch.tensor([1.0, 1.0]), entropies=torch.tensor([0.5, 0.5]),
    )
    np.testing.assert_allclose(result.policy.item(), -0.2, atol=1e-6)
    np.testing.assert_allclose(result.value.item(), 1.0, atol=1e-6)
    assert result.clip_fraction.item() == 1.0
    assert result.approximate_kl.item() > 0.0
    result.total.backward()
    assert torch.isfinite(new.grad).all()


def test_ppo_helpers_reject_misaligned_inputs():
    try:
        hierarchical_action_distribution(
            torch.zeros(4), torch.zeros(2), torch.tensor([0]),
            torch.tensor([True, True]),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched hierarchical action inputs were accepted.")

    try:
        generalized_advantage_estimate(
            torch.zeros(2), torch.zeros(1), torch.zeros(2), torch.zeros(2),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched GAE inputs were accepted.")


def test_actor_critic_produces_normalized_policy_value_and_gradients():
    _, _, group, model = _actor_critic_and_state()
    output = model(group)
    torch.testing.assert_close(
        output.distribution.action_log_probabilities.exp().sum(),
        torch.tensor(1.0),
    )
    assert output.value.shape == ()
    loss = -output.distribution.log_prob(0) + output.value.square()
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_critic_episode_phase_does_not_change_actor_structure():
    _, state, group, baseline = _actor_critic_and_state()
    phased = TopologyActorCritic(
        node_feature_count=group.node_features.shape[1],
        candidate_edge_feature_count=group.candidate_edge_features.shape[1],
        measurement_feature_count=group.measurement_features.shape[1],
        action_feature_count=group.action_features.shape[1],
        global_feature_count=len(state.policy_tensor.global_feature_names),
        hidden_size=16,
        explicit_action_pairing=True,
        critic_timestamp_horizon=6.0,
    )
    assert baseline.actor.state_dict().keys() == phased.actor.state_dict().keys()
    assert {
        name: tuple(value.shape)
        for name, value in baseline.actor.state_dict().items()
    } == {
        name: tuple(value.shape)
        for name, value in phased.actor.state_dict().items()
    }
    assert phased.critic[0].in_features == baseline.critic[0].in_features
    assert torch.count_nonzero(phased.critic_phase_projection.weight) == 0
    phased.actor.load_state_dict(baseline.actor.state_dict())
    phased.critic.load_state_dict(baseline.critic.state_dict())
    torch.testing.assert_close(phased(group).value, baseline(group).value)


def test_rollout_uses_legal_environment_actions_and_records_costs():
    environment, _, _, model = _actor_critic_and_state(episode_epochs=3)
    rollout = collect_topology_rollout(
        environment, model, seed=2,
        generator=torch.Generator().manual_seed(11),
    )
    assert len(rollout.transitions) == 3
    assert rollout.transitions[-1].terminated
    assert not any(transition.truncated for transition in rollout.transitions)
    assert rollout.cost_matrix.shape == (3, 6)
    assert rollout.rewards.shape == (3,)
    assert all(
        transition.environment_action_id >= 0
        and np.isfinite(transition.old_log_probability)
        and transition.type_entropy >= 0.0
        and transition.conditional_entropy >= 0.0
        for transition in rollout.transitions
    )
    assert rollout.final_value == 0.0


def test_rollout_bootstraps_value_at_truncated_training_window():
    environment, _, _, model = _actor_critic_and_state(
        episode_epochs=1, treat_horizon_as_truncation=True,
    )
    with torch.no_grad():
        for parameter in model.critic.parameters():
            parameter.zero_()
        model.critic[-1].bias.fill_(1.0)
    rollout = collect_topology_rollout(
        environment, model, seed=2,
        generator=torch.Generator().manual_seed(11),
    )
    assert not rollout.transitions[-1].terminated
    assert rollout.transitions[-1].truncated
    assert rollout.final_value == 1.0


def test_prepare_and_update_ppo_on_variable_action_graphs():
    environment, _, _, model = _actor_critic_and_state(episode_epochs=4)
    rollout = collect_topology_rollout(
        environment, model, seed=3,
        generator=torch.Generator().manual_seed(17),
    )
    prepared = prepare_topology_rollout(rollout)
    assert len(prepared.transitions) == 4
    assert prepared.advantages.shape == (4,)
    assert prepared.returns.shape == (4,)
    np.testing.assert_allclose(prepared.advantages.mean().item(), 0.0, atol=1e-5)
    before = tuple(parameter.detach().clone() for parameter in model.parameters())
    result = update_topology_ppo(
        model, torch.optim.Adam(model.parameters(), lr=1e-4), prepared,
        update_epochs=2, target_kl=None,
    )
    assert result.epochs_run == 2
    assert np.isfinite(result.final_loss)
    assert np.isfinite(result.approximate_kl)
    assert result.gradient_norm >= 0.0
    assert result.transition_count == 4
    assert 0 < result.actor_transition_count <= result.transition_count
    assert 0.0 <= result.positive_advantage_fraction <= 1.0
    assert np.isfinite(result.explained_variance_before_update)
    assert not result.stopped_early
    assert any(
        not torch.equal(previous, current.detach())
        for previous, current in zip(before, model.parameters())
    )


def test_ppo_supports_shuffled_minibatch_updates():
    environment, _, _, model = _actor_critic_and_state(episode_epochs=4)
    prepared = prepare_topology_rollout(
        collect_topology_rollout(environment, model, seed=6)
    )
    result = update_topology_ppo(
        model, torch.optim.Adam(model.parameters(), lr=1e-4), prepared,
        update_epochs=2, minibatch_size=2, target_kl=None,
        generator=torch.Generator().manual_seed(8),
    )
    assert result.epochs_run == 2
    assert np.isfinite(result.final_loss)


def test_ppo_kl_guard_can_stop_before_an_unsafe_update():
    environment, _, _, model = _actor_critic_and_state(episode_epochs=3)
    rollout = collect_topology_rollout(environment, model, seed=1)
    prepared = prepare_topology_rollout(rollout)
    shifted = type(prepared)(
        transitions=prepared.transitions,
        old_log_probabilities=prepared.old_log_probabilities - 2.0,
        advantages=prepared.advantages, returns=prepared.returns,
        actor_mask=prepared.actor_mask,
    )
    before = tuple(parameter.detach().clone() for parameter in model.parameters())
    result = update_topology_ppo(
        model, torch.optim.Adam(model.parameters(), lr=1e-3), shifted,
        update_epochs=4, target_kl=0.01,
    )
    assert result.epochs_run == 1
    assert result.stopped_early
    assert all(
        torch.equal(previous, current.detach())
        for previous, current in zip(before, model.parameters())
    )


def test_existing_hierarchical_checkpoint_warm_starts_actor_exactly():
    checkpoint_path = Path(
        "results/v15_walker_gnn_hierarchical_seed00_05_val06_07.pt"
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    model = build_warm_started_actor_critic(checkpoint_path)
    loaded = model.actor.state_dict()
    assert loaded.keys() == checkpoint["model_state_dict"].keys()
    assert all(
        torch.equal(loaded[name], expected)
        for name, expected in checkpoint["model_state_dict"].items()
    )


def test_old_checkpoint_expands_new_node_features_without_behavior_drift():
    checkpoint_path = Path(
        "results/v15_walker_gnn_hierarchical_seed00_05_val06_07.pt"
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    model = build_warm_started_actor_critic(
        checkpoint_path,
        node_feature_count=int(checkpoint["node_feature_count"]) + 2,
    )
    weights = model.actor.node_encoder[0].weight.detach()
    expected = checkpoint["model_state_dict"]["node_encoder.0.weight"]
    torch.testing.assert_close(weights[:, :13], expected[:, :13])
    torch.testing.assert_close(weights[:, 14:22], expected[:, 13:21])
    torch.testing.assert_close(weights[:, 23:31], expected[:, 21:29])
    assert torch.count_nonzero(weights[:, (13, 22)]) == 0


def test_combined_rollouts_preserve_episode_boundaries_and_actor_mask():
    environment, _, _, model = _actor_critic_and_state(episode_epochs=4)
    first = prepare_topology_rollout(
        collect_topology_rollout(environment, model, seed=1),
        normalize_advantages=False,
    )
    second = prepare_topology_rollout(
        collect_topology_rollout(environment, model, seed=2),
        normalize_advantages=False,
    )
    combined = combine_prepared_topology_rollouts((first, second))
    assert len(combined.transitions) == len(first.transitions) + len(second.transitions)
    torch.testing.assert_close(combined.returns[:len(first.returns)], first.returns)
    torch.testing.assert_close(combined.returns[len(first.returns):], second.returns)
    assert combined.actor_mask.dtype == torch.bool


def test_cooldown_keep_steps_are_critic_only():
    environment = TopologyControlEnvironment(
        node_count=3, episode_epochs=4, relative_modalities=("RANGE",),
        minimum_topology_dwell_decisions=2,
    )
    _, state, group, model = _actor_critic_and_state(episode_epochs=4)
    rollout = collect_topology_rollout(
        environment, model, seed=0,
        generator=torch.Generator().manual_seed(5),
    )
    prepared = prepare_topology_rollout(rollout)
    assert prepared.actor_mask.shape == prepared.returns.shape
    assert torch.any(~prepared.actor_mask)
    assert all(
        len(transition.group.action_features) == 1
        for transition, actor_enabled in zip(
            prepared.transitions, prepared.actor_mask
        ) if not actor_enabled
    )


def test_critic_updates_when_a_minibatch_has_no_actor_choice():
    environment = TopologyControlEnvironment(
        node_count=3, episode_epochs=4, relative_modalities=("RANGE",),
        minimum_topology_dwell_decisions=10,
    )
    _, _, _, model = _actor_critic_and_state(episode_epochs=4)
    prepared = prepare_topology_rollout(
        collect_topology_rollout(
            environment, model, seed=0,
            generator=torch.Generator().manual_seed(5),
        )
    )
    assert torch.any(~prepared.actor_mask)
    prepared = type(prepared)(
        transitions=prepared.transitions,
        old_log_probabilities=prepared.old_log_probabilities,
        advantages=prepared.advantages, returns=prepared.returns,
        actor_mask=torch.zeros_like(prepared.actor_mask),
    )
    # Keep one actor sample solely to satisfy the overall-batch guard; the
    # shuffled size-one minibatches still exercise critic-only updates.
    prepared.actor_mask[0] = True
    before = tuple(parameter.detach().clone() for parameter in model.critic.parameters())
    update_topology_ppo(
        model, torch.optim.Adam(model.parameters(), lr=1e-3), prepared,
        update_epochs=1, minibatch_size=1, target_kl=None,
        generator=torch.Generator().manual_seed(0),
    )
    assert any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, model.critic.parameters())
    )


def test_conservative_policy_gate_requires_margin_to_override_reference():
    kinds = torch.tensor([0, 0], dtype=torch.long)
    legal = torch.ones(2, dtype=torch.bool)
    type_logits = torch.zeros(4)
    policy = hierarchical_action_distribution(
        type_logits, torch.tensor([1.0, 0.98]), kinds, legal,
    )
    reference = hierarchical_action_distribution(
        type_logits, torch.tensor([0.0, 1.0]), kinds, legal,
    )
    assert conservative_policy_action_index(policy, reference, 0.05).item() == 1
    assert conservative_policy_action_index(policy, reference, 0.01).item() == 0


def test_advantage_gate_requires_predicted_gain_to_override_reference():
    kinds = torch.tensor([0, 0], dtype=torch.long)
    legal = torch.ones(2, dtype=torch.bool)
    type_logits = torch.zeros(4)
    policy = hierarchical_action_distribution(
        type_logits, torch.tensor([1.0, 0.0]), kinds, legal,
    )
    reference = hierarchical_action_distribution(
        type_logits, torch.tensor([0.0, 1.0]), kinds, legal,
    )
    assert advantage_gated_policy_action_index(
        policy, reference, torch.tensor([0.2, 0.1]), 0.05,
    ).item() == 0
    assert advantage_gated_policy_action_index(
        policy, reference, torch.tensor([0.12, 0.1]), 0.05,
    ).item() == 1
