from dataclasses import replace

import numpy as np

from cooperative.topology_action_space import build_topology_action_space
from experiments.topology_control_baselines import (
    AlwaysKeepPolicy,
    InformationGreedyPolicy,
    LowChurnObservablePolicy,
    RandomLegalPolicy,
    ShortHorizonOraclePolicy,
    run_topology_control_baseline_episode,
)
from experiments.topology_control_environment import TopologyControlEnvironment


def _environment():
    return TopologyControlEnvironment(
        node_count=3, episode_epochs=3, relative_modalities=("RANGE",),
    )


def test_keep_and_low_churn_baselines_complete_without_switches():
    keep = run_topology_control_baseline_episode(
        _environment(), AlwaysKeepPolicy(), seed=0,
    )
    low_churn = run_topology_control_baseline_episode(
        _environment(), LowChurnObservablePolicy(), seed=0,
    )
    assert keep.step_count == 3
    assert keep.selected_action_kind_counts == (("keep", 3),)
    assert keep.cumulative_costs.topology_switch == 0.0
    assert low_churn == type(low_churn)(
        **{**low_churn.__dict__, "policy_name": "low_churn_observable"}
    )
    assert low_churn.final_position_rmse == keep.final_position_rmse


def test_random_legal_baseline_is_seed_reproducible_and_never_falls_back():
    left = run_topology_control_baseline_episode(
        _environment(), RandomLegalPolicy(seed=9), seed=2,
    )
    right = run_topology_control_baseline_episode(
        _environment(), RandomLegalPolicy(seed=9), seed=2,
    )
    assert left == right
    assert left.fallback_reason_counts == ()
    assert left.cumulative_costs.action_fallback == 0.0
    assert sum(count for _, count in left.selected_action_kind_counts) == 3


def test_information_greedy_uses_only_current_observation_and_is_reproducible():
    left = run_topology_control_baseline_episode(
        _environment(), InformationGreedyPolicy(), seed=2,
    )
    right = run_topology_control_baseline_episode(
        _environment(), InformationGreedyPolicy(), seed=2,
    )
    assert left == right
    assert left.fallback_reason_counts == ()


def test_default_information_greedy_avoids_repeated_topology_churn():
    summary = run_topology_control_baseline_episode(
        TopologyControlEnvironment(
            node_count=5, episode_epochs=6, relative_modalities=("RANGE",),
            packet_loss=0.1, communication_delay=1.0,
        ),
        InformationGreedyPolicy(), seed=0,
    )
    assert summary.cumulative_costs.topology_switch == 1.0
    assert summary.cumulative_costs.resynchronization_count == 2.0
    assert summary.selected_action_kind_counts == (("add", 1), ("keep", 5))


def test_information_greedy_removes_redundant_invisible_edge():
    environment = TopologyControlEnvironment(
        node_count=3, episode_epochs=3, relative_modalities=("RANGE",),
    )
    state = environment.reset(seed=0)
    add_id = next(action.action_id for action in state.action_space.actions
                  if action.kind == "add")
    added_edge = state.action_space.actions[add_id].added_edges[0]
    state = environment.step(add_id).state
    invisible = tuple(
        replace(edge, geometrically_visible=False)
        if edge.nodes == added_edge
        else edge
        for edge in state.observation.candidate_edges
    )
    state = replace(
        state, observation=replace(state.observation, candidate_edges=invisible)
    )
    state = replace(
        state, action_space=build_topology_action_space(state.observation)
    )
    selected = InformationGreedyPolicy().select_action(state)
    assert state.action_space.actions[selected].kind == "remove"


def test_oracle_lookahead_does_not_mutate_live_environment_during_selection():
    environment = _environment()
    state = environment.reset(seed=3)
    before = state.policy_tensor.node_features.copy()
    oracle = ShortHorizonOraclePolicy(environment)
    action_id = oracle.select_action(state)
    after = environment._state()
    assert action_id in {
        action.action_id for action, allowed in zip(
            state.action_space.actions, state.action_space.legal_mask
        ) if allowed
    }
    np.testing.assert_allclose(after.policy_tensor.node_features, before)
    result = environment.step(action_id)
    assert not result.action_resolution.used_fallback


def test_oracle_supports_bounded_keep_rollout_and_validates_horizon():
    environment = _environment()
    state = environment.reset(seed=4)
    action_id = ShortHorizonOraclePolicy(
        environment, lookahead_steps=2,
    ).select_action(state)
    assert state.action_space.legal_mask[action_id]
    try:
        ShortHorizonOraclePolicy(environment, lookahead_steps=0)
    except ValueError:
        pass
    else:
        raise AssertionError("A zero-step Oracle horizon must be rejected.")
