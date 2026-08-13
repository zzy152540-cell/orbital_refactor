import numpy as np

from experiments.topology_control_environment import TopologyControlEnvironment
from scenarios.measurement_visibility import VisibilityConfig


def test_environment_reset_and_multistep_keep_are_reproducible():
    options = dict(node_count=3, episode_epochs=3, relative_modalities=("RANGE",))
    left = TopologyControlEnvironment(**options)
    right = TopologyControlEnvironment(**options)
    left_state = left.reset(seed=4)
    right_state = right.reset(seed=4)
    np.testing.assert_allclose(
        left_state.policy_tensor.node_features,
        right_state.policy_tensor.node_features,
    )
    assert left_state.action_space.actions[0].kind == "keep"
    left_step = left.step(0)
    right_step = right.step(0)
    np.testing.assert_allclose(
        left_step.state.policy_tensor.node_features,
        right_step.state.policy_tensor.node_features,
    )
    np.testing.assert_allclose(
        left_step.state.policy_tensor.edge_features,
        right_step.state.policy_tensor.edge_features,
    )
    assert left_step.reward == right_step.reward
    assert left_step.reward_terms == right_step.reward_terms
    assert left_step.constraint_costs == right_step.constraint_costs
    assert left_step.action_resolution == right_step.action_resolution
    assert left_step.terminated == right_step.terminated
    assert not left_step.action_resolution.used_fallback
    assert left_step.constraint_costs.topology_switch == 0.0


def test_environment_applies_add_and_invalid_action_falls_back():
    environment = TopologyControlEnvironment(
        node_count=3, episode_epochs=3, relative_modalities=("RANGE",)
    )
    state = environment.reset(seed=0)
    add_id = next(action.action_id for action in state.action_space.actions
                  if action.kind == "add")
    added = environment.step(add_id)
    assert added.constraint_costs.topology_switch == 1.0
    assert not added.action_resolution.used_fallback
    fallback = environment.step(999)
    assert fallback.action_resolution.used_fallback
    assert fallback.action_resolution.reason == "action_id_out_of_range"
    assert fallback.constraint_costs.action_fallback == 1.0


def test_environment_requires_reset_and_terminates_at_horizon():
    environment = TopologyControlEnvironment(
        node_count=3, episode_epochs=1, relative_modalities=("RANGE",)
    )
    try:
        environment.step(0)
    except RuntimeError as error:
        assert "reset" in str(error)
    else:
        raise AssertionError("Environment stepped before reset.")
    environment.reset(seed=0)
    result = environment.step(0)
    assert result.terminated
    assert not result.truncated


def test_environment_masks_new_edges_without_current_visible_measurements():
    environment = TopologyControlEnvironment(
        node_count=3, episode_epochs=2, relative_modalities=("RANGE",),
        visibility_by_modality={
            "RANGE": VisibilityConfig(maximum_range=1.0),
        },
    )
    state = environment.reset(seed=0)
    assert state.action_space.legal_mask.tolist() == [True, False, False, False]
    assert all(
        reason == "not_geometrically_visible"
        for reason in state.action_space.rejection_reason_by_action[1:]
    )
    assert all(
        not edge.geometrically_visible and not edge.measurement_modalities
        for edge in state.observation.candidate_edges
    )


def test_environment_visibility_mask_changes_from_current_measurements():
    environment = TopologyControlEnvironment(
        node_count=5, episode_epochs=30, relative_modalities=("RANGE",),
        visibility_by_modality={
            "RANGE": VisibilityConfig(maximum_range=1206.0),
        },
    )
    state = environment.reset(seed=0)
    visible_counts = []
    while True:
        visible_counts.append(sum(
            edge.geometrically_visible
            for edge in state.observation.candidate_edges
        ))
        result = environment.step(0)
        state = result.state
        if result.terminated:
            visible_counts.append(sum(
                edge.geometrically_visible
                for edge in state.observation.candidate_edges
            ))
            break
    assert max(visible_counts) > min(visible_counts)
    assert set(visible_counts) >= {2, 3, 4}


def test_environment_enforces_dwell_then_restores_topology_actions():
    environment = TopologyControlEnvironment(
        node_count=3, episode_epochs=4, relative_modalities=("RANGE",),
        minimum_topology_dwell_decisions=2,
    )
    state = environment.reset(seed=0)
    add_id = next(action.action_id for action in state.action_space.actions
                  if action.kind == "add")
    state = environment.step(add_id).state
    assert dict(state.observation.graph_metrics)[
        "topology_cooldown_remaining"
    ] == 2.0
    assert all(
        action.kind == "keep" or not allowed
        for action, allowed in zip(
            state.action_space.actions, state.action_space.legal_mask
        )
    )
    state = environment.step(0).state
    assert dict(state.observation.graph_metrics)[
        "topology_cooldown_remaining"
    ] == 1.0
    state = environment.step(0).state
    assert dict(state.observation.graph_metrics)[
        "topology_cooldown_remaining"
    ] == 0.0
    assert any(
        action.kind != "keep" and allowed
        for action, allowed in zip(
            state.action_space.actions, state.action_space.legal_mask
        )
    )


def test_walker_twenty_environment_uses_sparse_physical_candidates():
    environment = TopologyControlEnvironment(
        node_count=20, episode_epochs=2,
        relative_modalities=("RADAR", "INFRARED", "OPTICAL"),
        scenario_type="walker_20_5_3", walker_maximum_range=7000e3,
        top_k_candidate_neighbors=2, minimum_topology_dwell_decisions=2,
    )
    state = environment.reset(seed=0)
    assert len(state.observation.nodes) == 20
    assert len(state.observation.previous_active_edges) == 19
    assert len(state.observation.candidate_edges) == 28
    assert len(state.action_space.actions) == 53
    assert state.action_space.legal_mask.all()
    result = environment.step(0)
    assert not result.terminated
    assert result.constraint_costs.transmitted_messages > 0.0


def test_stage1_conditions_are_seeded_and_expose_navigation_availability():
    environment = TopologyControlEnvironment(
        node_count=3, episode_epochs=6, relative_modalities=("RANGE",),
        randomize_stage1_conditions=True,
    )
    first = environment.reset(seed=7)
    first_conditions = environment._episode_conditions
    repeated = environment.reset(seed=7)
    assert environment._episode_conditions == first_conditions
    assert 0.0 <= first_conditions["packet_loss"] <= 0.2
    assert 0.0 <= first_conditions["communication_delay"] <= 2.0
    navigation_feature = repeated.policy_tensor.node_feature_names.index(
        "log1p_absolute_navigation_available"
    )
    availability_mask = repeated.policy_tensor.node_feature_names.index(
        "available_absolute_navigation_available"
    )
    assert (repeated.policy_tensor.node_features[:, availability_mask] == 1.0).all()
    observed = [repeated.policy_tensor.node_features[:, navigation_feature].copy()]
    state = repeated
    while True:
        step = environment.step(0)
        state = step.state
        observed.append(state.policy_tensor.node_features[:, navigation_feature].copy())
        if step.terminated:
            break
    assert any((values == 0.0).any() for values in observed[1:])
