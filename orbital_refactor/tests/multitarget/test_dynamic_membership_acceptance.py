from experiments.multitarget_dynamic_membership_acceptance import (
    run_multitarget_dynamic_membership_acceptance,
)


def test_joining_node_inherits_identities_after_origin_nodes_leave():
    result = run_multitarget_dynamic_membership_acceptance()

    assert result.joined_node_ids == ("observer-c",)
    assert result.removed_node_ids == ("observer-a", "observer-b")
    assert result.initial_cooperative_target_count == 2
    assert result.final_cooperative_target_count == 2
    assert result.identity_switch_count == 0
    assert result.false_cooperative_birth_count == 0
    assert result.departed_contributor_count == 0
    assert result.final_target_count == 2
    assert result.brain_target_count == 2
    assert result.brain_valid_fraction == 1.0
    assert result.brain_direction_neuron_count == 180
    assert result.brain_place_heatmap_shape == (3, 3)
    assert result.brain_place_normalization_error < 1.0e-12
    assert result.passed
