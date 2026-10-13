import pytest

from cooperative.topology import NetworkTopology, chain_topology
from scenarios import (
    TEMPORARY_FRIENDLY_STATE_ASSUMPTION,
    BilateralMultiTargetScenario,
    HostileTargetRegistry,
    build_bilateral_acceptance_scenario,
    generate_bilateral_walker_truth,
)


def test_default_v2_scenario_separates_15_friendly_and_15_hostile_satellites():
    scenario = build_bilateral_acceptance_scenario()

    assert scenario.total_satellite_count == 30
    assert len(scenario.friendly_observer_ids) == 15
    assert len(scenario.hostile_target_ids) == 15
    assert set(scenario.friendly_topology.node_ids) == set(
        scenario.friendly_observer_ids
    )
    assert set(scenario.friendly_observer_ids).isdisjoint(
        scenario.hostile_target_ids
    )
    assert scenario.friendly_state_assumption == (
        TEMPORARY_FRIENDLY_STATE_ASSUMPTION
    )


def test_registered_roles_allow_only_friendly_to_hostile_observations():
    scenario = build_bilateral_acceptance_scenario()
    friendly = scenario.friendly_observer_ids[0]
    hostile = scenario.hostile_target_ids[0]

    scenario.require_observation_pair(friendly, hostile)

    with pytest.raises(ValueError, match="not a friendly observer"):
        scenario.require_observation_pair(hostile, hostile)
    with pytest.raises(ValueError, match="not a registered hostile target"):
        scenario.require_observation_pair(friendly, friendly)


def test_communication_is_limited_to_configured_friendly_edges():
    scenario = build_bilateral_acceptance_scenario(topology_policy="ring")
    source = scenario.friendly_observer_ids[0]
    receiver = scenario.friendly_topology.neighbors(source)[0]

    scenario.require_communication_edge(source, receiver)

    with pytest.raises(ValueError, match="restricted to friendly"):
        scenario.require_communication_edge(source, scenario.hostile_target_ids[0])
    with pytest.raises(ValueError, match="absent from"):
        scenario.require_communication_edge(
            source, scenario.friendly_observer_ids[2]
        )


def test_scenario_rejects_a_hostile_target_in_the_communication_topology():
    registry = HostileTargetRegistry(("hostile_01",))
    invalid_topology = NetworkTopology({
        "friendly_01": ("hostile_01",),
        "hostile_01": ("friendly_01",),
    })

    with pytest.raises(ValueError, match="exactly the friendly observers"):
        BilateralMultiTargetScenario(
            scene_id="scene",
            friendly_observer_ids=("friendly_01",),
            hostile_targets=registry,
            friendly_topology=invalid_topology,
        )


def test_scenario_rejects_overlapping_role_identifiers():
    with pytest.raises(ValueError, match="must be disjoint"):
        BilateralMultiTargetScenario(
            scene_id="scene",
            friendly_observer_ids=("shared",),
            hostile_targets=HostileTargetRegistry(("shared",)),
            friendly_topology=chain_topology(("shared",)),
        )


def test_hostile_registry_does_not_accept_unknown_or_duplicate_targets():
    registry = HostileTargetRegistry(
        ("hostile_01", "hostile_02"),
        metadata_by_target={"hostile_01": {"truth_only": True}},
    )

    assert registry.require("hostile_01") == "hostile_01"
    assert registry.metadata_by_target["hostile_01"]["truth_only"] is True
    with pytest.raises(KeyError, match="Unknown hostile"):
        registry.require("hostile_03")
    with pytest.raises(ValueError, match="unique"):
        HostileTargetRegistry(("hostile_01", "hostile_01"))


def test_acceptance_builder_enforces_total_satellite_requirement():
    with pytest.raises(ValueError, match="at least 25"):
        build_bilateral_acceptance_scenario(
            friendly_count=12, hostile_count=12
        )


def test_bilateral_walker_truth_binds_nonmaneuvering_histories_to_roles():
    truth = generate_bilateral_walker_truth(
        duration_seconds=2.0, sample_interval_seconds=1.0
    )

    assert truth.manifest.total_satellite_count == 30
    assert truth.timestamps.tolist() == [0.0, 1.0, 2.0]
    assert set(truth.friendly_truth_state_history_by_node) == set(
        truth.manifest.friendly_observer_ids
    )
    assert set(truth.hostile_truth_state_history_by_target) == set(
        truth.manifest.hostile_target_ids
    )
    assert all(
        history.shape == (3, 6)
        for history in (
            *truth.friendly_truth_state_history_by_node.values(),
            *truth.hostile_truth_state_history_by_target.values(),
        )
    )
    assert all(
        truth.manifest.hostile_targets.metadata_by_target[target_id]["truth_only"]
        for target_id in truth.manifest.hostile_target_ids
    )


def test_bilateral_walker_truth_rejects_invalid_timing():
    with pytest.raises(ValueError, match="duration_seconds"):
        generate_bilateral_walker_truth(duration_seconds=0.0)
    with pytest.raises(ValueError, match="sample_interval_seconds"):
        generate_bilateral_walker_truth(sample_interval_seconds=0.0)
