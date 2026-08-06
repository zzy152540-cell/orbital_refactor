from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cooperative.topology import (
    NetworkTopology,
    chain_topology,
    ring_topology,
    star_topology,
    two_hop_chain_topology,
)
from orbital_core.constants import R_EARTH
from orbital_core.orbit_elements import keplerian_to_eci
from scenarios.fleet_scenario import FleetScenario, generate_fleet_scenario
from scenarios.measurement_visibility import (
    VisibilityOpportunitySummary,
    generate_inter_satellite_observation_opportunities,
    stabilize_observation_opportunities,
    summarize_observation_opportunities,
)

Array = np.ndarray


@dataclass(frozen=True)
class FleetGeometrySetup:
    scenario: FleetScenario
    truth_initial_state_by_node: dict[str, Array]
    truth_state_history_by_node: dict[str, Array]
    topology: NetworkTopology


def build_fleet_geometry(
    *, timestamps, node_count: int, truth_initial_state_by_node,
    topology_type: str, topology_override: NetworkTopology | None,
) -> FleetGeometrySetup:
    """Build deterministic fleet truth and the configured candidate topology."""

    base = keplerian_to_eci(
        R_EARTH + 700e3, 0.001, np.deg2rad(23.0), 0.0, 0.0, 0.0
    )
    center = 0.5 * (node_count - 1)
    truth_initials = (
        {
            f"sat_{index + 1:02d}": base + np.array([
                1200.0 * (index - center),
                100.0 * np.sin(2.0 * np.pi * index / node_count),
                30.0 * np.cos(2.0 * np.pi * index / node_count),
                0.0,
                0.02 * (index - center),
                0.0,
            ])
            for index in range(node_count)
        }
        if truth_initial_state_by_node is None
        else {
            str(node_id): np.asarray(state, dtype=float).reshape(6).copy()
            for node_id, state in truth_initial_state_by_node.items()
        }
    )
    if len(truth_initials) != node_count:
        raise ValueError("Truth initial-state count must match node_count.")
    scenario = generate_fleet_scenario(
        timestamps=timestamps, initial_state_by_node=truth_initials
    )
    topology_builders = {
        "chain": chain_topology,
        "ring": ring_topology,
        "star": star_topology,
        "two_hop_chain": two_hop_chain_topology,
    }
    if topology_override is None:
        if topology_type not in topology_builders:
            raise ValueError("Unsupported topology_type.")
        topology = topology_builders[topology_type](list(scenario.node_ids))
    else:
        if set(topology_override.node_ids) != set(scenario.node_ids):
            raise ValueError("topology_override nodes must match scenario nodes.")
        topology = topology_override
    return FleetGeometrySetup(
        scenario=scenario,
        truth_initial_state_by_node=truth_initials,
        truth_state_history_by_node=scenario.truth_state_history_by_node,
        topology=topology,
    )


def normalize_attitude_inputs(
    *, frame, attitude_history_by_node, estimated_attitude_history_by_node,
    attitude_covariance, node_ids, sample_count,
):
    """Validate and normalize optional BODY-frame attitude inputs."""

    frame = str(frame).upper()
    if frame not in {"ECI", "BODY"}:
        raise ValueError("az_el_frame must be 'ECI' or 'BODY'.")
    if frame != "BODY":
        return (
            frame,
            attitude_history_by_node,
            estimated_attitude_history_by_node,
            attitude_covariance,
        )
    if attitude_history_by_node is None:
        raise ValueError("BODY AZ_EL requires attitude_history_by_node.")
    if set(attitude_history_by_node) != set(node_ids):
        raise ValueError("Attitude-history keys must match scenario nodes.")
    attitude_history_by_node = {
        node: np.asarray(values, dtype=float).reshape(sample_count, 4)
        for node, values in attitude_history_by_node.items()
    }
    if estimated_attitude_history_by_node is None:
        estimated_attitude_history_by_node = attitude_history_by_node
    if set(estimated_attitude_history_by_node) != set(node_ids):
        raise ValueError("Estimated-attitude keys must match scenario nodes.")
    estimated_attitude_history_by_node = {
        node: np.asarray(values, dtype=float).reshape(sample_count, 4)
        for node, values in estimated_attitude_history_by_node.items()
    }
    if attitude_covariance is not None:
        attitude_covariance = np.asarray(
            attitude_covariance, dtype=float
        ).reshape(3, 3)
    return (
        frame,
        attitude_history_by_node,
        estimated_attitude_history_by_node,
        attitude_covariance,
    )


def build_visibility_selection(
    *, timestamps, truth_state_history_by_node, topology,
    relative_modalities, visibility_by_modality,
    visibility_temporal_filter_by_modality, attitude_history_by_node,
    frame,
) -> tuple[VisibilityOpportunitySummary | None, set | None]:
    """Build the optional visibility summary and visible observation keys."""

    if visibility_by_modality is None:
        return None, None
    if set(visibility_by_modality) != set(relative_modalities):
        raise ValueError(
            "Visibility configurations must match enabled relative modalities."
        )
    opportunities = generate_inter_satellite_observation_opportunities(
        timestamps=timestamps,
        truth_state_history_by_node=truth_state_history_by_node,
        candidate_topology=topology,
        visibility_by_modality=visibility_by_modality,
        attitude_history_by_node=(
            attitude_history_by_node if frame == "BODY" else None
        ),
    )
    if visibility_temporal_filter_by_modality is not None:
        opportunities = stabilize_observation_opportunities(
            opportunities,
            visibility_by_modality=visibility_by_modality,
            temporal_filter_by_modality=visibility_temporal_filter_by_modality,
        )
    summary = summarize_observation_opportunities(opportunities)
    visible_keys = {
        (item.timestamp, item.observer_id, item.target_id, item.modality)
        for item in opportunities
        if item.visibility.visible
    }
    return summary, visible_keys
