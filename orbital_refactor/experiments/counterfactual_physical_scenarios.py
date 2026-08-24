from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbital_core.constants import R_EARTH
from orbital_core.orbit_elements import keplerian_to_eci
from scenarios.fleet_scenario import (
    centered_along_track_offsets,
    generate_differential_orbit_fleet_scenario,
)


@dataclass(frozen=True)
class CounterfactualPhysicalScenario:
    scenario_id: str
    along_track_spacing: float
    semi_major_axis_step: float
    base_true_anomaly: float
    truth_initial_states: tuple[tuple[str, tuple[float, ...]], ...]

    def truth_initial_state_by_node(self):
        return {
            node: np.asarray(state, dtype=float)
            for node, state in self.truth_initial_states
        }


@dataclass(frozen=True)
class RandomizedFleetPhysicalScenario:
    """One seeded, physically parameterized compact-fleet condition."""

    scenario_id: str
    family: str
    node_count: int
    semi_major_axis: float
    eccentricity: float
    inclination: float
    base_true_anomaly: float
    along_track_spacing: float
    semi_major_axis_step: float
    raan_separation: float
    plane_index_by_node: tuple[tuple[str, int], ...]
    truth_initial_states: tuple[tuple[str, tuple[float, ...]], ...]

    def truth_initial_state_by_node(self):
        return {
            node: np.asarray(state, dtype=float)
            for node, state in self.truth_initial_states
        }


FIVE_NODE_PHYSICAL_FAMILIES = (
    "compact_along_track",
    "differential_along_track",
    "two_plane_cluster",
)


def sample_five_satellite_physical_scenario(
    seed: int,
    *,
    families: tuple[str, ...] = FIVE_NODE_PHYSICAL_FAMILIES,
    family_assignment_mode: str = "random",
) -> RandomizedFleetPhysicalScenario:
    """Sample one reproducible five-satellite orbit condition.

    Every state is generated from valid Keplerian elements.  The three families
    deliberately separate compact debugging geometry, secularly evolving
    along-track geometry, and a small multi-plane transfer case.
    """

    if not families or len(set(families)) != len(families):
        raise ValueError("Physical scenario families must be unique and nonempty.")
    unknown = set(families) - set(FIVE_NODE_PHYSICAL_FAMILIES)
    if unknown:
        raise ValueError(f"Unsupported physical scenario families: {sorted(unknown)}")
    if family_assignment_mode not in {"random", "seed_cycle"}:
        raise ValueError("Unsupported physical family-assignment mode.")
    rng = np.random.default_rng(20260917 + int(seed))
    family = (
        str(rng.choice(families))
        if family_assignment_mode == "random"
        else families[int(seed) % len(families)]
    )
    semi_major_axis = float(R_EARTH + rng.uniform(550e3, 850e3))
    eccentricity = float(rng.uniform(0.0, 0.002))
    inclination = float(np.deg2rad(rng.uniform(20.0, 70.0)))
    base_true_anomaly = float(rng.uniform(0.0, 2.0 * np.pi))
    argument_of_perigee = float(rng.uniform(0.0, 2.0 * np.pi))
    raan_origin = float(rng.uniform(0.0, 2.0 * np.pi))
    if family == "compact_along_track":
        spacing = float(rng.uniform(1e3, 5e3))
        semi_major_axis_step = float(rng.uniform(0.0, 200.0))
        raan_separation = 0.0
        plane_indices = (0, 0, 0, 0, 0)
    elif family == "differential_along_track":
        spacing = float(rng.uniform(5e3, 30e3))
        semi_major_axis_step = float(rng.uniform(200.0, 1500.0))
        raan_separation = 0.0
        plane_indices = (0, 0, 0, 0, 0)
    else:
        spacing = float(rng.uniform(5e3, 20e3))
        semi_major_axis_step = float(rng.uniform(0.0, 500.0))
        raan_separation = float(np.deg2rad(rng.uniform(0.2, 1.0)))
        plane_indices = (0, 0, 0, 1, 1)

    center_by_plane = {
        plane: 0.5 * (plane_indices.count(plane) - 1)
        for plane in set(plane_indices)
    }
    slot_by_plane = {plane: 0 for plane in set(plane_indices)}
    states = []
    plane_records = []
    global_center = 0.5 * (len(plane_indices) - 1)
    for index, plane_index in enumerate(plane_indices):
        node = f"sat_{index + 1:02d}"
        slot = slot_by_plane[plane_index]
        slot_by_plane[plane_index] += 1
        anomaly_offset = (
            slot - center_by_plane[plane_index]
        ) * spacing / semi_major_axis
        state = keplerian_to_eci(
            semi_major_axis
            + (index - global_center) * semi_major_axis_step,
            eccentricity, inclination,
            raan_origin + plane_index * raan_separation,
            argument_of_perigee, base_true_anomaly + anomaly_offset,
        )
        states.append((node, tuple(float(value) for value in state)))
        plane_records.append((node, int(plane_index)))
    return RandomizedFleetPhysicalScenario(
        scenario_id=f"five_node_{family}_seed_{int(seed)}",
        family=family, node_count=5,
        semi_major_axis=semi_major_axis, eccentricity=eccentricity,
        inclination=inclination, base_true_anomaly=base_true_anomaly,
        along_track_spacing=spacing,
        semi_major_axis_step=semi_major_axis_step,
        raan_separation=raan_separation,
        plane_index_by_node=tuple(plane_records),
        truth_initial_states=tuple(states),
    )


def build_three_satellite_counterfactual_scenarios(
) -> tuple[CounterfactualPhysicalScenario, ...]:
    """Create a small first geometry axis without changing sensor semantics."""

    definitions = (
        ("compact_equal_orbit", 1500.0, 0.0, 0.0),
        ("medium_differential", 3500.0, 400.0, 0.15),
        ("wide_differential", 7000.0, 1000.0, 0.35),
    )
    return tuple(
        _build_scenario(
            scenario_id=scenario_id,
            spacing=spacing,
            semi_major_axis_step=semi_major_axis_step,
            base_true_anomaly=base_true_anomaly,
        )
        for (
            scenario_id, spacing, semi_major_axis_step, base_true_anomaly
        ) in definitions
    )


def _build_scenario(
    *, scenario_id, spacing, semi_major_axis_step, base_true_anomaly,
):
    orbital_radius = R_EARTH + 700e3
    offsets = centered_along_track_offsets(
        node_count=3,
        orbital_radius=orbital_radius,
        spacing=spacing,
        semi_major_axis_step=semi_major_axis_step,
    )
    fleet = generate_differential_orbit_fleet_scenario(
        timestamps=np.asarray([0.0]),
        base_semi_major_axis=orbital_radius,
        eccentricity=0.001,
        inclination=np.deg2rad(23.0),
        raan=0.0,
        argument_of_perigee=0.0,
        base_true_anomaly=base_true_anomaly,
        offset_by_node=offsets,
    )
    truth = tuple(
        (node, tuple(float(value) for value in history[0]))
        for node, history in fleet.truth_state_history_by_node.items()
    )
    return CounterfactualPhysicalScenario(
        scenario_id=scenario_id,
        along_track_spacing=spacing,
        semi_major_axis_step=semi_major_axis_step,
        base_true_anomaly=base_true_anomaly,
        truth_initial_states=truth,
    )
