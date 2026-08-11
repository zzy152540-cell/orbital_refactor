from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbital_core.constants import R_EARTH
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
