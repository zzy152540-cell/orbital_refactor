from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from orbital_core.coordinates import build_rtn_quaternion_history
from orbital_core.dynamics import propagate_absolute_orbit
from orbital_core.orbit_elements import keplerian_to_eci

Array = np.ndarray


@dataclass(frozen=True)
class FleetTrajectory:
    satellite_id: str
    timestamps: Array
    state_history_eci: Array
    q_eci2rtn_history: Array


@dataclass(frozen=True)
class FleetScenario:
    """Symmetric N-satellite truth scenario with no target/observer distinction."""

    timestamps: Array
    trajectories: dict[str, FleetTrajectory]

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(self.trajectories)

    @property
    def truth_state_history_by_node(self) -> dict[str, Array]:
        return {
            node_id: trajectory.state_history_eci
            for node_id, trajectory in self.trajectories.items()
        }

    def stacked_state_history(self) -> Array:
        return np.hstack(
            [self.trajectories[node_id].state_history_eci for node_id in self.node_ids]
        )


@dataclass(frozen=True)
class DifferentialOrbitOffset:
    semi_major_axis: float = 0.0
    true_anomaly: float = 0.0


def generate_fleet_scenario(
    *,
    timestamps: Array,
    initial_state_by_node: Mapping[str, Array],
) -> FleetScenario:
    times = np.asarray(timestamps, dtype=float).reshape(-1)
    if times.size == 0:
        raise ValueError("timestamps cannot be empty.")
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError("timestamps must be strictly increasing.")
    if not initial_state_by_node:
        raise ValueError("At least one satellite initial state is required.")

    trajectories: dict[str, FleetTrajectory] = {}
    for raw_node_id, initial_state in initial_state_by_node.items():
        node_id = str(raw_node_id)
        if node_id in trajectories:
            raise ValueError(f"Duplicate satellite ID after normalization: {node_id}")
        history = propagate_absolute_orbit(initial_state, times)
        trajectories[node_id] = FleetTrajectory(
            satellite_id=node_id,
            timestamps=times.copy(),
            state_history_eci=history,
            q_eci2rtn_history=build_rtn_quaternion_history(history),
        )
    return FleetScenario(timestamps=times.copy(), trajectories=trajectories)


def generate_differential_orbit_fleet_scenario(
    *, timestamps: Array,
    base_semi_major_axis: float,
    eccentricity: float,
    inclination: float,
    raan: float,
    argument_of_perigee: float,
    base_true_anomaly: float,
    offset_by_node: Mapping[str, DifferentialOrbitOffset],
) -> FleetScenario:
    """Generate independently propagated satellites from differential elements."""

    if not offset_by_node:
        raise ValueError("At least one differential orbit node is required.")
    initial_states = {}
    for raw_node_id, offset in offset_by_node.items():
        node_id = str(raw_node_id)
        if node_id in initial_states:
            raise ValueError(f"Duplicate satellite ID after normalization: {node_id}")
        if not isinstance(offset, DifferentialOrbitOffset):
            raise TypeError("Every node offset must be DifferentialOrbitOffset.")
        semi_major_axis = float(base_semi_major_axis + offset.semi_major_axis)
        if semi_major_axis <= 0.0:
            raise ValueError("Offset semi-major axes must remain positive.")
        initial_states[node_id] = keplerian_to_eci(
            semi_major_axis, eccentricity, inclination, raan,
            argument_of_perigee, base_true_anomaly + offset.true_anomaly,
        )
    return generate_fleet_scenario(
        timestamps=timestamps, initial_state_by_node=initial_states,
    )
