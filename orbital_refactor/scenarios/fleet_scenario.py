from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

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


def centered_along_track_offsets(
    *, node_count: int, orbital_radius: float, spacing: float,
    semi_major_axis_step: float = 0.0,
    semi_major_axis_offsets: Sequence[float] | None = None,
    node_prefix: str = "sat",
) -> dict[str, DifferentialOrbitOffset]:
    """Build a centered, uniformly spaced along-track formation.

    ``spacing`` is converted to a true-anomaly separation using the reference
    orbital radius.  The returned offsets can be passed directly to
    :func:`generate_differential_orbit_fleet_scenario` and keep the formation
    definition independent from Cartesian test perturbations.
    """

    if node_count < 2:
        raise ValueError("node_count must be at least two.")
    if orbital_radius <= 0.0:
        raise ValueError("orbital_radius must be positive.")
    if spacing <= 0.0:
        raise ValueError("spacing must be positive.")
    if semi_major_axis_offsets is not None:
        if semi_major_axis_step != 0.0:
            raise ValueError(
                "Specify either semi_major_axis_step or explicit offsets, not both."
            )
        if len(semi_major_axis_offsets) != node_count:
            raise ValueError("Explicit semi-major-axis offsets must match node_count.")
        semi_major_axes = [float(value) for value in semi_major_axis_offsets]
        if not np.all(np.isfinite(semi_major_axes)):
            raise ValueError("Semi-major-axis offsets must be finite.")
    else:
        semi_major_axes = [
            (index - 0.5 * (node_count - 1)) * semi_major_axis_step
            for index in range(node_count)
        ]
    prefix = str(node_prefix)
    if not prefix:
        raise ValueError("node_prefix cannot be empty.")
    center = 0.5 * (node_count - 1)
    return {
        f"{prefix}_{index + 1:02d}": DifferentialOrbitOffset(
            semi_major_axis=semi_major_axes[index],
            true_anomaly=(index - center) * spacing / orbital_radius,
        )
        for index in range(node_count)
    }


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
