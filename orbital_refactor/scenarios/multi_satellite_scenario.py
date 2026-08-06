from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbital_core.coordinates import build_rtn_quaternion_history, state_history_eci_to_spri
from orbital_core.dynamics import propagate_absolute_orbit

Array = np.ndarray


@dataclass(frozen=True)
class SatelliteTrajectory:
    satellite_id: str
    timestamps: Array
    state_history_eci: Array
    q_eci2pri_history: Array


@dataclass(frozen=True)
class CooperativeScenario:
    target_id: str
    timestamps: Array
    target_trajectory: SatelliteTrajectory
    observer_trajectories: dict[str, SatelliteTrajectory]
    relative_state_eci_by_node: dict[str, Array]
    relative_state_spri_by_node: dict[str, Array]


def generate_cooperative_scenario(
    *,
    timestamps: Array,
    target_id: str,
    target_initial_state_eci: Array,
    observer_initial_states_eci: dict[str, Array],
) -> CooperativeScenario:
    """Generate one target and multiple observer trajectories under the same model."""
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
    if not observer_initial_states_eci:
        raise ValueError("At least one observer initial state is required.")

    target_history = propagate_absolute_orbit(target_initial_state_eci, timestamps)
    target_q = build_rtn_quaternion_history(target_history)
    target = SatelliteTrajectory(target_id, timestamps.copy(), target_history, target_q)

    observers: dict[str, SatelliteTrajectory] = {}
    relative_eci: dict[str, Array] = {}
    relative_spri: dict[str, Array] = {}
    for node_id, initial_state in observer_initial_states_eci.items():
        observer_history = propagate_absolute_orbit(initial_state, timestamps)
        q_history = build_rtn_quaternion_history(observer_history)
        rel_eci = target_history - observer_history
        rel_spri = state_history_eci_to_spri(rel_eci, q_history)
        observers[node_id] = SatelliteTrajectory(
            node_id, timestamps.copy(), observer_history, q_history
        )
        relative_eci[node_id] = rel_eci
        relative_spri[node_id] = rel_spri

    return CooperativeScenario(
        target_id=target_id,
        timestamps=timestamps.copy(),
        target_trajectory=target,
        observer_trajectories=observers,
        relative_state_eci_by_node=relative_eci,
        relative_state_spri_by_node=relative_spri,
    )
