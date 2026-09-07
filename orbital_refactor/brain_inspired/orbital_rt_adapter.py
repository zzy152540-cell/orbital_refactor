from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class OrbitalRTOffset:
    timestamp: float
    radial_position: float
    along_track_position: float
    radial_rate: float
    along_track_rate: float
    frame_angular_rate: float
    source_id: str | None = None


def extract_orbital_rt_offset(
    *, timestamp: float, state_eci: Array, reference_state_eci: Array,
    source_id: str | None = None,
) -> OrbitalRTOffset:
    """Project state-minus-reference into the rotating reference RT frame."""
    if not np.isfinite(timestamp):
        raise ValueError("RT offset timestamp must be finite.")
    state = _state(state_eci, "state_eci")
    reference = _state(reference_state_eci, "reference_state_eci")
    radius = float(np.linalg.norm(reference[:3]))
    momentum = np.cross(reference[:3], reference[3:])
    momentum_norm = float(np.linalg.norm(momentum))
    if radius <= np.finfo(float).tiny or momentum_norm <= np.finfo(float).tiny:
        raise ValueError("Reference state must define a nonsingular RT frame.")
    radial_axis = reference[:3] / radius
    normal_axis = momentum / momentum_norm
    transverse_axis = np.cross(normal_axis, radial_axis)
    delta_position = state[:3] - reference[:3]
    delta_velocity = state[3:] - reference[3:]
    radial = float(delta_position @ radial_axis)
    transverse = float(delta_position @ transverse_axis)
    inertial_radial_rate = float(delta_velocity @ radial_axis)
    inertial_transverse_rate = float(delta_velocity @ transverse_axis)
    angular_rate = momentum_norm / radius**2
    return OrbitalRTOffset(
        timestamp=float(timestamp), radial_position=radial,
        along_track_position=transverse,
        radial_rate=float(inertial_radial_rate + angular_rate * transverse),
        along_track_rate=float(inertial_transverse_rate - angular_rate * radial),
        frame_angular_rate=float(angular_rate), source_id=source_id,
    )


def _state(value, name):
    state = np.asarray(value, dtype=float).reshape(-1)
    if state.shape != (6,) or np.any(~np.isfinite(state)):
        raise ValueError(f"{name} must be a finite 6-vector.")
    return state
