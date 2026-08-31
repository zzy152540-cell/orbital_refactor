from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brain_inspired.passive_phase_observer import PeriodicStateInput

Array = np.ndarray
TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class OrbitalPlaneFrame:
    """Fixed ECI basis used to define argument of latitude on one orbit plane."""

    ascending_node_axis: Array
    in_plane_quadrature_axis: Array
    normal_axis: Array

    @classmethod
    def from_state_eci(cls, state_eci: Array) -> "OrbitalPlaneFrame":
        """Build a fixed orbital-plane basis from one nonsingular ECI state."""

        state = np.asarray(state_eci, dtype=float).reshape(-1)
        if state.shape != (6,) or np.any(~np.isfinite(state)):
            raise ValueError("state_eci must be a finite [position, velocity] 6-vector.")
        position, velocity = state[:3], state[3:]
        position_norm = float(np.linalg.norm(position))
        angular_momentum = np.cross(position, velocity)
        momentum_norm = float(np.linalg.norm(angular_momentum))
        if position_norm <= np.finfo(float).tiny or momentum_norm <= np.finfo(float).tiny:
            raise ValueError("state_eci must define a nonsingular orbital plane.")
        ascending = position / position_norm
        normal = angular_momentum / momentum_norm
        quadrature = np.cross(normal, ascending)
        return cls(ascending, quadrature, normal)

    @classmethod
    def from_raan_inclination(
        cls, *, raan: float, inclination: float,
    ) -> "OrbitalPlaneFrame":
        if not np.isfinite(raan) or not np.isfinite(inclination):
            raise ValueError("RAAN and inclination must be finite.")
        if not 0.0 <= inclination <= np.pi:
            raise ValueError("Inclination must be in [0, pi].")
        cosine_raan, sine_raan = np.cos(raan), np.sin(raan)
        cosine_inc, sine_inc = np.cos(inclination), np.sin(inclination)
        ascending = np.array([cosine_raan, sine_raan, 0.0])
        quadrature = np.array([
            -sine_raan * cosine_inc,
            cosine_raan * cosine_inc,
            sine_inc,
        ])
        normal = np.cross(ascending, quadrature)
        return cls(ascending, quadrature, normal)

    def __post_init__(self) -> None:
        axes = tuple(np.asarray(axis, dtype=float).reshape(-1) for axis in (
            self.ascending_node_axis, self.in_plane_quadrature_axis,
            self.normal_axis,
        ))
        if any(axis.shape != (3,) or np.any(~np.isfinite(axis)) for axis in axes):
            raise ValueError("Orbital-plane axes must be finite 3-vectors.")
        basis = np.vstack(axes)
        if not np.allclose(basis @ basis.T, np.eye(3), atol=1.0e-10):
            raise ValueError("Orbital-plane axes must be orthonormal.")
        if np.linalg.det(basis) < 1.0 - 1.0e-10:
            raise ValueError("Orbital-plane axes must form a right-handed basis.")
        for field_name, axis in zip((
            "ascending_node_axis", "in_plane_quadrature_axis", "normal_axis",
        ), axes):
            object.__setattr__(self, field_name, axis.copy())


@dataclass(frozen=True)
class OrbitalPhaseState:
    timestamp: float
    argument_of_latitude: float
    argument_of_latitude_rate: float
    in_plane_radius: float
    cross_track_position: float
    source_id: str | None = None

    def as_periodic_input(self, *, use_phase_hint: bool) -> PeriodicStateInput:
        return PeriodicStateInput(
            timestamp=self.timestamp,
            phase_rate=self.argument_of_latitude_rate,
            phase_hint=(self.argument_of_latitude if use_phase_hint else None),
            phase_hint_valid=bool(use_phase_hint),
            source_id=self.source_id,
        )


def extract_orbital_phase_state(
    *, timestamp: float, state_eci: Array, frame: OrbitalPlaneFrame,
    source_id: str | None = None,
) -> OrbitalPhaseState:
    """Project an absolute ECI Cartesian state onto a fixed orbital plane."""

    if not np.isfinite(timestamp):
        raise ValueError("Orbital-state timestamp must be finite.")
    state = np.asarray(state_eci, dtype=float).reshape(-1)
    if state.shape != (6,) or np.any(~np.isfinite(state)):
        raise ValueError("state_eci must be a finite [position, velocity] 6-vector.")
    position, velocity = state[:3], state[3:]
    x = float(position @ frame.ascending_node_axis)
    y = float(position @ frame.in_plane_quadrature_axis)
    x_rate = float(velocity @ frame.ascending_node_axis)
    y_rate = float(velocity @ frame.in_plane_quadrature_axis)
    radius_squared = x * x + y * y
    if radius_squared <= np.finfo(float).tiny:
        raise ValueError("Projected in-plane position must be nonzero.")
    return OrbitalPhaseState(
        timestamp=float(timestamp),
        argument_of_latitude=float(np.arctan2(y, x) % TWO_PI),
        argument_of_latitude_rate=float(
            (x * y_rate - y * x_rate) / radius_squared
        ),
        in_plane_radius=float(np.sqrt(radius_squared)),
        cross_track_position=float(position @ frame.normal_axis),
        source_id=source_id,
    )
