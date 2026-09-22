from __future__ import annotations

import numpy as np

from .constants import MU_EARTH


def keplerian_to_eci(
    semi_major_axis: float,
    eccentricity: float,
    inclination: float,
    raan: float,
    argument_of_perigee: float,
    true_anomaly: float,
    *,
    mu: float = MU_EARTH,
) -> np.ndarray:
    """Convert classical Keplerian elements to an ECI Cartesian state.

    Angular inputs are in radians and the returned state is ``[r, v]`` in SI units.
    """
    a = float(semi_major_axis)
    e = float(eccentricity)
    if a <= 0.0:
        raise ValueError("semi_major_axis must be positive.")
    if not 0.0 <= e < 1.0:
        raise ValueError("Only elliptic orbits with 0 <= eccentricity < 1 are supported.")

    p = a * (1.0 - e * e)
    cnu, snu = np.cos(true_anomaly), np.sin(true_anomaly)
    radius_pf = p / (1.0 + e * cnu) * np.array([cnu, snu, 0.0])
    velocity_pf = np.sqrt(mu / p) * np.array([-snu, e + cnu, 0.0])

    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inclination), np.sin(inclination)
    cw, sw = np.cos(argument_of_perigee), np.sin(argument_of_perigee)
    rotation = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si, cw * si, ci],
    ])
    return np.hstack((rotation @ radius_pf, rotation @ velocity_pf))


def eci_to_keplerian(state_eci: np.ndarray, *, mu: float = MU_EARTH) -> tuple[float, ...]:
    """Convert an elliptic ECI Cartesian state to classical osculating elements.

    The returned tuple is ``(a, e, i, raan, argument_of_perigee, true_anomaly)``
    in SI units and radians. Circular/equatorial singularities use the standard
    zero-angle convention while preserving a round-trip Cartesian state.
    """

    state = np.asarray(state_eci, dtype=float).reshape(6)
    if np.any(~np.isfinite(state)):
        raise ValueError("ECI state must contain six finite values.")
    position, velocity = state[:3], state[3:]
    radius = float(np.linalg.norm(position))
    speed_squared = float(velocity @ velocity)
    if radius <= 0.0:
        raise ValueError("ECI position magnitude must be positive.")
    angular_momentum = np.cross(position, velocity)
    h_norm = float(np.linalg.norm(angular_momentum))
    if h_norm <= 0.0:
        raise ValueError("ECI state must have nonzero angular momentum.")
    node = np.cross(np.array([0.0, 0.0, 1.0]), angular_momentum)
    node_norm = float(np.linalg.norm(node))
    eccentricity_vector = (
        np.cross(velocity, angular_momentum) / float(mu) - position / radius
    )
    eccentricity = float(np.linalg.norm(eccentricity_vector))
    specific_energy = 0.5 * speed_squared - float(mu) / radius
    if specific_energy >= 0.0 or eccentricity >= 1.0:
        raise ValueError("Only bound elliptic ECI states are supported.")
    semi_major_axis = -float(mu) / (2.0 * specific_energy)
    inclination = float(np.arccos(np.clip(angular_momentum[2] / h_norm, -1.0, 1.0)))

    def oriented_angle(first, second, normal):
        cosine = np.clip(
            float(first @ second) / (np.linalg.norm(first) * np.linalg.norm(second)),
            -1.0, 1.0,
        )
        angle = float(np.arccos(cosine))
        return angle if float(np.dot(np.cross(first, second), normal)) >= 0.0 else 2.0 * np.pi - angle

    tolerance = 1.0e-12
    raan = float(np.mod(np.arctan2(node[1], node[0]), 2.0 * np.pi)) if node_norm > tolerance else 0.0
    if eccentricity > tolerance:
        argument = (
            oriented_angle(node, eccentricity_vector, angular_momentum)
            if node_norm > tolerance else
            float(np.mod(np.arctan2(eccentricity_vector[1], eccentricity_vector[0]), 2.0 * np.pi))
        )
        true_anomaly = oriented_angle(eccentricity_vector, position, angular_momentum)
    else:
        argument = 0.0
        reference = node if node_norm > tolerance else np.array([1.0, 0.0, 0.0])
        true_anomaly = oriented_angle(reference, position, angular_momentum)
    return (
        semi_major_axis, eccentricity, inclination, raan,
        float(argument), float(true_anomaly),
    )
