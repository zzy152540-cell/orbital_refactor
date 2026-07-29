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
