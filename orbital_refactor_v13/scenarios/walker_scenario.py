from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbital_core.orbit_elements import keplerian_to_eci
from scenarios.fleet_scenario import FleetScenario, generate_fleet_scenario

Array = np.ndarray


@dataclass(frozen=True)
class WalkerDeltaConfig:
    """Classical Walker-Delta ``T/P/F`` constellation definition.

    Plane ``p`` uses ``RAAN = raan_origin + 2*pi*p/P``. Satellite ``s`` in
    that plane uses
    ``true_anomaly = base_true_anomaly + 2*pi*s/(T/P) + 2*pi*F*p/T``.
    This explicit convention is intended to match a later STK configuration.
    """

    total_satellites: int
    plane_count: int
    phasing: int
    semi_major_axis: float
    eccentricity: float
    inclination: float
    raan_origin: float = 0.0
    argument_of_perigee: float = 0.0
    base_true_anomaly: float = 0.0
    node_prefix: str = "sat"

    def __post_init__(self) -> None:
        if self.total_satellites < 2:
            raise ValueError("total_satellites must be at least two.")
        if self.plane_count < 1:
            raise ValueError("plane_count must be positive.")
        if self.total_satellites % self.plane_count != 0:
            raise ValueError("total_satellites must be divisible by plane_count.")
        if not 0 <= self.phasing < self.plane_count:
            raise ValueError("phasing must be in [0, plane_count).")
        if self.semi_major_axis <= 0.0:
            raise ValueError("semi_major_axis must be positive.")
        if not 0.0 <= self.eccentricity < 1.0:
            raise ValueError("eccentricity must be in [0, 1).")
        if not 0.0 <= self.inclination <= np.pi:
            raise ValueError("inclination must be in [0, pi].")
        angular_values = (
            self.raan_origin, self.argument_of_perigee,
            self.base_true_anomaly,
        )
        if not np.all(np.isfinite(angular_values)):
            raise ValueError("Walker angular parameters must be finite.")
        if not str(self.node_prefix):
            raise ValueError("node_prefix cannot be empty.")

    @property
    def satellites_per_plane(self) -> int:
        return self.total_satellites // self.plane_count


@dataclass(frozen=True)
class WalkerSatelliteElements:
    satellite_id: str
    plane_index: int
    slot_index: int
    raan: float
    true_anomaly: float


@dataclass(frozen=True)
class WalkerDeltaScenario:
    config: WalkerDeltaConfig
    fleet: FleetScenario
    elements_by_node: dict[str, WalkerSatelliteElements]

    @property
    def node_ids(self) -> tuple[str, ...]:
        return self.fleet.node_ids

    @property
    def truth_state_history_by_node(self) -> dict[str, Array]:
        return self.fleet.truth_state_history_by_node


def generate_walker_delta_scenario(
    *, timestamps: Array, config: WalkerDeltaConfig,
) -> WalkerDeltaScenario:
    """Generate and independently propagate every Walker satellite in ECI."""

    initial_states = {}
    elements = {}
    slots = config.satellites_per_plane
    for plane_index in range(config.plane_count):
        raan = config.raan_origin + 2.0 * np.pi * plane_index / config.plane_count
        plane_phase = 2.0 * np.pi * config.phasing * plane_index / config.total_satellites
        for slot_index in range(slots):
            satellite_id = (
                f"{config.node_prefix}_p{plane_index + 1:02d}"
                f"_s{slot_index + 1:02d}"
            )
            true_anomaly = (
                config.base_true_anomaly
                + 2.0 * np.pi * slot_index / slots
                + plane_phase
            )
            initial_states[satellite_id] = keplerian_to_eci(
                config.semi_major_axis, config.eccentricity,
                config.inclination, raan, config.argument_of_perigee,
                true_anomaly,
            )
            elements[satellite_id] = WalkerSatelliteElements(
                satellite_id=satellite_id, plane_index=plane_index,
                slot_index=slot_index, raan=float(np.mod(raan, 2.0 * np.pi)),
                true_anomaly=float(np.mod(true_anomaly, 2.0 * np.pi)),
            )
    fleet = generate_fleet_scenario(
        timestamps=timestamps, initial_state_by_node=initial_states,
    )
    return WalkerDeltaScenario(config=config, fleet=fleet, elements_by_node=elements)
