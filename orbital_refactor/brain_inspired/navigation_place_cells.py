from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brain_inspired.navigation_brain_state import NavigationBrainStateHistory

Array = np.ndarray


@dataclass(frozen=True)
class NavigationPlaceCellConfig:
    phase_cell_count: int = 12
    radial_centers_m: tuple[float, ...] = (-100.0, 0.0, 100.0)
    along_track_centers_m: tuple[float, ...] = (-100.0, 0.0, 100.0)
    phase_concentration: float = 4.0
    radial_sigma_m: float = 50.0
    along_track_sigma_m: float = 50.0

    def validate(self):
        if self.phase_cell_count < 3:
            raise ValueError("phase_cell_count must be at least three.")
        radial = np.asarray(self.radial_centers_m, dtype=float)
        along = np.asarray(self.along_track_centers_m, dtype=float)
        if (radial.size == 0 or along.size == 0
                or np.any(~np.isfinite(radial)) or np.any(~np.isfinite(along))):
            raise ValueError("RT place-cell centers must be finite and nonempty.")
        positive = (
            self.phase_concentration, self.radial_sigma_m,
            self.along_track_sigma_m,
        )
        if np.any(~np.isfinite(positive)) or min(positive) <= 0.0:
            raise ValueError("Place-cell kernel scales must be finite and positive.")

    @property
    def cell_count(self):
        return (
            self.phase_cell_count * len(self.radial_centers_m)
            * len(self.along_track_centers_m)
        )


@dataclass(frozen=True)
class NavigationPlaceCellOutput:
    phase: float
    radial_position: float
    along_track_position: float
    decoded_phase: float
    decoded_radial_position: float
    decoded_along_track_position: float
    activity: Array
    peak_activity: float
    normalized_entropy: float
    valid: bool


@dataclass(frozen=True)
class NavigationPlaceCellHistory:
    node_id: str
    timestamps: Array
    activity: Array
    decoded_phase: Array
    decoded_rt: Array
    peak_activity: Array
    normalized_entropy: Array
    valid: Array


class NavigationPlaceCellEncoder:
    """Deterministic population display of phase and local RT position."""

    def __init__(self, config=None):
        self.config = config or NavigationPlaceCellConfig()
        self.config.validate()
        phase = np.linspace(
            0.0, 2.0 * np.pi, self.config.phase_cell_count, endpoint=False,
        )
        radial = np.asarray(self.config.radial_centers_m, dtype=float)
        along = np.asarray(self.config.along_track_centers_m, dtype=float)
        self.phase_centers, self.radial_centers, self.along_centers = (
            value.reshape(-1) for value in np.meshgrid(
                phase, radial, along, indexing="ij",
            )
        )

    def encode(self, *, phase, radial_position, along_track_position):
        values = np.asarray(
            [phase, radial_position, along_track_position], dtype=float,
        )
        if np.any(~np.isfinite(values)):
            raise ValueError("Place-cell inputs must be finite.")
        log_activity = (
            self.config.phase_concentration
            * (np.cos(float(phase) - self.phase_centers) - 1.0)
            - 0.5 * ((float(radial_position) - self.radial_centers)
                     / self.config.radial_sigma_m) ** 2
            - 0.5 * ((float(along_track_position) - self.along_centers)
                     / self.config.along_track_sigma_m) ** 2
        )
        activity = np.exp(log_activity - np.max(log_activity))
        activity /= np.sum(activity)
        phase_vector = np.sum(activity * np.exp(1j * self.phase_centers))
        decoded_phase = float(np.mod(np.angle(phase_vector), 2.0 * np.pi))
        decoded_radial = float(activity @ self.radial_centers)
        decoded_along = float(activity @ self.along_centers)
        entropy = -float(np.sum(activity * np.log(np.maximum(activity, 1e-15))))
        entropy /= np.log(activity.size)
        return NavigationPlaceCellOutput(
            phase=float(np.mod(phase, 2.0 * np.pi)),
            radial_position=float(radial_position),
            along_track_position=float(along_track_position),
            decoded_phase=decoded_phase,
            decoded_radial_position=decoded_radial,
            decoded_along_track_position=decoded_along,
            activity=activity, peak_activity=float(np.max(activity)),
            normalized_entropy=float(entropy),
            valid=bool(
                np.all(np.isfinite(activity)) and np.all(activity >= 0.0)
                and np.isclose(np.sum(activity), 1.0)
            ),
        )


def build_navigation_place_cell_histories(
    *, navigation_by_node: dict[str, NavigationBrainStateHistory], config=None,
):
    if not navigation_by_node:
        raise ValueError("Navigation histories must be nonempty.")
    encoder = NavigationPlaceCellEncoder(config)
    result = {}
    for node, navigation in navigation_by_node.items():
        outputs = [encoder.encode(
            phase=navigation.decoded_phase[index],
            radial_position=navigation.decoded_rt[index, 0],
            along_track_position=navigation.decoded_rt[index, 1],
        ) for index in range(navigation.timestamps.size)]
        result[node] = NavigationPlaceCellHistory(
            node_id=node, timestamps=navigation.timestamps.copy(),
            activity=np.asarray([item.activity for item in outputs]),
            decoded_phase=np.asarray([item.decoded_phase for item in outputs]),
            decoded_rt=np.asarray([[
                item.decoded_radial_position,
                item.decoded_along_track_position,
            ] for item in outputs]),
            peak_activity=np.asarray([item.peak_activity for item in outputs]),
            normalized_entropy=np.asarray([
                item.normalized_entropy for item in outputs
            ]),
            valid=navigation.valid & np.asarray([
                item.valid for item in outputs
            ], dtype=bool),
        )
    return result
