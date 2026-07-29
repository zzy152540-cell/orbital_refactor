from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from interfaces.data_objects import InterSatelliteObservation

Array = np.ndarray


@dataclass(frozen=True)
class InterSatelliteRangeAdapterResult:
    range_measurements_by_node: dict[str, dict[str, Array]]
    range_variance: float


@dataclass(frozen=True)
class InterSatelliteObservationAdapterResult:
    measurements_by_node: dict[str, dict[str, dict[str, Array]]]
    covariance_by_modality: dict[str, Array]

    @property
    def variance_by_modality(self) -> dict[str, float]:
        return {
            modality: float(covariance[0, 0])
            for modality, covariance in self.covariance_by_modality.items()
            if covariance.shape == (1, 1)
        }


_MODALITY_ALIASES = {
    "RANGE": "RANGE",
    "INTER_SATELLITE_RANGE": "RANGE",
    "RANGE_RATE": "RANGE_RATE",
    "RANGERATE": "RANGE_RATE",
    "INTER_SATELLITE_RANGE_RATE": "RANGE_RATE",
    "AZ_EL": "AZ_EL",
    "AZEL": "AZ_EL",
    "ANGLE": "AZ_EL",
    "INTER_SATELLITE_AZ_EL": "AZ_EL",
}


def adapt_inter_satellite_range_observations(
    observations: Iterable[InterSatelliteObservation],
    *,
    timestamps: Array,
) -> InterSatelliteRangeAdapterResult:
    """Convert RANGE inter-satellite observations to runner range histories."""

    adapted = adapt_inter_satellite_observations(observations, timestamps=timestamps)
    if set(adapted.variance_by_modality) != {"RANGE"}:
        raise ValueError("Only RANGE observations can be converted to raw range histories.")
    return InterSatelliteRangeAdapterResult(
        range_measurements_by_node={
            source: {
                target: modalities["RANGE"]
                for target, modalities in targets.items()
                if "RANGE" in modalities
            }
            for source, targets in adapted.measurements_by_node.items()
        },
        range_variance=adapted.variance_by_modality["RANGE"],
    )


def adapt_inter_satellite_observations(
    observations: Iterable[InterSatelliteObservation],
    *,
    timestamps: Array,
) -> InterSatelliteObservationAdapterResult:
    """Convert inter-satellite observations to modality-keyed histories."""

    times = np.asarray(timestamps, dtype=float).reshape(-1)
    if times.size == 0:
        raise ValueError("timestamps cannot be empty.")
    timestamp_to_index = {float(value): index for index, value in enumerate(times)}
    result: dict[str, dict[str, dict[str, Array]]] = {}
    covariance_by_modality: dict[str, Array] = {}

    for observation in observations:
        raw_modality = str(observation.modality).upper()
        if raw_modality not in _MODALITY_ALIASES:
            raise ValueError(f"Unsupported inter-satellite modality: {observation.modality}")
        modality = _MODALITY_ALIASES[raw_modality]
        timestamp = float(observation.timestamp)
        if timestamp not in timestamp_to_index:
            raise ValueError(f"Observation timestamp {timestamp} is not in runtime timestamps.")
        measurement = np.asarray(observation.measurement, dtype=float).reshape(-1)
        covariance = np.asarray(observation.covariance, dtype=float)
        dimension = 2 if modality == "AZ_EL" else 1
        if measurement.shape != (dimension,) or covariance.shape != (dimension, dimension):
            raise ValueError(
                f"{modality} observation has incompatible measurement/covariance dimensions."
            )
        confidence = float(np.clip(observation.confidence, 1e-6, 1.0))
        effective_covariance = covariance / confidence
        if modality not in covariance_by_modality:
            covariance_by_modality[modality] = effective_covariance
        elif not np.allclose(
            covariance_by_modality[modality],
            effective_covariance,
            rtol=1e-8,
            atol=1e-12,
        ):
            raise ValueError("Current adapter requires one constant covariance per modality.")

        source = str(observation.source_node_id)
        target = str(observation.target_node_id)
        values = (
            result
            .setdefault(source, {})
            .setdefault(target, {})
            .setdefault(
                modality,
                np.full((times.size, dimension), np.nan, dtype=float)
                if dimension > 1
                else np.full(times.size, np.nan, dtype=float),
            )
        )
        index = timestamp_to_index[timestamp]
        if np.all(np.isfinite(values[index])):
            raise ValueError(
                f"Duplicate {modality} observation for {source}->{target} at {timestamp}."
            )
        if observation.valid_flag:
            values[index] = measurement if dimension > 1 else float(measurement[0])

    if not covariance_by_modality:
        raise ValueError("At least one inter-satellite observation is required.")
    return InterSatelliteObservationAdapterResult(
        measurements_by_node=result,
        covariance_by_modality=covariance_by_modality,
    )
