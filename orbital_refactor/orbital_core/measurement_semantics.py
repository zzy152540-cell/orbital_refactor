from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SensorMeasurementContract:
    """Physical sensor semantics independent of estimator message granularity."""

    sensor_modality: str
    measurement_type: str
    components: tuple[str, ...]
    dimension: int
    angular: bool


SINGLE_SATELLITE_SENSOR_CONTRACTS = {
    "OPTICAL": SensorMeasurementContract(
        "OPTICAL", "NORMALIZED_IMAGE_COORDINATES",
        ("IMAGE_U", "IMAGE_V"), 2, False,
    ),
    "INFRARED": SensorMeasurementContract(
        "INFRARED", "AZIMUTH_ELEVATION",
        ("AZIMUTH", "ELEVATION"), 2, True,
    ),
    "RADAR": SensorMeasurementContract(
        "RADAR", "RANGE_RANGE_RATE", ("RANGE", "RANGE_RATE"), 2, False,
    ),
}


_INTER_SATELLITE_COMPONENT_TO_SENSOR = {
    "RADAR": ("RADAR", "RANGE_RANGE_RATE"),
    "RANGE": ("RADAR", "RANGE"),
    "RANGE_RATE": ("RADAR", "RANGE_RATE"),
    "AZ_EL": ("INFRARED", "AZIMUTH_ELEVATION"),
    "INFRARED": ("INFRARED", "AZIMUTH_ELEVATION"),
    "OPTICAL": ("OPTICAL", "NORMALIZED_IMAGE_COORDINATES"),
}


def sensor_semantics_for_inter_satellite_component(
    component_modality: str,
) -> tuple[str, str]:
    """Map a legacy estimator component to its physical sensor semantics.

    Radar range and range-rate are transported as separate scalar messages by
    the current distributed estimator, but remain components of one RADAR
    modality. ``AZ_EL`` denotes the infrared angular model; traditional optical
    normalized image coordinates must not be silently aliased to it.
    """

    normalized = str(component_modality).upper()
    try:
        return _INTER_SATELLITE_COMPONENT_TO_SENSOR[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported inter-satellite measurement component: {component_modality}"
        ) from exc


def inter_satellite_semantic_metadata(component_modality: str) -> dict[str, str]:
    sensor, measurement_type = sensor_semantics_for_inter_satellite_component(
        component_modality
    )
    return {
        "sensor_modality": sensor,
        "measurement_type": measurement_type,
        "measurement_component": str(component_modality).upper(),
    }
