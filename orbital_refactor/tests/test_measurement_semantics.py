import pytest

from orbital_core.measurement_semantics import (
    SINGLE_SATELLITE_SENSOR_CONTRACTS,
    inter_satellite_semantic_metadata,
    sensor_semantics_for_inter_satellite_component,
)


def test_single_satellite_three_sensor_contract_is_explicit():
    contracts = SINGLE_SATELLITE_SENSOR_CONTRACTS

    assert set(contracts) == {"OPTICAL", "INFRARED", "RADAR"}
    assert contracts["OPTICAL"].measurement_type == "NORMALIZED_IMAGE_COORDINATES"
    assert contracts["OPTICAL"].components == ("IMAGE_U", "IMAGE_V")
    assert contracts["INFRARED"].measurement_type == "AZIMUTH_ELEVATION"
    assert contracts["RADAR"].components == ("RANGE", "RANGE_RATE")


@pytest.mark.parametrize(
    ("component", "sensor", "measurement_type"),
    (
        ("RADAR", "RADAR", "RANGE_RANGE_RATE"),
        ("RANGE", "RADAR", "RANGE"),
        ("RANGE_RATE", "RADAR", "RANGE_RATE"),
        ("AZ_EL", "INFRARED", "AZIMUTH_ELEVATION"),
        ("INFRARED", "INFRARED", "AZIMUTH_ELEVATION"),
        ("OPTICAL", "OPTICAL", "NORMALIZED_IMAGE_COORDINATES"),
    ),
)
def test_legacy_components_map_to_physical_sensor_semantics(
    component, sensor, measurement_type,
):
    assert sensor_semantics_for_inter_satellite_component(component) == (
        sensor, measurement_type,
    )
    assert inter_satellite_semantic_metadata(component) == {
        "sensor_modality": sensor,
        "measurement_type": measurement_type,
        "measurement_component": component,
    }


def test_az_el_is_not_silently_classified_as_traditional_optical():
    sensor, _ = sensor_semantics_for_inter_satellite_component("AZ_EL")
    assert sensor == "INFRARED"


def test_unknown_component_is_rejected():
    with pytest.raises(ValueError, match="Unsupported inter-satellite"):
        sensor_semantics_for_inter_satellite_component("LIDAR")
