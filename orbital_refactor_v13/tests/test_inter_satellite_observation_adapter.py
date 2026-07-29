import numpy as np

from cooperative.inter_satellite_observation_adapter import (
    adapt_inter_satellite_observations,
    adapt_inter_satellite_range_observations,
)
from interfaces.data_objects import InterSatelliteObservation


def _range_observation(timestamp, source="sat_01", target="sat_02", value=10.0):
    return InterSatelliteObservation(
        timestamp=timestamp,
        source_node_id=source,
        target_node_id=target,
        modality="RANGE",
        measurement=np.array([value]),
        covariance=np.array([[4.0]]),
        confidence=1.0,
        valid_flag=True,
    )


def test_adapt_inter_satellite_range_observations_groups_by_node_pair():
    result = adapt_inter_satellite_range_observations(
        [
            _range_observation(0.0, value=10.0),
            _range_observation(1.0, value=11.0),
        ],
        timestamps=np.array([0.0, 1.0]),
    )

    np.testing.assert_allclose(
        result.range_measurements_by_node["sat_01"]["sat_02"],
        np.array([10.0, 11.0]),
    )
    assert result.range_variance == 4.0


def test_adapt_inter_satellite_range_observations_rejects_non_range():
    observation = _range_observation(0.0)
    observation.modality = "LOS"
    try:
        adapt_inter_satellite_range_observations(
            [observation],
            timestamps=np.array([0.0]),
        )
    except ValueError as exc:
        assert "Unsupported" in str(exc)
    else:
        raise AssertionError("Expected unsupported modality to be rejected.")


def test_adapt_inter_satellite_observations_supports_range_rate():
    observations = [
        _range_observation(0.0, value=10.0),
        InterSatelliteObservation(
            timestamp=0.0,
            source_node_id="sat_01",
            target_node_id="sat_02",
            modality="RANGE_RATE",
            measurement=np.array([0.2]),
            covariance=np.array([[0.01]]),
            confidence=1.0,
            valid_flag=True,
        ),
    ]

    result = adapt_inter_satellite_observations(
        observations,
        timestamps=np.array([0.0]),
    )

    assert set(result.variance_by_modality) == {"RANGE", "RANGE_RATE"}
    assert result.measurements_by_node["sat_01"]["sat_02"]["RANGE"][0] == 10.0
    assert result.measurements_by_node["sat_01"]["sat_02"]["RANGE_RATE"][0] == 0.2


def test_adapt_inter_satellite_observations_supports_az_el():
    observation = InterSatelliteObservation(
        timestamp=0.0,
        source_node_id="sat_01",
        target_node_id="sat_02",
        modality="AZ_EL",
        measurement=np.array([0.1, -0.2]),
        covariance=np.diag([0.01, 0.02]),
        confidence=1.0,
        valid_flag=True,
    )

    result = adapt_inter_satellite_observations(
        [observation],
        timestamps=np.array([0.0]),
    )

    np.testing.assert_allclose(
        result.measurements_by_node["sat_01"]["sat_02"]["AZ_EL"][0],
        np.array([0.1, -0.2]),
    )
    np.testing.assert_allclose(
        result.covariance_by_modality["AZ_EL"],
        np.diag([0.01, 0.02]),
    )
