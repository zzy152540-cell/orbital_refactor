from copy import deepcopy

import numpy as np
import pytest

from adapters.infrared_image_adapter import InfraredCameraConfig
from adapters.multimodal_sensor_simulator import MultimodalSensorSimulationConfig
from adapters.optical_image_adapter import OpticalCameraConfig
from adapters.radar_range_doppler_adapter import RadarRangeDopplerConfig
from tracking import (
    adapt_external_multitarget_input,
    build_standard_multitarget_scenario_from_input,
)


def _satellite(name, norad, anomaly):
    return {
        "assetName": name,
        "satelliteId": f"satellite-{norad}",
        "norad": str(norad),
        "semiMajorAxis": "7078.0",
        "eccentricity": "0.001",
        "inclination": "55.0",
        "raan": "15.0",
        "argPerigee": "0.0",
        "meanAnomaly": str(anomaly),
        "epoch": "26261.16666667",
        "tleLine1": "",
        "tleLine2": "",
    }


def _payload():
    satellites = [
        _satellite("observer-1", 90001, 0.0),
        _satellite("observer-2", 90002, 0.2),
        _satellite("target-1", 91001, 0.6),
        _satellite("target-2", 91002, -0.7),
    ]
    return {
        "sceneId": "external-multitarget",
        "name": "external multi-target example",
        "startTime": "2026-09-18 04:00:00",
        "endTime": "2026-09-18 04:00:02",
        "satelliteCount": len(satellites),
        "satellites": satellites,
        "estimationConfig": {
            "observerIds": ["observer-1", "observer-2"],
            "observerCovarianceDiagonal": [1, 1, 1, 0.01, 0.01, 0.01],
            "defaultTargetCovarianceDiagonal": [100, 100, 100, 0.2, 0.2, 0.2],
            "targets": [
                {
                    "nodeId": "target-1",
                    "trackId": "enemy-track-001",
                    "sensorAvailability": {
                        "radar": True, "infrared": True, "optical": True,
                    },
                },
                {
                    "nodeId": "target-2",
                    "covarianceDiagonal": [400, 400, 400, 1, 1, 1],
                },
            ],
        },
    }


def test_external_scene_assigns_observer_and_multiple_target_roles():
    result = adapt_external_multitarget_input(_payload())

    assert set(result.observer_states) == {"observer-1", "observer-2"}
    assert set(result.target_initial_states) == {"target-1", "target-2"}
    assert result.target_initial_states["target-1"].track_id == "enemy-track-001"
    np.testing.assert_allclose(
        np.diag(result.target_initial_states["target-2"].covariance_eci),
        [400, 400, 400, 1, 1, 1],
    )
    assert result.config["timestamps"].tolist() == [0.0, 1.0, 2.0]
    assert result.config["initialOrbitDeterminationPerformed"] is False
    assert result.config["sensorAvailabilityByTarget"]["target-1"]["RADAR"]


def test_external_scene_rejects_unknown_or_overlapping_roles():
    unknown = deepcopy(_payload())
    unknown["estimationConfig"]["observerIds"] = ["missing"]
    with pytest.raises(ValueError, match="unknown satellites"):
        adapt_external_multitarget_input(unknown)

    overlap = deepcopy(_payload())
    overlap["estimationConfig"]["observerIds"].append("target-1")
    with pytest.raises(ValueError, match="roles overlap"):
        adapt_external_multitarget_input(overlap)


def test_external_scene_requires_explicit_estimation_roles():
    payload = _payload()
    payload.pop("estimationConfig")
    with pytest.raises(ValueError, match="requires estimationConfig"):
        adapt_external_multitarget_input(payload)


def test_external_roles_bridge_to_all_observer_target_local_tasks():
    payload = _payload()
    payload["estimationConfig"]["targets"][1]["sensorAvailability"] = {
        "optical": False,
    }
    value = adapt_external_multitarget_input(payload)
    common = dict(width=17, height=17, read_noise_sigma=0.0)
    sensors = MultimodalSensorSimulationConfig(
        optical=OpticalCameraConfig(**common),
        infrared=InfraredCameraConfig(**common),
        radar=RadarRangeDopplerConfig(**common),
    )

    scenario = build_standard_multitarget_scenario_from_input(
        value, sensor_config=sensors, random_seed=4,
    )

    assert len(scenario.module_inputs) == 4
    assert scenario.initial_target_states["target-1"].track_id == "enemy-track-001"
    np.testing.assert_allclose(
        scenario.initial_target_states["target-2"].covariance_eci,
        value.target_initial_states["target-2"].covariance_eci,
    )
    target_2_tasks = [
        task for task in scenario.module_inputs
        if task.initial_state.target_id == "target-2"
    ]
    assert target_2_tasks
    assert all(
        not item.valid_flag
        for task in target_2_tasks
        for item in task.sensor_measurements
        if item.modality == "OPTICAL"
    )
