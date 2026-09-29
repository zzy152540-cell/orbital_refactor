import numpy as np

from adapters.infrared_image_adapter import InfraredCameraConfig
from adapters.multimodal_sensor_simulator import MultimodalSensorSimulationConfig
from adapters.optical_image_adapter import OpticalCameraConfig
from adapters.radar_range_doppler_adapter import RadarRangeDopplerConfig
from orbital_core.constants import R_EARTH
from orbital_core.orbit_elements import keplerian_to_eci
from tracking import (
    TrackLifecycle,
    build_standard_multitarget_scenario,
    run_standard_multitarget_validation,
)


def _state(anomaly_deg, inclination_deg=55.0, altitude_m=700e3):
    return keplerian_to_eci(
        R_EARTH + altitude_m,
        0.001,
        np.deg2rad(inclination_deg),
        np.deg2rad(15.0),
        0.0,
        np.deg2rad(anomaly_deg),
    )


def _small_sensors():
    common = dict(width=33, height=33, read_noise_sigma=0.0)
    return MultimodalSensorSimulationConfig(
        optical=OpticalCameraConfig(**common),
        infrared=InfraredCameraConfig(**common),
        radar=RadarRangeDopplerConfig(**common),
    )


def test_three_observer_two_target_raw_scenario_runs_and_reports_each_target():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    observers = {
        "observer-1": _state(0.0, 54.8, 699e3),
        "observer-2": _state(0.2, 54.9, 701e3),
        "observer-3": _state(-0.2, 54.7, 698e3),
    }
    targets = {
        "target-1": _state(0.6, 55.2, 703e3),
        "target-2": _state(-0.7, 55.3, 704e3),
    }
    outage = {
        (observer, "target-2", modality): ((1.0, 1.0),)
        for observer in observers
        for modality in ("RADAR", "INFRARED", "OPTICAL")
    }
    scenario = build_standard_multitarget_scenario(
        scene_id="three-by-two",
        timestamps=times,
        observer_initial_states_eci=observers,
        target_initial_states_eci=targets,
        initial_error_by_link={
            (observer, target): np.array([20.0, -10.0, 15.0, 0.02, -0.01, 0.01])
            for observer in observers for target in targets
        },
        sensor_config=_small_sensors(),
        dropout_windows=outage,
        random_seed=7,
    )

    assert len(scenario.module_inputs) == 6
    for task in scenario.module_inputs:
        assert {item.modality for item in task.sensor_measurements} == {
            "RADAR", "INFRARED", "OPTICAL",
        }

    result = run_standard_multitarget_validation(scenario, max_coast_epochs=2)

    assert set(result.metrics_by_target) == {"target-1", "target-2"}
    assert result.history.track_history_by_target["target-2"][1].lifecycle is TrackLifecycle.COASTING
    assert result.metrics_by_target["target-2"].reacquisition_count == 1
    for metrics in result.metrics_by_target.values():
        assert np.isfinite(metrics.position_rmse_m)
        assert np.isfinite(metrics.velocity_rmse_mps)
        assert np.isfinite(metrics.mean_nees)
        assert metrics.minimum_active_observers in {0, 3}


def test_standard_scenario_rejects_unknown_dropout_modality():
    with np.testing.assert_raises_regex(ValueError, "Unsupported dropout modality"):
        build_standard_multitarget_scenario(
            scene_id="bad-dropout",
            timestamps=np.array([0.0, 1.0]),
            observer_initial_states_eci={"observer": _state(0.0)},
            target_initial_states_eci={"target": _state(0.5)},
            dropout_windows={
                ("observer", "target", "LIDAR"): ((0.0, 1.0),),
            },
        )
