import copy

import numpy as np

from adapters.synthetic_measurement_adapter import create_nn_state_observations
from cooperative.multi_sat_pipeline import build_module_inputs
from interfaces.state_awareness_module import StateAwarenessModule
from orbital_core.constants import R_EARTH
from orbital_core.orbit_elements import keplerian_to_eci
from scenarios.multi_satellite_scenario import generate_cooperative_scenario


def _module_input(*, architecture: str = "federated_ci"):
    timestamps = np.arange(0.0, 31.0, 10.0)
    target = keplerian_to_eci(R_EARTH + 700e3, 0.001, 0.3, 0.0, 0.0, 0.0)
    observer = keplerian_to_eci(
        R_EARTH + 700e3, 0.001, 0.3, 0.0, 0.0, 0.0005,
    )
    scenario = generate_cooperative_scenario(
        timestamps=timestamps,
        target_id="target",
        target_initial_state_eci=target,
        observer_initial_states_eci={"sat_01": observer},
    )
    observations = create_nn_state_observations(
        timestamps=timestamps,
        relative_state_eci=scenario.relative_state_eci_by_node["sat_01"],
        covariance=np.diag([10., 10., 10., 0.05, 0.05, 0.05]) ** 2,
        observer_id="sat_01",
        target_id="target",
        rng=np.random.default_rng(4),
        include_velocity=True,
    )
    module_input = build_module_inputs(
        scenario=scenario,
        observations_by_node={"sat_01": observations},
        modality_config_by_node={
            "sat_01": {"nn": {
                "nn_meas_frame": "eci", "nn_use_pseudo_velocity": True,
            }},
        },
    )["sat_01"]
    module_input.config["filter"]["architecture"] = architecture
    return module_input


def test_passive_cann_sidecar_does_not_change_single_satellite_filter():
    baseline_input = _module_input()
    enabled_input = copy.deepcopy(baseline_input)
    enabled_input.config["brain_inspired"] = {
        "cann": {"enabled": True, "cue_interval_samples": 2},
    }

    baseline = StateAwarenessModule().run_history(baseline_input)
    enabled = StateAwarenessModule().run_history(enabled_input)

    assert baseline.cann_sidecar_history is None
    assert enabled.cann_sidecar_history is not None
    np.testing.assert_array_equal(
        enabled.fused_state_history, baseline.fused_state_history,
    )
    np.testing.assert_array_equal(
        enabled.fused_covariance_history, baseline.fused_covariance_history,
    )
    assert enabled.cann_sidecar_history.valid.all()
    assert enabled.cann_sidecar_history.source_id == "sat_01:target:fused"
    np.testing.assert_array_equal(
        enabled.cann_sidecar_history.cue_applied,
        np.array([False, False, True, False]),
    )


def test_centralized_single_satellite_can_enable_same_passive_sidecar():
    module_input = _module_input(architecture="centralized")
    module_input.config["brain_inspired"] = {
        "cann": {"enabled": True, "cue_interval_samples": 1},
    }

    history = StateAwarenessModule().run_history(module_input)

    assert history.cann_sidecar_history is not None
    assert history.cann_sidecar_history.valid.all()
    np.testing.assert_array_equal(
        history.cann_sidecar_history.cue_applied,
        np.array([False, True, True, True]),
    )
