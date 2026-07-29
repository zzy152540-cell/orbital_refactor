import numpy as np

from adapters.synthetic_measurement_adapter import create_nn_state_observations
from cooperative.multi_node_ci import fuse_local_histories
from cooperative.multi_sat_pipeline import build_module_inputs, run_cooperative_pipeline
from orbital_core.constants import R_EARTH
from orbital_core.orbit_elements import keplerian_to_eci
from scenarios.multi_satellite_scenario import generate_cooperative_scenario


def _small_case():
    timestamps = np.arange(0.0, 31.0, 10.0)
    target = keplerian_to_eci(R_EARTH + 700e3, 0.001, 0.3, 0.0, 0.0, 0.0)
    observers = {
        "a": keplerian_to_eci(R_EARTH + 700e3, 0.001, 0.3, 0.0, 0.0, 0.0005),
        "b": keplerian_to_eci(R_EARTH + 701e3, 0.001, 0.3, 0.0002, 0.0, -0.0005),
    }
    scenario = generate_cooperative_scenario(
        timestamps=timestamps,
        target_id="target",
        target_initial_state_eci=target,
        observer_initial_states_eci=observers,
    )
    observations = {}
    for index, node_id in enumerate(observers):
        observations[node_id] = create_nn_state_observations(
            timestamps=timestamps,
            relative_state_eci=scenario.relative_state_eci_by_node[node_id],
            covariance=np.diag([10., 10., 10., 0.05, 0.05, 0.05]) ** 2,
            observer_id=node_id,
            target_id="target",
            rng=np.random.default_rng(index + 1),
            include_velocity=True,
        )
    modality_config = {
        node_id: {"nn": {"nn_meas_frame": "eci", "nn_use_pseudo_velocity": True}}
        for node_id in observers
    }
    return scenario, observations, modality_config


def test_build_module_inputs_keeps_independent_node_runtime_data():
    scenario, observations, modality_config = _small_case()
    inputs = build_module_inputs(
        scenario=scenario,
        observations_by_node=observations,
        modality_config_by_node=modality_config,
    )
    assert set(inputs) == {"a", "b"}
    assert inputs["a"].config["runtime"]["node_id"] == "a"
    assert inputs["b"].config["runtime"]["node_id"] == "b"
    assert not np.shares_memory(
        inputs["a"].config["runtime"]["chief_state_history_eci"],
        inputs["b"].config["runtime"]["chief_state_history_eci"],
    )


def test_end_to_end_cooperative_pipeline_runs_and_returns_metrics():
    scenario, observations, modality_config = _small_case()
    result = run_cooperative_pipeline(
        scenario=scenario,
        observations_by_node=observations,
        initial_error_by_node={"a": np.ones(6), "b": -np.ones(6)},
        modality_config_by_node=modality_config,
        ci_grid_points=11,
    )
    assert result.cooperative_history.state_history_eci.shape == (4, 6)
    assert result.cooperative_history.covariance_history.shape == (4, 6, 6)
    assert np.isfinite(result.metrics.cooperative_position_rmse)
    assert set(result.metrics.local_position_rmse) == {"a", "b"}


def test_complete_node_outage_holds_previous_cooperative_posterior():
    timestamps = np.array([0.0, 1.0])
    relative = {"a": np.zeros((2, 6)), "b": np.zeros((2, 6))}
    observer = {"a": np.ones((2, 6)), "b": np.ones((2, 6))}
    covariance = {name: np.tile(np.eye(6), (2, 1, 1)) for name in relative}
    validity = {"a": np.array([True, False]), "b": np.array([True, False])}
    result = fuse_local_histories(
        timestamps=timestamps,
        relative_state_history_by_node=relative,
        covariance_history_by_node=covariance,
        observer_state_history_by_node=observer,
        target_id="target",
        validity_history_by_node=validity,
    )
    np.testing.assert_allclose(result.state_history_eci[1], result.state_history_eci[0])
    assert result.node_weight_history[1] == {}
    assert result.active_node_history[1] == []
