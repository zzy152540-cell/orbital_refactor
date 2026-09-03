import numpy as np

from adapters.synthetic_measurement_adapter import (
    create_infrared_observations,
    create_optical_observations,
    create_radar_observations,
)
from cooperative.multi_sat_pipeline import build_module_inputs, run_cooperative_pipeline
from experiments.multimodal_cann_preprocessor import (
    MultimodalCANNPreprocessor,
)
from orbital_core.constants import R_EARTH
from orbital_core.coordinates import state_history_eci_to_spri
from orbital_core.orbit_elements import keplerian_to_eci
from scenarios.multi_satellite_scenario import generate_cooperative_scenario


def _case():
    timestamps = np.arange(0.0, 10.0, 2.0)
    target = keplerian_to_eci(R_EARTH + 700e3, 0.001, 0.3, 0.0, 0.0, 0.0)
    observers = {
        "sat_a": keplerian_to_eci(
            R_EARTH + 701e3, 0.001, 0.3, 0.0, 0.0, 0.001,
        ),
        "sat_b": keplerian_to_eci(
            R_EARTH + 702e3, 0.001, 0.3, 0.0, 0.0, 0.002,
        ),
    }
    scenario = generate_cooperative_scenario(
        timestamps=timestamps, target_id="target",
        target_initial_state_eci=target,
        observer_initial_states_eci=observers,
    )
    streams = {}
    for offset, node_id in enumerate(observers):
        observer = scenario.observer_trajectories[node_id]
        relative = state_history_eci_to_spri(
            scenario.relative_state_eci_by_node[node_id],
            observer.q_eci2pri_history,
        )
        rng = np.random.default_rng(10 + offset)
        streams[node_id] = [
            *create_optical_observations(
                timestamps=timestamps, relative_position_spri=relative[:, :3],
                covariance=np.diag([2e-4, 2e-4]) ** 2,
                observer_id=node_id, target_id="target", rng=rng,
            ),
            *create_infrared_observations(
                timestamps=timestamps, relative_position_spri=relative[:, :3],
                covariance=np.diag(np.deg2rad([0.05, 0.05])) ** 2,
                observer_id=node_id, target_id="target", rng=rng,
            ),
            *create_radar_observations(
                timestamps=timestamps, relative_position_spri=relative[:, :3],
                relative_velocity_spri=relative[:, 3:],
                covariance=np.diag([30.0, 0.05]) ** 2,
                observer_id=node_id, target_id="target", rng=rng,
            ),
        ]
    return scenario, streams


def test_three_modal_cann_connects_to_two_nodes_without_shared_state():
    scenario, streams = _case()
    preprocessors = {
        "sat_a": MultimodalCANNPreprocessor(),
        "sat_b": MultimodalCANNPreprocessor(),
    }

    inputs = build_module_inputs(
        scenario=scenario, observations_by_node=streams,
        measurement_preprocessor_by_node=preprocessors,
    )

    assert preprocessors["sat_a"].process_count == 1
    assert preprocessors["sat_b"].process_count == 1
    for node_id, module_input in inputs.items():
        processed = module_input.sensor_measurements
        assert len(processed) == 15
        assert all(item.observer_id == node_id for item in processed)
        assert all(
            item.source_type == "CANN_PREPROCESSED"
            for item in processed if item.valid_flag
        )
        assert all(
            item.source_type != "CANN_PROPAGATED"
            for item in processed if not item.valid_flag
        )
        assert {item.modality.lower() for item in processed} == {
            "optical", "infrared", "radar",
        }
    assert all(item.source_type != "CANN_PREPROCESSED"
               for stream in streams.values() for item in stream)


def test_three_modal_cann_runs_through_unchanged_local_filters_and_ci():
    scenario, streams = _case()
    result = run_cooperative_pipeline(
        scenario=scenario, observations_by_node=streams,
        measurement_preprocessor_by_node={
            node_id: MultimodalCANNPreprocessor() for node_id in streams
        },
    )

    assert set(result.local_histories) == set(streams)
    assert np.isfinite(result.cooperative_history.state_history_eci).all()
    assert np.isfinite(result.cooperative_history.covariance_history).all()
    assert np.isfinite(result.metrics.cooperative_position_rmse)
