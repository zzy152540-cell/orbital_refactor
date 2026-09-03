import numpy as np
import pytest

from brain_inspired.cann_measurement_adapter import preprocess_observation
from cooperative.multi_sat_pipeline import build_module_inputs
from interfaces.data_objects import Observation
from orbital_core.constants import R_EARTH
from orbital_core.orbit_elements import keplerian_to_eci
from scenarios.multi_satellite_scenario import generate_cooperative_scenario


class RecordingCANNPreprocessor:
    def __init__(self):
        self.calls = 0

    def process(self, observations, timestamps):
        self.calls += 1
        assert len(timestamps) == 3
        return [
            preprocess_observation(
                item, measurement=item.measurement.copy(),
                diagnostics={"stream_call": self.calls},
            )
            for item in observations
        ]


def _case():
    timestamps = np.array([0.0, 2.0, 4.0])
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
    observations = {
        node_id: [
            Observation(
                timestamp=float(timestamp), observer_id=node_id,
                target_id="target", modality="RADAR", source_type="RADAR",
                measurement=np.array([1000.0 + index, 0.5]),
                covariance=np.diag([30.0, 0.05]) ** 2,
                confidence=1.0, frame="SPRI", valid_flag=True,
                metadata={"observation_id": f"{node_id}:{index}"},
            )
            for index, timestamp in enumerate(timestamps)
        ]
        for node_id in observers
    }
    return scenario, observations


def test_multi_satellite_inputs_accept_isolated_cann_preprocessors():
    scenario, observations = _case()
    processors = {
        "sat_a": RecordingCANNPreprocessor(),
        "sat_b": RecordingCANNPreprocessor(),
    }

    inputs = build_module_inputs(
        scenario=scenario, observations_by_node=observations,
        measurement_preprocessor_by_node=processors,
    )

    assert processors["sat_a"].calls == processors["sat_b"].calls == 1
    for node_id, module_input in inputs.items():
        assert all(item.observer_id == node_id for item in module_input.sensor_measurements)
        assert all(item.source_type == "CANN_PREPROCESSED"
                   for item in module_input.sensor_measurements)
        assert all(item.metadata["cann_diagnostics"]["stream_call"] == 1
                   for item in module_input.sensor_measurements)
    assert all(item.source_type == "RADAR"
               for stream in observations.values() for item in stream)


def test_multi_satellite_preprocessor_cannot_cross_node_boundary():
    scenario, observations = _case()

    class InvalidPreprocessor:
        def process(self, items, timestamps):
            del timestamps
            items[0].observer_id = "sat_b"
            return items

    with pytest.raises(ValueError, match="between observers"):
        build_module_inputs(
            scenario=scenario, observations_by_node=observations,
            measurement_preprocessor_by_node={"sat_a": InvalidPreprocessor()},
        )
