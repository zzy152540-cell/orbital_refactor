from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from adapters.module_input_adapter import adapt_module_input_centralized
from exporters.result_exporter import export_run_bundle
from interfaces.data_objects import InitialState, ModuleInput, Observation
from interfaces.state_awareness_module import StateAwarenessModule
from orbital_core.centralized_filter import CentralizedDynamicsEKF
from pipelines.centralized import run_centralized_filter


def _legacy_module():
    path = Path(__file__).parents[1] / "legacy" / "centralized_dynamics_fusion_ekf.py"
    spec = importlib.util.spec_from_file_location("legacy_centralized", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _runtime(count: int = 5):
    timestamps = np.arange(count, dtype=float)
    chief = np.tile(np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0]), (count, 1))
    quaternion = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (count, 1))
    state = np.array([100.0, 20.0, 1000.0, 0.1, -0.02, 0.0])
    covariance = np.diag([10.0, 10.0, 10.0, 0.1, 0.1, 0.1])
    return timestamps, chief, quaternion, state, covariance


def test_centralized_filter_matches_legacy_for_opt_ir_rad():
    legacy = _legacy_module()
    timestamps, chief, quaternion, state, covariance = _runtime()
    process_noise = np.eye(6) * 1e-8
    covariances = {
        "opt": np.diag([1e-6, 1e-6]),
        "ir": np.diag([1e-6, 1e-6]),
        "rad": np.diag([4.0, 0.04]),
    }
    measurements = {name: [] for name in covariances}
    for index in range(len(timestamps)):
        measurements["opt"].append(legacy.h_optical_spri(state, quaternion[index]))
        measurements["ir"].append(legacy.h_ir_spri(state, quaternion[index]))
        measurements["rad"].append(legacy.h_radar_spri(state, quaternion[index]))
    measurements = {name: np.asarray(values) for name, values in measurements.items()}
    flags = {name: np.ones(len(timestamps), dtype=bool) for name in measurements}

    old_filter = legacy.CentralizedDynamicsEKF(process_noise, covariances)
    old = legacy.run_centralized_filter(
        timestamps, np.tile(state, (len(timestamps), 1)), chief, quaternion,
        measurements["opt"], flags["opt"], measurements["ir"], flags["ir"],
        measurements["rad"], flags["rad"], old_filter, state, covariance,
    )
    new_filter = CentralizedDynamicsEKF(
        process_noise=process_noise, measurement_covariances=covariances
    )
    new = run_centralized_filter(
        timestamps=timestamps, chief_state_history_eci=chief,
        q_eci2pri_history=quaternion, measurements_by_modality=measurements,
        valid_flags_by_modality=flags, ekf=new_filter,
        initial_state=state, initial_covariance=covariance,
    )
    np.testing.assert_allclose(new.state_history, old["x_hat"], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(new.covariance_history, old["P_hist"], rtol=1e-6, atol=1e-7)


def test_standard_interface_selects_centralized_and_exports(tmp_path: Path):
    timestamps, chief, quaternion, state, covariance = _runtime(4)
    observations = []
    for index, timestamp in enumerate(timestamps):
        observations.append(Observation(
            timestamp=float(timestamp), observer_id="sat_1", target_id="target_1",
            modality="RADAR", source_type="TRADITIONAL",
            measurement=np.array([1005.18655, 0.00955]),
            covariance=np.diag([4.0, 0.04]), confidence=1.0,
            frame="SPRI", valid_flag=index != 2,
        ))
    module_input = ModuleInput(
        initial_state=InitialState("target_1", 0.0, state, covariance),
        sensor_measurements=observations,
        config={
            "runtime": {"timestamps": timestamps, "chief_state_history_eci": chief,
                        "q_eci2pri_history": quaternion, "node_id": "sat_1"},
            "filter": {"architecture": "centralized", "process_noise": np.eye(6) * 1e-8},
        },
    )
    output = StateAwarenessModule().run(module_input)
    assert output.state_output.position_estimate.shape == (3,)
    assert any(event.event_type == "MODALITY_MISSING" for event in output.abnormal_events)

    adapted = adapt_module_input_centralized(module_input)
    history = run_centralized_filter(
        timestamps=adapted.timestamps, chief_state_history_eci=adapted.chief_state_history_eci,
        q_eci2pri_history=adapted.q_eci2pri_history,
        measurements_by_modality=adapted.measurements_by_modality,
        valid_flags_by_modality=adapted.valid_flags_by_modality,
        ekf=adapted.centralized_filter, initial_state=adapted.initial_state,
        initial_covariance=adapted.initial_covariance,
        node_id=adapted.node_id, target_id=adapted.target_id,
    )
    paths = export_run_bundle(history, output, tmp_path, "centralized")
    assert all(path.exists() for path in paths.values())
    with paths["json"].open(encoding="utf-8") as file:
        payload = json.load(file)
    assert payload["state_output"]["target_id"] == "target_1"
    with np.load(paths["npz"]) as payload_npz:
        assert payload_npz["state_estimate"].shape == (4, 6)
