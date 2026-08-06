from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from adapters.module_input_adapter import adapt_module_input
from interfaces.data_objects import InitialState, ModuleInput, Observation
from interfaces.state_awareness_module import StateAwarenessModule
from orbital_core.dynamics import make_process_noise
from orbital_core.filters import LocalDynamicsEKF


ROOT = Path(__file__).resolve().parents[1]


def _load_legacy_nn_module():
    path = ROOT / "legacy" / "federated_ci3_nn_ir_rad_fusion_ekf.py"
    spec = importlib.util.spec_from_file_location("legacy_nn_federated", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_nn_filter_matches_legacy_position_velocity_update():
    legacy = _load_legacy_nn_module()
    q_process = make_process_noise(1.0, 1e-4)
    r_measurement = np.diag([4.0, 9.0, 16.0, 0.01, 0.02, 0.03])
    new = LocalDynamicsEKF(
        q_process,
        r_measurement,
        "nn",
        nn_meas_frame="spri",
        nn_use_pseudo_velocity=True,
    )
    old = legacy.LocalDynamicsEKF(
        q_process,
        r_measurement,
        "nn",
        nn_meas_frame="spri",
        nn_use_pseudo_velocity=True,
    )
    state = np.array([120.0, -80.0, 350.0, 0.2, -0.1, 0.05])
    covariance = np.diag([25.0, 25.0, 25.0, 0.04, 0.04, 0.04])
    chief = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    predicted_new = new.predict(state, covariance, chief, 1.0)
    predicted_old = old.predict(state, covariance, chief, 1.0)
    np.testing.assert_allclose(predicted_new[0], predicted_old[0], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(predicted_new[1], predicted_old[1], rtol=1e-8, atol=1e-8)

    measurement = predicted_new[0] + np.array([1.0, -2.0, 0.5, 0.01, -0.02, 0.005])
    state_new, covariance_new, diagnostics = new.update(
        predicted_new[0], predicted_new[1], measurement, quaternion
    )
    state_old, covariance_old, nis_old, gated_old, skipped_old = old.update(
        predicted_old[0], predicted_old[1], measurement, quaternion
    )
    np.testing.assert_allclose(state_new, state_old, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(covariance_new, covariance_old, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(diagnostics.nis, nis_old, rtol=1e-8, atol=1e-10)
    assert diagnostics.gated == gated_old
    assert diagnostics.skipped == skipped_old


def _make_module_input() -> ModuleInput:
    timestamps = np.array([0.0, 1.0, 2.0])
    chief = np.array(
        [
            [7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0],
            [7.0e6, 7500.0, 0.0, -8.0, 7500.0, 0.0],
            [7.0e6 - 8.0, 15000.0, 0.0, -16.0, 7499.99, 0.0],
        ]
    )
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (3, 1))
    initial = InitialState(
        target_id="target_01",
        timestamp=0.0,
        state_estimate=np.array([100.0, 50.0, 500.0, 0.1, -0.05, 0.0]),
        covariance=np.diag([100.0, 100.0, 100.0, 0.1, 0.1, 0.1]),
    )
    observations = []
    for index, timestamp in enumerate(timestamps):
        observations.append(
            Observation(
                timestamp=float(timestamp),
                observer_id="sat_01",
                target_id="target_01",
                modality="OPTICAL",
                source_type="LEARNING",
                measurement=np.array([100.0 + 0.1 * index, 50.0 - 0.05 * index, 500.0]),
                covariance=np.diag([9.0, 9.0, 16.0]),
                confidence=0.9,
                frame="ECI",
                valid_flag=index != 1,
            )
        )
        observations.append(
            Observation(
                timestamp=float(timestamp),
                observer_id="sat_01",
                target_id="target_01",
                modality="RADAR",
                source_type="TRADITIONAL",
                measurement=np.array([512.35, 0.02]),
                covariance=np.diag([25.0, 0.04]),
                confidence=1.0,
                frame="SPRI",
                valid_flag=True,
            )
        )
    return ModuleInput(
        initial_state=initial,
        sensor_measurements=observations,
        config={
            "runtime": {
                "timestamps": timestamps,
                "chief_state_history_eci": chief,
                "q_eci2pri_history": quaternions,
                "node_id": "sat_01",
            },
            "filter": {
                "process_noise": make_process_noise(1.0, 1e-4),
                "reset_feedback": True,
                "ci_objective": "trace",
                "ci_grid_points": 11,
            },
            "modalities": {
                "nn": {"nn_meas_frame": "eci", "nn_use_pseudo_velocity": False},
                "radar": {},
            },
        },
    )


def test_module_input_adapter_preserves_documented_objects():
    adapted = adapt_module_input(_make_module_input())
    assert set(adapted.measurements_by_modality) == {"nn", "rad"}
    assert adapted.measurements_by_modality["nn"].shape == (3, 3)
    assert adapted.measurements_by_modality["rad"].shape == (3, 2)
    assert adapted.valid_flags_by_modality["nn"].tolist() == [True, False, True]
    assert adapted.local_filters["nn"].mode_name == "nn"
    assert adapted.node_id == "sat_01"


def test_state_awareness_module_returns_standard_output():
    output = StateAwarenessModule().run(_make_module_input())
    assert output.state_output.target_id == "target_01"
    assert output.state_output.position_estimate.shape == (3,)
    assert output.state_output.velocity_estimate.shape == (3,)
    assert output.state_output.acceleration_estimate.shape == (3,)
    assert output.state_output.covariance.shape == (6, 6)
    assert output.fusion_status.active_nodes == ["sat_01"]
    assert set(output.fusion_status.modality_valid_flags) == {"nn", "rad"}
    assert any(event.event_type == "MODALITY_MISSING" for event in output.abnormal_events)
