from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from orbital_core.dynamics import make_process_noise
from orbital_core.filters import LocalDynamicsEKF
from pipelines.federated_ci import run_federated_ci_filter


ROOT = Path(__file__).resolve().parents[1]
LEGACY_PATH = ROOT / "legacy" / "federated_ci_dynamics_fusion_ekf.py"


def _load_legacy_module():
    spec = importlib.util.spec_from_file_location("legacy_federated", LEGACY_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _synthetic_inputs(sample_count: int = 5):
    timestamps = np.arange(sample_count, dtype=float)
    chief = np.tile(np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0]), (sample_count, 1))
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (sample_count, 1))
    x0 = np.array([100.0, 60.0, 1000.0, 0.1, -0.05, 0.02])
    p0 = np.diag([25.0, 25.0, 25.0, 0.01, 0.01, 0.01])

    z_opt = np.tile(np.array([0.1, 0.06]), (sample_count, 1))
    z_ir = np.tile(np.array([np.arctan2(60.0, 100.0), np.arctan2(1000.0, np.hypot(100.0, 60.0))]), (sample_count, 1))
    rho = np.linalg.norm(x0[:3])
    rhodot = float(x0[:3] @ x0[3:] / rho)
    z_rad = np.tile(np.array([rho, rhodot]), (sample_count, 1))
    valid = np.ones(sample_count, dtype=bool)
    return timestamps, chief, quaternions, x0, p0, z_opt, z_ir, z_rad, valid


def _new_filters(dt: float = 1.0):
    q = make_process_noise(dt, 1e-6)
    return {
        "opt": LocalDynamicsEKF(q, np.diag([1e-6, 1e-6]), "opt"),
        "ir": LocalDynamicsEKF(q, np.diag([1e-6, 1e-6]), "ir"),
        "rad": LocalDynamicsEKF(q, np.diag([1.0, 1e-4]), "rad"),
    }


def _legacy_filters(legacy, dt: float = 1.0):
    q = legacy.make_process_noise(dt, 1e-6)
    return {
        "opt": legacy.LocalDynamicsEKF(q, np.diag([1e-6, 1e-6]), "opt"),
        "ir": legacy.LocalDynamicsEKF(q, np.diag([1e-6, 1e-6]), "ir"),
        "rad": legacy.LocalDynamicsEKF(q, np.diag([1.0, 1e-4]), "rad"),
    }


def test_federated_pipeline_matches_legacy_without_dropout():
    legacy = _load_legacy_module()
    t, chief, qhist, x0, p0, zopt, zir, zrad, valid = _synthetic_inputs()
    truth = np.tile(x0, (len(t), 1))
    legacy_result = legacy.run_federated_ci_filter(
        t, truth, chief, qhist,
        zopt, valid, zir, valid, zrad, valid,
        _legacy_filters(legacy), x0, p0,
        reset_feedback=True, ci_objective_name="trace", ci_grid_points=11,
    )
    new_result = run_federated_ci_filter(
        timestamps=t,
        chief_state_history_eci=chief,
        q_eci2pri_history=qhist,
        measurements_by_modality={"opt": zopt, "ir": zir, "rad": zrad},
        valid_flags_by_modality={"opt": valid, "ir": valid, "rad": valid},
        local_filters=_new_filters(),
        initial_state=x0,
        initial_covariance=p0,
        reset_feedback=True,
        ci_objective="trace",
        ci_grid_points=11,
    )
    np.testing.assert_allclose(new_result.fused_state_history, legacy_result["x_fused"], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(new_result.fused_covariance_history, legacy_result["P_fused"], rtol=1e-6, atol=1e-7)
    for modality in ("opt", "ir", "rad"):
        np.testing.assert_allclose(new_result.local_state_history[modality], legacy_result["x_local"][modality], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(new_result.local_covariance_history[modality], legacy_result["P_local"][modality], rtol=1e-6, atol=1e-7)


def test_dropout_generates_events_and_valid_module_output():
    t, chief, qhist, x0, p0, zopt, zir, zrad, valid = _synthetic_inputs()
    valid_opt = valid.copy(); valid_opt[2:] = False
    result = run_federated_ci_filter(
        timestamps=t,
        chief_state_history_eci=chief,
        q_eci2pri_history=qhist,
        measurements_by_modality={"opt": zopt, "ir": zir, "rad": zrad},
        valid_flags_by_modality={"opt": valid_opt, "ir": valid, "rad": valid},
        local_filters=_new_filters(),
        initial_state=x0,
        initial_covariance=p0,
        reset_feedback=False,
        ci_grid_points=7,
        node_id="sat_01",
        target_id="target_01",
    )
    output = result.to_module_output(node_id="sat_01", target_id="target_01")
    assert output.state_output.valid_flag
    assert output.state_output.position_estimate.shape == (3,)
    assert output.state_output.velocity_estimate.shape == (3,)
    assert output.state_output.acceleration_estimate.shape == (3,)
    assert output.state_output.covariance.shape == (6, 6)
    assert any(event.event_type == "MODALITY_MISSING" for event in output.abnormal_events)
    assert output.fusion_status.modality_valid_flags["opt"] is False
    assert np.isclose(sum(output.fusion_status.modality_weights.values()), 1.0)
    assert result.integrity_status_history["opt"][2] == "PREDICTION_ONLY"
    assert result.measurement_covariance_scale_history["ir"].shape == t.shape
    assert result.consecutive_anomaly_history["rad"].shape == t.shape


def test_all_modalities_missing_propagates_previous_fused_state():
    t, chief, qhist, x0, p0, zopt, zir, zrad, valid = _synthetic_inputs()
    all_missing = valid.copy(); all_missing[2] = False
    result = run_federated_ci_filter(
        timestamps=t,
        chief_state_history_eci=chief,
        q_eci2pri_history=qhist,
        measurements_by_modality={"opt": zopt, "ir": zir, "rad": zrad},
        valid_flags_by_modality={"opt": all_missing, "ir": all_missing, "rad": all_missing},
        local_filters=_new_filters(),
        initial_state=x0,
        initial_covariance=p0,
        ci_grid_points=7,
    )
    filters = _new_filters()
    expected_state, expected_covariance = filters["opt"].predict(
        result.fused_state_history[1], result.fused_covariance_history[1],
        chief[1], float(t[2] - t[1]),
    )
    np.testing.assert_allclose(result.fused_state_history[2], expected_state)
    np.testing.assert_allclose(result.fused_covariance_history[2], expected_covariance)
    assert result.ci_weight_history[2] is None
    assert any(event.event_type == "ALL_MODALITIES_UNAVAILABLE" for event in result.abnormal_events)
