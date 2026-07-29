import importlib.util
from pathlib import Path

import numpy as np

from orbital_core.ci_fusion import ci_fuse_pair, ci_fuse_posteriors
from orbital_core.dynamics import make_process_noise
from orbital_core.filters import LocalDynamicsEKF


ROOT = Path(__file__).resolve().parents[1]


def _load_legacy_single_modal():
    path = ROOT / "legacy" / "single_modal_dynamics_integrated_final.py"
    spec = importlib.util.spec_from_file_location("legacy_single_modal", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_new_ekf_matches_legacy_predict_and_update():
    legacy = _load_legacy_single_modal()
    q = make_process_noise(1.0, 1e-4)
    r = np.diag([1e-6, 1e-6])
    kwargs = dict(
        gate_enable=True,
        gate_threshold=16.0,
        gate_mode="soft",
        soft_scale=20.0,
    )
    old = legacy.DynamicsEKF(q, r, mode_name="ir", **kwargs)
    new = LocalDynamicsEKF(q, r, mode_name="ir", **kwargs)

    x = np.array([120.0, -40.0, 30.0, 0.1, -0.05, 0.02])
    p = np.diag([100.0, 100.0, 100.0, 0.01, 0.01, 0.01])
    chief = np.array([7.0e6, 1.0e5, 2.0e5, -100.0, 7500.0, 30.0])
    q_eci2pri = np.array([1.0, 0.0, 0.0, 0.0])

    old_xp, old_pp = old.predict(x, p, chief, 1.0)
    new_xp, new_pp = new.predict(x, p, chief, 1.0)
    np.testing.assert_allclose(new_xp, old_xp, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(new_pp, old_pp, rtol=1e-12, atol=1e-12)

    z = legacy.h_ir_spri(old_xp, q_eci2pri) + np.array([1e-4, -2e-4])
    old_xu, old_pu, old_nis, old_gated = old.update(old_xp, old_pp, z, q_eci2pri)
    new_xu, new_pu, diag = new.update(new_xp, new_pp, z, q_eci2pri)
    np.testing.assert_allclose(new_xu, old_xu, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(new_pu, old_pu, rtol=1e-11, atol=1e-11)
    assert np.isclose(diag.nis, old_nis)
    assert diag.gated == old_gated


def test_ci_pair_returns_normalized_weights_and_psd_covariance():
    x1 = np.zeros(2)
    x2 = np.ones(2)
    p1 = np.diag([1.0, 4.0])
    p2 = np.diag([4.0, 1.0])
    x, p, w = ci_fuse_pair(x1, p1, x2, p2, grid_points=51)
    assert x.shape == (2,)
    assert 0.0 <= w <= 1.0
    assert np.allclose(p, p.T)
    assert np.min(np.linalg.eigvalsh(p)) >= -1e-12


def test_ci_posteriors_preserves_input_names():
    result = ci_fuse_posteriors([
        ("nn", np.zeros(2), np.eye(2)),
        ("ir", np.ones(2), 2.0 * np.eye(2)),
        ("rad", 2.0 * np.ones(2), 3.0 * np.eye(2)),
    ], grid_points=11)
    assert set(result.weights) == {"nn", "ir", "rad"}
    assert np.isclose(sum(result.weights.values()), 1.0)


def _reference_ci_fuse_three(
    state_1,
    covariance_1,
    state_2,
    covariance_2,
    state_3,
    covariance_3,
    *,
    objective="trace",
    grid_points=31,
):
    """Original scalar-loop implementation retained only for regression testing."""
    information = [
        np.linalg.pinv(covariance_1),
        np.linalg.pinv(covariance_2),
        np.linalg.pinv(covariance_3),
    ]
    states = [state_1, state_2, state_3]
    best_value = np.inf
    best = None
    grid = np.linspace(0.0, 1.0, int(grid_points))
    for weight_1 in grid:
        for weight_2 in grid:
            weight_3 = 1.0 - weight_1 - weight_2
            if weight_3 < 0.0:
                continue
            weights = np.array([weight_1, weight_2, weight_3], dtype=float)
            fused_information = sum(
                weight * info
                for weight, info in zip(weights, information, strict=True)
            )
            covariance = np.linalg.pinv(fused_information)
            value = (
                float(np.trace(covariance))
                if objective == "trace"
                else float(np.linalg.slogdet(covariance)[1])
            )
            if value < best_value:
                information_vector = sum(
                    weight * info @ state
                    for weight, info, state in zip(
                        weights, information, states, strict=True
                    )
                )
                state = covariance @ information_vector
                best_value = value
                best = (state, 0.5 * (covariance + covariance.T), weights)
    assert best is not None
    return best


def test_optimized_three_way_ci_matches_original_grid_search():
    from orbital_core.ci_fusion import ci_fuse_three

    rng = np.random.default_rng(20260720)
    states = [rng.normal(size=6) for _ in range(3)]
    covariances = []
    for _ in range(3):
        factor = rng.normal(size=(6, 6))
        covariances.append(factor @ factor.T + 0.5 * np.eye(6))

    reference = _reference_ci_fuse_three(
        states[0], covariances[0],
        states[1], covariances[1],
        states[2], covariances[2],
        objective="trace",
        grid_points=21,
    )
    optimized = ci_fuse_three(
        states[0], covariances[0],
        states[1], covariances[1],
        states[2], covariances[2],
        objective="trace",
        grid_points=21,
    )

    np.testing.assert_allclose(optimized[2], reference[2], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(optimized[0], reference[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(optimized[1], reference[1], rtol=1e-12, atol=1e-12)


def test_three_way_ci_avoids_candidate_wise_pinv(monkeypatch):
    from orbital_core.ci_fusion import ci_fuse_three

    original_pinv = np.linalg.pinv
    call_count = 0

    def counting_pinv(matrix, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_pinv(matrix, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "pinv", counting_pinv)

    ci_fuse_three(
        np.zeros(6), np.eye(6),
        np.ones(6), 2.0 * np.eye(6),
        2.0 * np.ones(6), 3.0 * np.eye(6),
        grid_points=41,
    )

    # Only the three fixed local covariance matrices use pinv. Candidate
    # information matrices are inverted in one fast batched operation.
    assert call_count == 3
