import numpy as np

from brain_inspired import build_multitarget_brain_sidecar
from orbital_core.dynamics import propagate_absolute_orbit
from orbital_core.orbit_elements import keplerian_to_eci


def _history(anomaly_deg):
    initial = keplerian_to_eci(
        7_078_137.0, 0.001, np.deg2rad(55.0), np.deg2rad(15.0),
        0.0, np.deg2rad(anomaly_deg),
    )
    return propagate_absolute_orbit(initial, np.array([0.0, 10.0, 20.0]))


def test_multitarget_brain_sidecar_is_keyed_by_global_target_identity():
    sidecar = build_multitarget_brain_sidecar(
        timestamps=np.array([10.0, 20.0, 30.0]),
        state_history_by_target={
            "cooperative-target-0001": _history(1.0),
            "cooperative-target-0002": _history(5.0),
        },
    )

    first = sidecar.snapshot("cooperative-target-0001", -1)
    second = sidecar.snapshot("cooperative-target-0002", -1)

    assert first.valid and second.valid
    assert first.direction_ring.shape == (180,)
    assert first.place_heatmap_3x3.shape == (3, 3)
    assert np.isclose(np.sum(first.place_heatmap_3x3), 1.0)
    assert not np.allclose(first.direction_ring, second.direction_ring)
    assert set(sidecar.navigation_by_target) == {
        "cooperative-target-0001", "cooperative-target-0002",
    }


def test_multitarget_brain_sidecar_rejects_misaligned_history():
    try:
        build_multitarget_brain_sidecar(
            timestamps=np.array([0.0, 10.0]),
            state_history_by_target={"target": _history(1.0)},
        )
    except ValueError as error:
        assert "shape (N, 6)" in str(error)
    else:
        raise AssertionError("Expected history-length validation failure.")
