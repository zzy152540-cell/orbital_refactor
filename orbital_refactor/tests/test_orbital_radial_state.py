import numpy as np
import pytest

from brain_inspired.orbital_phase_adapter import OrbitalPlaneFrame
from brain_inspired.orbital_radial_state import OrbitalRadialState
from orbital_core.dynamics import rk4_step_absolute


def _initial_state():
    return np.array([7_000_000.0, 0.0, 0.0, 0.0, 7_500.0, 100.0])


def test_radial_state_predicts_and_applies_only_trusted_anchor():
    initial = _initial_state()
    radial = OrbitalRadialState(
        node_id="sat", frame=OrbitalPlaneFrame.from_state_eci(initial),
    )
    first = radial.initialize_from_state(timestamp=0.0, state_eci=initial)
    propagated = rk4_step_absolute(initial, 2.0)
    predicted = radial.predict_from_state(
        timestamp=2.0, predicted_state_eci=propagated,
    )
    rejected = radial.anchor_from_state(
        timestamp=2.0, posterior_state_eci=propagated,
        confidence=1.0, trusted=False,
    )
    accepted = radial.anchor_from_state(
        timestamp=2.0, posterior_state_eci=propagated,
        confidence=0.8, trusted=True,
    )
    assert first.decoded_displacement == pytest.approx(0.0, abs=1e-8)
    assert predicted.valid
    assert not rejected.cue_applied
    assert accepted.cue_applied
    assert accepted.anchor_gain == pytest.approx(0.2)
    assert accepted.anchor_age == 0.0


def test_radial_state_reports_boundary_saturation():
    initial = _initial_state()
    radial = OrbitalRadialState(
        node_id="sat", frame=OrbitalPlaneFrame.from_state_eci(initial),
    )
    radial.initialize_from_state(timestamp=0.0, state_eci=initial)
    shifted = initial.copy()
    shifted[:3] *= 1.01
    output = radial.predict_from_state(
        timestamp=1.0, predicted_state_eci=shifted,
        radial_rate_bias=1.0e6,
    )
    assert output.saturated_at_boundary
    assert output.valid
