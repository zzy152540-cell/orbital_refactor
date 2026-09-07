import numpy as np
import pytest

from brain_inspired.orbital_direction_state import OrbitalDirectionState
from brain_inspired.orbital_phase_adapter import OrbitalPlaneFrame
from orbital_core.dynamics import rk4_step_absolute


def _initial_state():
    return np.array([7_000_000.0, 0.0, 0.0, 0.0, 7_546.0, 0.0])


def test_direction_state_predicts_then_applies_only_trusted_anchor():
    initial = _initial_state()
    direction = OrbitalDirectionState(
        node_id="sat-01", frame=OrbitalPlaneFrame.from_state_eci(initial),
    )
    first = direction.initialize_from_state(timestamp=0.0, state_eci=initial)
    predicted_state = rk4_step_absolute(initial, 2.0)
    predicted = direction.predict_from_state(
        timestamp=2.0, predicted_state_eci=predicted_state,
    )
    rejected = direction.anchor_from_state(
        timestamp=2.0, posterior_state_eci=predicted_state,
        confidence=1.0, trusted=False,
    )
    accepted = direction.anchor_from_state(
        timestamp=2.0, posterior_state_eci=predicted_state,
        confidence=0.8, trusted=True,
    )

    assert first.anchor_age == 0.0
    assert predicted.anchor_age == pytest.approx(2.0)
    assert not rejected.cue_applied
    assert rejected.last_anchor_timestamp == 0.0
    assert accepted.cue_applied
    assert accepted.anchor_gain == pytest.approx(0.2)
    assert accepted.last_anchor_timestamp == 2.0
    assert accepted.anchor_age == 0.0


def test_direction_instances_do_not_share_neural_state():
    initial = _initial_state()
    frame = OrbitalPlaneFrame.from_state_eci(initial)
    left = OrbitalDirectionState(node_id="left", frame=frame)
    right = OrbitalDirectionState(node_id="right", frame=frame)
    left_output = left.initialize_from_state(timestamp=0.0, state_eci=initial)
    shifted = initial.copy()
    shifted[:3] = np.array([0.0, 7_000_000.0, 0.0])
    shifted[3:] = np.array([-7_546.0, 0.0, 0.0])
    right_output = right.initialize_from_state(
        timestamp=0.0, state_eci=shifted,
    )
    assert abs(left_output.decoded_phase - right_output.decoded_phase) > 1.0
    assert not np.shares_memory(
        left_output.neural_activity, right_output.neural_activity,
    )


def test_direction_anchor_requires_current_timestamp():
    initial = _initial_state()
    direction = OrbitalDirectionState(
        node_id="sat-01", frame=OrbitalPlaneFrame.from_state_eci(initial),
    )
    direction.initialize_from_state(timestamp=0.0, state_eci=initial)
    with pytest.raises(ValueError, match="current direction timestamp"):
        direction.anchor_from_state(
            timestamp=2.0, posterior_state_eci=initial,
            confidence=1.0, trusted=True,
        )


def test_direction_state_accepts_controlled_phase_rate_bias():
    initial = _initial_state()
    frame = OrbitalPlaneFrame.from_state_eci(initial)
    nominal = OrbitalDirectionState(node_id="nominal", frame=frame)
    biased = OrbitalDirectionState(node_id="biased", frame=frame)
    nominal.initialize_from_state(timestamp=0.0, state_eci=initial)
    biased.initialize_from_state(timestamp=0.0, state_eci=initial)
    propagated = rk4_step_absolute(initial, 2.0)
    nominal_output = nominal.predict_from_state(
        timestamp=2.0, predicted_state_eci=propagated,
    )
    biased_output = biased.predict_from_state(
        timestamp=2.0, predicted_state_eci=propagated,
        phase_rate_bias=0.01,
    )
    assert biased_output.decoded_phase != pytest.approx(
        nominal_output.decoded_phase,
    )
