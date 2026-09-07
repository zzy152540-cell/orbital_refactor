import numpy as np

from brain_inspired.orbital_rt_multiscale_state import OrbitalRTMultiScaleState
from orbital_core.dynamics import rk4_step_absolute


def _state():
    return np.array([7e6, 0.0, 0.0, 0.0, 7500.0, 0.0])


def test_multiscale_rt_state_represents_large_radial_offset():
    reference = _state()
    state = reference.copy()
    state[0] += 15_123.0
    output = OrbitalRTMultiScaleState(node_id="sat").initialize(
        timestamp=0.0, state_eci=state, reference_state_eci=reference,
    )
    assert output.valid
    assert abs(output.decoded_rt[0] - 15_123.0) < 1e-5
    assert abs(output.fine_residual_rt[0]) <= 625.0


def test_multiscale_rt_state_predicts_and_rejects_large_anchor():
    reference = _state()
    tracker = OrbitalRTMultiScaleState(node_id="sat")
    tracker.initialize(timestamp=0.0, state_eci=reference,
                       reference_state_eci=reference)
    next_reference = rk4_step_absolute(reference, 2.0)
    tracker.predict(timestamp=2.0, predicted_state_eci=next_reference,
                    reference_state_eci=next_reference)
    outlier = next_reference.copy()
    outlier[0] += 500.0
    output = tracker.anchor(
        timestamp=2.0, posterior_state_eci=outlier,
        reference_state_eci=next_reference, confidence=1.0, trusted=True,
    )
    assert output.anchor_rejected
    assert not output.cue_applied
