import numpy as np
import pytest

from brain_inspired.orbital_rt_adapter import extract_orbital_rt_offset


def test_rt_offset_includes_reference_frame_rotation():
    reference = np.array([7e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    state = reference.copy()
    state[0] += 100.0
    state[1] += 200.0
    offset = extract_orbital_rt_offset(
        timestamp=0.0, state_eci=state, reference_state_eci=reference,
    )
    omega = 7500.0 / 7e6
    assert offset.radial_position == pytest.approx(100.0)
    assert offset.along_track_position == pytest.approx(200.0)
    assert offset.radial_rate == pytest.approx(omega * 200.0)
    assert offset.along_track_rate == pytest.approx(-omega * 100.0)


def test_rt_offset_is_zero_for_reference_state():
    reference = np.array([7e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    offset = extract_orbital_rt_offset(
        timestamp=0.0, state_eci=reference, reference_state_eci=reference,
    )
    assert np.allclose([
        offset.radial_position, offset.along_track_position,
        offset.radial_rate, offset.along_track_rate,
    ], 0.0)
