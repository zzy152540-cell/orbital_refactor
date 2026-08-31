import numpy as np

from experiments.walker_cann_sidecar_comparison import (
    _circular_difference,
    _unwrap_aligned,
)


def test_unwrapped_phase_is_aligned_to_equivalent_reference_turn():
    phase = np.deg2rad([359.9, 0.1, 0.3])
    aligned = np.rad2deg(_unwrap_aligned(phase, np.deg2rad(0.0)))
    np.testing.assert_allclose(aligned, [-0.1, 0.1, 0.3], atol=1.0e-12)


def test_circular_difference_uses_shortest_ring_distance():
    difference = _circular_difference(np.deg2rad(359.0), np.deg2rad(1.0))
    assert np.isclose(np.rad2deg(difference), -2.0)
