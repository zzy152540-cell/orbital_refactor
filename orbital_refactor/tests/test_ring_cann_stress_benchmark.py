import numpy as np
import pytest

from brain_inspired.passive_phase_observer import PeriodicStateInput
from experiments.ring_cann_stress_benchmark import (
    _left_integral,
    run_ring_cann_stress_benchmark,
)


def test_left_integral_preserves_initial_phase_epoch():
    np.testing.assert_allclose(
        _left_integral(np.array([1.0, 2.0, 3.0]), 0.5),
        [0.0, 0.5, 1.5],
    )


def test_short_stress_benchmark_is_reproducible_and_aligned():
    first = run_ring_cann_stress_benchmark(
        duration=20.0, sample_dt=2.0, seed=4,
        outage_window=(8.0, 12.0), hint_interval=4.0,
    )
    second = run_ring_cann_stress_benchmark(
        duration=20.0, sample_dt=2.0, seed=4,
        outage_window=(8.0, 12.0), hint_interval=4.0,
    )
    assert first.timestamps.shape == first.truth_phase.shape
    assert set(first.phase_rmse_deg_by_mode) == {
        "dead_reckoning", "gated_complementary", "cann_no_cue",
        "cann_sparse_cue", "cann_gated_cue",
    }
    np.testing.assert_allclose(first.measured_phase_rate, second.measured_phase_rate)
    assert np.all(np.isfinite(tuple(first.phase_rmse_deg_by_mode.values())))


def test_periodic_input_rejects_invalid_cue_gain():
    with pytest.raises(ValueError, match="Cue gain"):
        PeriodicStateInput(
            timestamp=1.0, phase_rate=0.0, cue_gain=-0.1,
        ).validate()
