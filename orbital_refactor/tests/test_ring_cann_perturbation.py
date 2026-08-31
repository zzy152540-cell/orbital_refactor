import numpy as np
import pytest

from brain_inspired.ring_cann import RingCANN
from experiments.ring_cann_perturbation_benchmark import (
    run_ring_cann_perturbation_benchmark,
)


def test_transient_perturbation_requires_initialization_and_valid_shape():
    cann = RingCANN()
    with pytest.raises(RuntimeError, match="reset"):
        cann.apply_transient_perturbation(additive_input=np.zeros(180))
    cann.reset(0.0)
    with pytest.raises(ValueError, match="finite ring vector"):
        cann.apply_transient_perturbation(additive_input=np.zeros(10))


def test_transient_silencing_changes_activity_without_advancing_time():
    cann = RingCANN()
    baseline = cann.reset(np.deg2rad(120.0), timestamp=5.0)
    mask = np.zeros(180, dtype=bool)
    mask[55:66] = True
    perturbed = cann.apply_transient_perturbation(silenced_neuron_mask=mask)
    assert perturbed.timestamp == baseline.timestamp
    assert not np.allclose(perturbed.neural_activity, baseline.neural_activity)


def test_short_perturbation_benchmark_produces_finite_recovery_traces():
    traces = run_ring_cann_perturbation_benchmark(
        recovery_duration=0.1, sample_dt=0.02, seed=2,
    )
    assert len(traces) == 6
    for trace in traces.values():
        assert trace.timestamps.shape == trace.phase_error_deg.shape
        assert np.all(np.isfinite(trace.concentration))
        assert np.all(np.isfinite(trace.width))
