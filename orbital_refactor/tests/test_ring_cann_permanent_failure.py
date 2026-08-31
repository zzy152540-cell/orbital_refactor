import numpy as np
import pytest

from brain_inspired.ring_cann import RingCANN
from experiments.ring_cann_permanent_failure_benchmark import (
    run_ring_cann_permanent_failure_benchmark,
)


def test_neuron_failure_mask_is_persistent_during_integration():
    cann = RingCANN()
    cann.reset(np.deg2rad(120.0))
    mask = np.zeros(180, dtype=bool)
    mask[::5] = True
    cann.set_neuron_failure_mask(mask)
    output = cann.step(0.0, 0.1)
    assert np.all(output.neural_activity[mask] == 0.0)
    assert np.any(output.neural_activity[~mask] > 0.0)
    assert output.valid


def test_neuron_failure_mask_requires_live_neuron_and_correct_shape():
    cann = RingCANN()
    cann.reset(0.0)
    with pytest.raises(ValueError, match="wrong ring dimension"):
        cann.set_neuron_failure_mask(np.zeros(10, dtype=bool))
    with pytest.raises(ValueError, match="remain available"):
        cann.set_neuron_failure_mask(np.ones(180, dtype=bool))


def test_short_permanent_failure_benchmark_remains_numerically_valid():
    traces = run_ring_cann_permanent_failure_benchmark(
        duration=0.2, sample_dt=0.1, seed=3,
    )
    assert len(traces) == 5
    for trace in traces.values():
        assert np.all(trace.valid)
        assert np.all(np.isfinite(trace.phase_error_deg))
        assert np.all(trace.concentration > 0.0)
