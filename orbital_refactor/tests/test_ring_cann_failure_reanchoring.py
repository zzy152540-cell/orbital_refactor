import numpy as np
import pytest

from experiments.ring_cann_failure_reanchoring import (
    run_failure_reanchoring_benchmark,
)


def test_short_failure_reanchoring_benchmark_covers_all_conditions():
    traces = run_failure_reanchoring_benchmark(
        duration=0.5, sample_dt=0.1, cue_interval=0.5, seed=2,
    )
    assert len(traces) == 16
    for trace in traces.values():
        assert trace.timestamps.shape == trace.phase_error_deg.shape
        assert np.all(np.isfinite(trace.phase_error_deg))
        assert np.all(trace.concentration > 0.0)
    assert traces["random_10pct:no_cue"].cue_applied.sum() == 0
    assert traces["random_10pct:weak_cue"].cue_applied.sum() == 1


def test_failure_reanchoring_requires_aligned_cue_interval():
    with pytest.raises(ValueError, match="integer multiple"):
        run_failure_reanchoring_benchmark(
            duration=1.0, sample_dt=0.1, cue_interval=0.25,
        )
