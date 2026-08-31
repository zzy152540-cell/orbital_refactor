import csv

import numpy as np

from experiments.ring_cann_benchmark import (
    run_ring_cann_benchmark,
    write_ring_cann_benchmark_csv,
)


def test_short_ring_cann_benchmark_has_finite_aligned_traces():
    traces = run_ring_cann_benchmark(sample_dt=0.02, duration_scale=0.02)
    assert set(traces) == {
        "static", "positive_rate", "negative_rate", "wrap", "external_cue",
    }
    for name, trace in traces.items():
        assert trace.scenario == name
        assert trace.time[0] == 0.0
        assert np.all(np.diff(trace.time) > 0.0)
        assert all(values.shape == trace.time.shape for values in (
            trace.truth_phase, trace.decoded_phase, trace.phase_error,
            trace.bump_concentration, trace.bump_width,
        ))
        assert np.all(np.isfinite(trace.phase_error))
        assert np.all((trace.bump_concentration >= 0.0)
                      & (trace.bump_concentration <= 1.0))


def test_ring_cann_benchmark_csv_contains_all_scenarios(tmp_path):
    traces = run_ring_cann_benchmark(sample_dt=0.02, duration_scale=0.01)
    output = write_ring_cann_benchmark_csv(traces, tmp_path / "benchmark.csv")
    with output.open(newline="", encoding="utf-8") as stream:
        rows = tuple(csv.DictReader(stream))
    assert {row["scenario"] for row in rows} == set(traces)
    assert len(rows) == sum(trace.time.size for trace in traces.values())
