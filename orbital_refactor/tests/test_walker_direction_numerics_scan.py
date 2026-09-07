import numpy as np

from experiments.walker_direction_numerics_scan import run_walker_direction_numerics_scan


def test_direction_numerics_scan_records_each_discretization():
    records = run_walker_direction_numerics_scan(
        duration=2.0, dt=2.0, node_limit=1,
        neuron_counts=(36,), internal_steps=(0.001, 0.002),
    )
    assert len(records) == 2
    assert {record.internal_dt for record in records} == {0.001, 0.002}
    assert all(record.phase_rmse_deg >= 0.0 for record in records)
    assert all(np.isfinite(record.drift_rate_deg_per_hour) for record in records)
