from dataclasses import replace

from brain_inspired.ring_cann import RingCANNConfig
from experiments.walker_direction_anchor_scan import run_walker_direction_anchor_scan


def test_anchor_scan_returns_complete_finite_grid():
    records = run_walker_direction_anchor_scan(
        duration=2.0, dt=2.0, node_limit=1,
        phase_rate_biases_deg_per_hour=(1.0,),
        maximum_anchor_gains=(0.1, 0.2),
        anchor_intervals_seconds=(2.0,),
        ring_config=replace(
            RingCANNConfig(), num_neurons=36,
            initialization_duration=0.02,
        ),
    )
    assert len(records) == 2
    assert {record.maximum_anchor_gain for record in records} == {0.1, 0.2}
    assert all(record.phase_rmse_deg >= 0.0 for record in records)
