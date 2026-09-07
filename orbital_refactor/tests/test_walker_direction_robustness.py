from dataclasses import replace

from brain_inspired.ring_cann import RingCANNConfig
from experiments.walker_direction_robustness import (
    run_walker_direction_robustness,
    summarize_direction_robustness,
)


def test_direction_robustness_reuses_each_seed_for_all_policies():
    records = run_walker_direction_robustness(
        seeds=(0,), duration=2.0, dt=2.0, node_limit=1,
        anchor_interval_seconds=2.0, anchor_outage_window=(3.0, 4.0),
        anchor_gains=(0.25, 0.4),
        ring_config=replace(
            RingCANNConfig(), num_neurons=36,
            initialization_duration=0.02,
        ),
    )
    assert [record.policy for record in records] == [
        "free_running", "outage_gain_0.25", "outage_gain_0.4",
    ]
    assert all(record.valid_fraction == 1.0 for record in records)
    assert set(summarize_direction_robustness(records)) == {
        "free_running", "outage_gain_0.25", "outage_gain_0.4",
    }


def test_direction_robustness_can_sweep_outage_lengths():
    records = run_walker_direction_robustness(
        seeds=(0,), duration=2.0, dt=2.0, node_limit=1,
        anchor_interval_seconds=2.0,
        anchor_outage_windows=((0.0, 1.0), (0.0, 2.0)),
        anchor_gains=(0.25,),
        ring_config=replace(
            RingCANNConfig(), num_neurons=36,
            initialization_duration=0.02,
        ),
    )
    assert [record.policy for record in records] == [
        "free_running", "outage_1s_gain_0.25", "outage_2s_gain_0.25",
    ]
