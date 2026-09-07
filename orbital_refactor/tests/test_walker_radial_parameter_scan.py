from experiments.walker_radial_parameter_scan import run_walker_radial_parameter_scan


def test_radial_parameter_scan_returns_requested_grid():
    records = run_walker_radial_parameter_scan(
        neuron_counts=(81,), tuning_widths_m=(500.0,),
        anchor_gains=(0.25, 0.4), duration=2.0, dt=2.0,
    )
    assert len(records) == 2
    assert {record.maximum_anchor_gain for record in records} == {0.25, 0.4}
    assert all(record.radial_rmse_m >= 0.0 for record in records)
