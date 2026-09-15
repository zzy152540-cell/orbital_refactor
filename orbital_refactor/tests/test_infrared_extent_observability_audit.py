import numpy as np

from experiments.infrared_extent_observability_audit import (
    h_ir_with_log_extent,
    run_infrared_extent_observability_audit,
)


def test_log_extent_decreases_with_range():
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    near = np.array([1000.0, 20.0, 10.0, 0.0, 0.0, 0.0])
    far = near.copy(); far[:3] *= 2.0
    assert h_ir_with_log_extent(near, quaternion, 10.0)[2] > (
        h_ir_with_log_extent(far, quaternion, 10.0)[2]
    )


def test_extent_adds_instantaneous_radial_information_in_shadow_audit():
    report = run_infrared_extent_observability_audit(
        duration=4.0, dt=2.0, extent_fractional_sigmas=(0.1,),
    )
    records = {record.modalities: record for record in report.records}
    assert report.baseline_unchanged
    assert report.relative_range_minimum_m > 0.0
    assert report.angular_extent_at_300px_focal_maximum_pixels < 1.0
    assert records["ir"].instantaneous_rank_max == 2
    assert records["ir_extent"].instantaneous_rank_min == 3
    assert records["ir_extent+rad"].horizon_smallest_singular_value > (
        records["ir+rad"].horizon_smallest_singular_value
    )
