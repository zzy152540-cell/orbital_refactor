import numpy as np

from experiments.infrared_extent_observability_audit import (
    h_ir_with_log_extent,
    run_infrared_extent_observability_audit,
)
from experiments.single_satellite_cann_comparison import (
    run_single_satellite_cann_comparison,
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


def test_extent_can_enter_federated_infrared_branch_without_new_modality():
    result = run_single_satellite_cann_comparison(
        duration=4.0, dt=2.0, seed=0, outage_modalities=(),
        enable_cann=False, infrared_extent_enabled=True,
    )

    assert result["summary"]["infrared_extent_enabled"]
    assert set(result["ci_weight_history"][-1]) <= {"opt", "ir", "rad"}
    assert "ir" in result["ci_weight_history"][-1]
    assert np.isfinite(result["summary"]["position_rmse_m"])
