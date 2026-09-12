import numpy as np

from experiments.external_single_cann_robustness_audit import (
    run_external_single_cann_robustness_audit,
)


def test_short_cann_robustness_audit_preserves_missing_measurement_boundary():
    report = run_external_single_cann_robustness_audit(
        seeds=(0,), modalities=("rad",), duration=12.0, dt=2.0,
        outage_window=(4.0, 6.0), recovery_fault_samples=1,
        recovery_horizon=4.0,
    )
    assert len(report.records) == 1
    record = report.records[0]
    assert record.pure_outage_error_change_with_cann_m == 0.0
    assert np.isfinite(record.recovery_fault_mitigation_m)
