from experiments.multitarget_communication_robustness_acceptance import (
    run_multitarget_communication_robustness_acceptance,
)


def test_short_multiseed_communication_acceptance_passes():
    result = run_multitarget_communication_robustness_acceptance(
        seeds=(0, 1),
        duration=80.0,
        dt=10.0,
        observer_count=5,
        outage_node_id="observer-5",
        outage_start=20.0,
        outage_end=30.0,
        topology_switch_time=40.0,
    )

    assert result.passed
    assert result.minimum_final_target_retention_fraction == 1.0
    assert result.minimum_final_consensus_availability_fraction >= 0.8
    assert result.maximum_outage_recovery_delay_seconds <= 20.0
    assert result.total_covariance_psd_failure_count == 0
