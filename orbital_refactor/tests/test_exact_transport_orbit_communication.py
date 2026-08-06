from experiments.exact_transport_orbit_communication import (
    run_exact_transport_orbit_communication_validation,
)


def test_orbit_communication_exposes_delay_and_loss_baseline_failures():
    result = run_exact_transport_orbit_communication_validation(seeds=3, epochs=5)
    assert result["ideal"].acceptance_rate == 1.0
    assert result["ideal"].maximum_covariance_disagreement < 1e-8
    assert result["ideal"].minimum_joint_eigenvalue >= -1e-8
    assert result["one_epoch_delay"].acceptance_rate < 0.1
    delayed = result["one_epoch_delay_rollback"]
    assert delayed.accepted_messages == delayed.delivered_messages
    assert delayed.acceptance_rate == 0.8
    two_epoch = result["two_epoch_delay_rollback"]
    assert two_epoch.accepted_messages == two_epoch.delivered_messages
    assert result["two_epoch_delay_window_one"].accepted_messages == 0
    assert result["two_epoch_delay_window_one"].rejection_counts["history_unavailable"] > 0
    assert result["loss_20_percent"].acceptance_rate < 1.0
    assert result["loss_20_percent_ack"].accepted_messages == result["loss_20_percent_ack"].delivered_messages
    combined = result["delay_loss_ack_rollback"]
    assert combined.accepted_messages == combined.delivered_messages
    assert combined.rejection_counts == {}
