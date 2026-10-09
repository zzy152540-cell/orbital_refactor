from experiments.multitarget_multimodal_dynamic_visibility import (
    run_multitarget_multimodal_dynamic_visibility,
)


def test_raw_multimodal_dynamic_visibility_reaches_distributed_consensus():
    result = run_multitarget_multimodal_dynamic_visibility(
        duration=60.0,
        dt=10.0,
        random_seed=7,
    )

    assert result.observer_ids == ("observer-1", "observer-2", "observer-3")
    assert result.target_ids == ("target-01", "target-02")
    assert set(result.opportunity_count_by_modality) == {
        "RADAR", "INFRARED", "OPTICAL",
    }
    assert all(
        0 < result.visible_count_by_modality[modality]
        < result.opportunity_count_by_modality[modality]
        for modality in result.opportunity_count_by_modality
    )
    assert result.valid_measurement_count_by_modality == (
        result.visible_count_by_modality
    )
    assert result.feedback_application_count > 0
    assert result.covariance_psd_failure_count == 0
    assert set(result.final_position_error_m_by_node_target) == {
        (observer_id, target_id)
        for observer_id in result.observer_ids
        for target_id in result.target_ids
    }
    assert result.maximum_final_position_error_m < 5_000.0
    assert result.maximum_final_velocity_error_mps < 20.0
    assert max(
        result.consensus_contributor_count_by_epoch_node_target.values()
    ) >= 2


def test_visibility_changes_are_geometry_driven_over_longer_arc():
    result = run_multitarget_multimodal_dynamic_visibility(
        duration=600.0,
        dt=60.0,
        random_seed=3,
    )

    assert sum(result.visibility_switch_count_by_modality.values()) > 0
    assert result.covariance_psd_failure_count == 0


def test_five_observers_retain_both_targets_and_use_all_ci_contributors():
    result = run_multitarget_multimodal_dynamic_visibility(
        duration=40.0,
        dt=10.0,
        random_seed=11,
        observer_count=5,
    )

    assert len(result.observer_ids) == 5
    assert set(result.final_position_error_m_by_node_target) == {
        (observer_id, target_id)
        for observer_id in result.observer_ids
        for target_id in result.target_ids
    }
    assert max(
        result.consensus_contributor_count_by_epoch_node_target.values()
    ) == 5
    assert result.maximum_final_position_error_m < 5_000.0
    assert result.maximum_final_velocity_error_mps < 20.0
    assert result.covariance_psd_failure_count == 0
    assert result.elapsed_seconds > 0.0


def test_ten_observer_controlled_prescan_retains_both_targets():
    result = run_multitarget_multimodal_dynamic_visibility(
        duration=30.0,
        dt=10.0,
        random_seed=5,
        observer_count=10,
    )

    assert set(result.final_position_error_m_by_node_target) == {
        (observer_id, target_id)
        for observer_id in result.observer_ids
        for target_id in result.target_ids
    }
    assert max(
        result.consensus_contributor_count_by_epoch_node_target.values()
    ) == 10
    assert result.covariance_psd_failure_count == 0


def test_delay_loss_and_node_outage_recover_without_losing_targets():
    result = run_multitarget_multimodal_dynamic_visibility(
        duration=80.0,
        dt=10.0,
        random_seed=13,
        observer_count=5,
        communication_delay_seconds=10.0,
        packet_loss_rate=0.2,
        outage_node_id="observer-5",
        outage_start=30.0,
        outage_end=50.0,
        topology="fully_connected",
        topology_switch_time=60.0,
        topology_after_switch="chain",
    )

    assert result.outage_epoch_count == 3
    assert result.topology_switch_count == 1
    assert result.received_message_count > 0
    assert result.recovered_source_event_count > 0
    assert set(result.final_position_error_m_by_node_target) == {
        (observer_id, target_id)
        for observer_id in result.observer_ids
        for target_id in result.target_ids
    }
    assert result.covariance_psd_failure_count == 0
    assert result.maximum_position_error_m < 10_000.0
    assert result.maximum_velocity_error_mps < 50.0


def test_full_to_chain_topology_switch_preserves_all_target_tracks():
    result = run_multitarget_multimodal_dynamic_visibility(
        duration=80.0,
        dt=10.0,
        random_seed=17,
        observer_count=5,
        topology="fully_connected",
        topology_switch_time=40.0,
        topology_after_switch="chain",
    )

    assert result.topology_switch_count == 1
    assert set(result.final_position_error_m_by_node_target) == {
        (observer_id, target_id)
        for observer_id in result.observer_ids
        for target_id in result.target_ids
    }
    final_timestamp = result.timestamps[-1]
    final_contributors = [
        value
        for (timestamp, _node_id, _target_id), value in (
            result.consensus_contributor_count_by_epoch_node_target.items()
        )
        if timestamp == final_timestamp
    ]
    assert final_contributors
    assert max(final_contributors) <= 3
    assert result.covariance_psd_failure_count == 0
