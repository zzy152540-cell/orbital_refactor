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
