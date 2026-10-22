from collections import Counter

import numpy as np
import pytest
import tracking.association.online_tracker as online_tracker_module

from experiments.bilateral_multitarget_frontend import (
    BilateralFrontendConfig,
    _topology_with_inactive_nodes,
    build_bilateral_raw_frontend,
    run_bilateral_distributed_online_comparison,
    run_bilateral_online_comparison,
    run_bilateral_online_smoke,
)
from tracking.association.frontend_adapter import (
    unlabeled_iod_observations_from_messages,
)


def test_frontend_process_noise_must_be_finite_and_nonnegative():
    assert BilateralFrontendConfig(
        process_noise_acceleration=3e-5,
    ).process_noise_acceleration == 3e-5
    with pytest.raises(ValueError, match="process_noise_acceleration"):
        BilateralFrontendConfig(process_noise_acceleration=-1.0)
    with pytest.raises(ValueError, match="multi_epoch_bearing_window_epochs"):
        BilateralFrontendConfig(multi_epoch_bearing_window_epochs=0)
    with pytest.raises(ValueError, match="requires"):
        BilateralFrontendConfig(use_inverse_depth_los_update=True)


def test_radar_dropout_can_use_experimental_multi_epoch_bearing_window():
    result = run_bilateral_online_smoke(BilateralFrontendConfig(
        duration_seconds=5.0,
        random_seed=4,
        dropout_modality="RADAR",
        dropout_start_seconds=3.0,
        dropout_end_seconds=4.0,
        multi_epoch_bearing_window_epochs=3,
    ))

    assert result.final_active_track_count == 15
    assert result.joint_bearing_update_count > 0
    assert result.joint_bearing_measurement_count > (
        result.joint_bearing_update_count
    )
    assert result.all_states_finite


def test_default_frontend_covers_every_target_from_multiple_friendly_observers():
    result = build_bilateral_raw_frontend()

    assert len(result.truth.manifest.friendly_observer_ids) == 15
    assert len(result.truth.manifest.hostile_target_ids) == 15
    assert result.minimum_visible_observers_per_target_epoch >= 3
    assert result.maximum_visible_observers_per_target_epoch <= 4
    assert set(result.truth_target_by_detection_group.values()) == set(
        result.truth.manifest.hostile_target_ids
    )


def test_algorithm_messages_are_target_free_and_keep_three_modalities_per_detection():
    result = build_bilateral_raw_frontend(
        BilateralFrontendConfig(duration_seconds=1.0)
    )
    counts = Counter(message.detection_group_id for message in result.messages)

    assert counts
    assert set(counts.values()) == {3}
    assert all(message.valid_flag for message in result.messages)
    assert all(
        message.observer_id in result.truth.manifest.friendly_observer_ids
        for message in result.messages
    )
    assert all(
        not any("target" in str(key).lower() for key in message.metadata)
        for message in result.messages
    )
    assert all(
        target_id not in message.message_id
        for message in result.messages
        for target_id in result.truth.manifest.hostile_target_ids
    )


def test_target_free_raw_messages_convert_to_j2000_iod_inputs():
    result = build_bilateral_raw_frontend(
        BilateralFrontendConfig(duration_seconds=1.0)
    )
    converted = unlabeled_iod_observations_from_messages(
        result.messages,
        observer_state_by_epoch=result.observer_state_by_epoch,
    )

    assert len(converted) == len(result.messages)
    assert {item.modality for item in converted} == {"RADAR", "LOS"}
    assert all(np.all(np.isfinite(item.measurement)) for item in converted)
    assert all(
        item.observer_id in result.truth.manifest.friendly_observer_ids
        for item in converted
    )


def test_frontend_is_deterministic_and_does_not_retain_images_by_default():
    config = BilateralFrontendConfig(duration_seconds=1.0, random_seed=7)
    first = build_bilateral_raw_frontend(config)
    second = build_bilateral_raw_frontend(config)

    assert not first.raw_frames_by_detection_group
    assert len(first.messages) == len(second.messages)
    assert all(
        left.message_id == right.message_id
        and np.array_equal(left.measurement, right.measurement)
        for left, right in zip(first.messages, second.messages)
    )


def test_all_fifteen_hostile_targets_reach_online_tracking_from_raw_data():
    result = run_bilateral_online_smoke(
        BilateralFrontendConfig(duration_seconds=3.0, random_seed=3)
    )

    assert result.frame_count == 4
    assert result.initialized_track_count == 15
    assert result.final_active_track_count == 15
    assert result.retired_track_count == 0
    assert result.all_states_finite
    # The calibrated raw frontend now realizes its reported sub-pixel angular
    # noise.  At multi-megametre separation this produces kilometre-scale
    # short-arc uncertainty; this smoke bound guards divergence, not accuracy.
    assert result.final_position_rmse_m < 5_000.0
    assert result.final_maximum_position_error_m < 10_000.0
    assert result.identity_switch_count == 0
    assert result.nearest_truth_disagreement_count == 0
    assert result.track_replacement_count == 0
    assert result.missing_truth_sample_count == 0
    assert len(result.position_rmse_m_by_target) == 15
    assert all(
        target_id.startswith("hostile_")
        for target_id in result.position_rmse_m_by_target
    )
    assert result.worst_target_position_rmse_m >= (
        result.final_position_rmse_m
    )
    assert not result.consensus_phase_metrics
    assert result.matched_local_consensus_metrics is None


def test_finite_modality_dropout_removes_only_the_selected_window_and_targets():
    config = BilateralFrontendConfig(
        duration_seconds=4.0,
        random_seed=4,
        dropout_modality="OPTICAL",
        dropout_start_seconds=3.0,
        dropout_end_seconds=3.0,
        dropout_target_indices=(0, 1, 2, 3, 4),
    )
    result = build_bilateral_raw_frontend(config)
    normal_count = len(result.messages_by_timestamp[2.0])
    dropout_messages = result.messages_by_timestamp[3.0]
    affected_targets = set(result.truth.manifest.hostile_target_ids[:5])

    assert len(dropout_messages) < normal_count
    assert len(result.messages_by_timestamp[4.0]) == normal_count
    assert all(
        not (
            message.modality == "OPTICAL"
            and result.truth_target_by_detection_group[
                message.detection_group_id
            ] in affected_targets
        )
        for message in dropout_messages
    )
    assert any(
        message.modality == "OPTICAL"
        and result.truth_target_by_detection_group[
            message.detection_group_id
        ] not in affected_targets
        for message in dropout_messages
    )


def test_empty_dropout_target_set_defines_control_phases_without_removing_data():
    result = build_bilateral_raw_frontend(BilateralFrontendConfig(
        duration_seconds=4.0,
        dropout_modality="OPTICAL",
        dropout_start_seconds=3.0,
        dropout_end_seconds=3.0,
        dropout_target_indices=(),
    ))

    assert len(result.messages_by_timestamp[3.0]) == (
        len(result.messages_by_timestamp[2.0])
    )


def test_radar_dropout_uses_multi_observer_optical_and_infrared_updates():
    result = run_bilateral_online_smoke(BilateralFrontendConfig(
        duration_seconds=4.0,
        random_seed=4,
        dropout_modality="RADAR",
        dropout_start_seconds=3.0,
        dropout_end_seconds=3.0,
    ))

    assert result.final_active_track_count == 15
    assert result.joint_bearing_update_count > 0
    assert result.joint_bearing_measurement_count > (
        result.joint_bearing_update_count
    )
    assert result.local_modality_weight_sample_count["ir"] > 0
    assert result.local_modality_weight_sample_count["opt"] > 0


def test_radar_dropout_joint_bearing_path_runs_incremental_cann_preprocessing():
    result = run_bilateral_online_comparison(BilateralFrontendConfig(
        duration_seconds=4.0,
        random_seed=4,
        dropout_modality="RADAR",
        dropout_start_seconds=3.0,
        dropout_end_seconds=3.0,
    ))

    assert result.without_cann.joint_bearing_update_count > 0
    assert result.incremental_cann.joint_bearing_update_count > 0
    assert result.without_cann.cann_preprocessed_observation_count == 0
    assert result.incremental_cann.cann_preprocessed_observation_count > 0
    assert result.incremental_cann.joint_bearing_measurement_count > (
        result.incremental_cann.joint_bearing_update_count
    )


def test_radial_backup_is_not_discarded_by_joint_bearing_route():
    result = run_bilateral_online_comparison(BilateralFrontendConfig(
        duration_seconds=4.0,
        random_seed=4,
        dropout_modality="RADAR",
        dropout_start_seconds=3.0,
        dropout_end_seconds=3.0,
        use_local_unified_multimodal_measurement_update=True,
        backup_radial_range_sigma_m=0.2,
        backup_radial_range_rate_sigma_mps=0.0125,
    ))

    assert result.incremental_cann.final_active_track_count == 15
    assert result.without_cann.joint_bearing_update_count == 0
    assert result.incremental_cann.joint_bearing_update_count == 0


def test_radial_backup_can_be_restricted_to_selected_observers():
    result = build_bilateral_raw_frontend(BilateralFrontendConfig(
        duration_seconds=2.0,
        random_seed=4,
        backup_radial_range_sigma_m=1.0,
        backup_radial_range_rate_sigma_mps=0.0125,
        backup_radial_observer_indices=(0,),
    ))
    selected_id = result.truth.manifest.friendly_observer_ids[0]
    messages = tuple(result.messages_by_timestamp[1.0])
    selected = [
        item for item in messages
        if item.observer_id == selected_id and item.modality == "INFRARED"
    ]
    unselected = [
        item for item in messages
        if item.observer_id != selected_id and item.modality == "INFRARED"
    ]

    assert selected
    assert unselected
    assert all("rangeProxyM" in item.metadata for item in selected)
    assert all("rangeRateProxyMps" in item.metadata for item in selected)
    assert all("rangeProxyM" not in item.metadata for item in unselected)
    assert all("rangeRateProxyMps" not in item.metadata for item in unselected)


def test_independent_radial_backup_survives_infrared_and_radar_dropout():
    for missing_modality in ("INFRARED", "RADAR"):
        result = build_bilateral_raw_frontend(BilateralFrontendConfig(
            duration_seconds=4.0,
            random_seed=4,
            dropout_modality=missing_modality,
            dropout_start_seconds=3.0,
            dropout_end_seconds=3.0,
            backup_radial_range_sigma_m=1.0,
            backup_radial_range_rate_sigma_mps=0.0125,
            independent_radial_backup_channel=True,
        ))
        messages = tuple(result.messages_by_timestamp[3.0])
        backups = [
            item for item in messages
            if item.metadata.get("sourceModality") == "RADIAL_BACKUP"
        ]

        assert backups
        assert all(item.modality == "RADAR" for item in backups)
        assert all(
            item.metadata["radialBackupCarrierModality"]
            == "INDEPENDENT_CHANNEL"
            for item in backups
        )
        assert not any(
            item.modality == missing_modality
            and item.metadata.get("sourceModality") != "RADIAL_BACKUP"
            for item in messages
        )
        assert all(
            "rangeProxyM" not in item.metadata
            for item in messages if item.modality == "INFRARED"
        )


def test_unified_multimodal_path_handles_control_and_radar_dropout_epochs():
    result = run_bilateral_online_comparison(BilateralFrontendConfig(
        duration_seconds=4.0,
        random_seed=4,
        dropout_modality="RADAR",
        dropout_start_seconds=3.0,
        dropout_end_seconds=3.0,
        use_unified_multimodal_measurement_update=True,
    ))

    assert result.incremental_cann.final_active_track_count == 15
    assert result.incremental_cann.unified_multimodal_update_count > 0
    assert result.incremental_cann.unified_multimodal_observation_count > (
        result.incremental_cann.unified_multimodal_update_count
    )
    assert result.incremental_cann.joint_bearing_update_count == 0
    assert result.incremental_cann.cann_preprocessed_observation_count > 0


def test_unified_radar_dropout_can_use_inverse_depth_los_update(monkeypatch):
    calls = []
    original = online_tracker_module.inverse_depth_multi_observer_bearing_update

    def recording_update(*args, **kwargs):
        calls.append(float(kwargs["timestamp"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        online_tracker_module,
        "inverse_depth_multi_observer_bearing_update",
        recording_update,
    )
    result = run_bilateral_online_smoke(BilateralFrontendConfig(
        duration_seconds=4.0,
        random_seed=4,
        dropout_modality="RADAR",
        dropout_start_seconds=3.0,
        dropout_end_seconds=3.0,
        use_unified_multimodal_measurement_update=True,
        use_inverse_depth_los_update=True,
    ))

    assert calls
    assert set(calls) == {3.0}
    assert result.final_active_track_count == 15
    assert result.all_states_finite


def test_seed_two_control_rejects_inconsistent_angular_ci_posterior():
    result = run_bilateral_online_smoke(BilateralFrontendConfig(
        duration_seconds=10.0,
        random_seed=2,
    ))

    assert result.final_active_track_count == 15
    assert result.identity_switch_count == 0
    assert result.missing_truth_sample_count == 0
    assert result.worst_target_position_rmse_m < 20.0


def test_three_friendly_observers_fail_only_inside_the_configured_window():
    config = BilateralFrontendConfig(
        duration_seconds=4.0,
        random_seed=6,
        observer_failure_start_seconds=2.0,
        observer_failure_end_seconds=3.0,
        failed_observer_indices=(0, 5, 10),
    )
    result = build_bilateral_raw_frontend(config)
    failed_ids = {
        result.truth.manifest.friendly_observer_ids[index]
        for index in config.failed_observer_indices
    }

    assert len(result.messages_by_timestamp[2.0]) < (
        len(result.messages_by_timestamp[1.0])
    )
    assert len(result.messages_by_timestamp[3.0]) < (
        len(result.messages_by_timestamp[4.0])
    )
    assert all(
        message.observer_id not in failed_ids
        for timestamp in (2.0, 3.0)
        for message in result.messages_by_timestamp[timestamp]
    )
    assert all(
        any(
            message.observer_id == observer_id
            for message in result.messages_by_timestamp[4.0]
        )
        for observer_id in failed_ids
    )
    assert set(result.observer_state_by_epoch) == {
        (observer_id, float(timestamp))
        for observer_id in result.truth.manifest.friendly_observer_ids
        for timestamp in result.truth.timestamps
    }


def test_observer_failure_control_window_keeps_all_observations():
    result = build_bilateral_raw_frontend(BilateralFrontendConfig(
        duration_seconds=3.0,
        observer_failure_start_seconds=1.0,
        observer_failure_end_seconds=2.0,
        failed_observer_indices=(),
    ))

    assert len(result.messages_by_timestamp[2.0]) == (
        len(result.messages_by_timestamp[0.0])
    )


def test_failed_observers_have_no_inbound_or_outbound_fixed_topology_edges():
    frontend = build_bilateral_raw_frontend(BilateralFrontendConfig(
        duration_seconds=1.0,
    ))
    base = frontend.truth.manifest.friendly_topology
    failed = {
        frontend.truth.manifest.friendly_observer_ids[index]
        for index in (0, 5, 10)
    }
    active = _topology_with_inactive_nodes(
        base, inactive_node_ids=failed,
    )

    assert all(active.neighbors(node_id) == () for node_id in failed)
    assert all(
        not (set(active.neighbors(node_id)) & failed)
        for node_id in active.node_ids
    )
    assert sum(len(active.neighbors(node)) for node in active.node_ids) < (
        sum(len(base.neighbors(node)) for node in base.node_ids)
    )


def test_distributed_node_failure_disables_edges_and_recovers_consensus_links():
    result = run_bilateral_distributed_online_comparison(
        BilateralFrontendConfig(
            duration_seconds=5.0,
            random_seed=8,
            observer_failure_start_seconds=2.0,
            observer_failure_end_seconds=3.0,
            failed_observer_indices=(0, 5, 10),
        )
    )
    arm = result.incremental_cann

    assert arm.final_active_track_count == 15
    assert arm.retired_track_count == 0
    assert arm.consensus_received_message_count > 0
    assert arm.consensus_recovered_link_count > 0
    assert arm.consensus_identity_switch_count == 0
    assert arm.consensus_nearest_truth_disagreement_count == 0
    assert arm.consensus_track_replacement_count == 0
    assert arm.consensus_missing_truth_sample_count > 0
    assert len(arm.consensus_position_rmse_m_by_target) == 15
    assert all(
        target_id.startswith("hostile_")
        for target_id in arm.consensus_position_rmse_m_by_target
    )
    assert set(arm.consensus_phase_metrics) == {
        "PRE_DROPOUT", "DROPOUT", "RECOVERY",
    }
    assert arm.active_communication_node_count_by_timestamp[2.0] == 12
    assert arm.active_communication_node_count_by_timestamp[4.0] == 15
    assert arm.active_communication_edge_count_by_timestamp[2.0] < (
        arm.active_communication_edge_count_by_timestamp[4.0]
    )
    matched = arm.matched_local_consensus_metrics
    assert matched is not None
    assert matched.common_sample_count > 0
    assert matched.common_sample_count <= matched.local_sample_count
    assert matched.common_sample_count <= matched.consensus_sample_count
    assert matched.local_only_sample_count == (
        matched.local_sample_count - matched.common_sample_count
    )
    assert matched.consensus_only_sample_count == (
        matched.consensus_sample_count - matched.common_sample_count
    )
    assert np.isfinite(matched.position_mean_error_improvement_percent)
    assert np.isfinite(matched.position_rmse_improvement_percent)
    assert np.isfinite(matched.velocity_mean_error_improvement_percent)
    assert np.isfinite(matched.velocity_rmse_improvement_percent)


def test_paired_cann_comparison_uses_the_same_dropout_frames_and_keeps_tracks():
    result = run_bilateral_online_comparison(BilateralFrontendConfig(
        duration_seconds=3.0,
        random_seed=5,
        dropout_modality="OPTICAL",
        dropout_start_seconds=1.0,
        dropout_end_seconds=2.0,
        dropout_target_indices=(0, 1, 2),
    ))

    assert result.message_count_by_timestamp[2.0] < (
        result.message_count_by_timestamp[3.0]
    )
    assert result.without_cann.final_active_track_count == 15
    assert result.incremental_cann.final_active_track_count == 15
    assert result.without_cann.retired_track_count == 0
    assert result.incremental_cann.retired_track_count == 0
    assert result.without_cann.cann_preprocessed_observation_count == 0
    assert result.incremental_cann.cann_preprocessed_observation_count > 0
    assert set(result.without_cann.mean_local_modality_weight) == {
        "ir", "opt", "rad",
    }
    assert all(
        count > 0
        for count in result.without_cann.local_modality_weight_sample_count.values()
    )
    assert result.without_cann.cann_action_counts == {}
    assert all(
        action.startswith(("RADAR.", "OPTICAL.", "INFRARED."))
        for action in result.incremental_cann.cann_action_counts
    )
    assert np.isfinite(result.without_cann.final_acceleration_rmse_mps2)
    assert np.isfinite(result.incremental_cann.final_acceleration_rmse_mps2)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"dropout_modality": "OPTICAL"}, "requires a dropout interval"),
        (
            {
                "dropout_start_seconds": 1.0,
                "dropout_end_seconds": 2.0,
            },
            "requires dropout_modality",
        ),
        (
            {
                "dropout_modality": "OPTICAL",
                "dropout_start_seconds": 2.0,
                "dropout_end_seconds": 1.0,
            },
            "must lie inside",
        ),
        (
            {"failed_observer_indices": (0, 5, 10)},
            "requires an observer failure interval",
        ),
        (
            {
                "observer_failure_start_seconds": 1.0,
                "observer_failure_end_seconds": 2.0,
            },
            "requires failed_observer_indices",
        ),
    ],
)
def test_dropout_configuration_rejects_incomplete_or_invalid_windows(
    overrides, match,
):
    with pytest.raises(ValueError, match=match):
        BilateralFrontendConfig(duration_seconds=3.0, **overrides)
