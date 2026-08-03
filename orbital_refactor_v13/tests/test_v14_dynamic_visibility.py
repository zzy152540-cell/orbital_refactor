import numpy as np

from experiments.v14_dynamic_visibility import (
    run_v14_az_el_sensitivity,
    run_v14_attitude_error_consistency,
    run_v14_dynamic_visibility_experiment,
    run_v14_range_rate_sensitivity,
)


def test_dynamic_visibility_experiment_runs_all_comparisons_safely():
    result = run_v14_dynamic_visibility_experiment(
        seeds=1, duration=120.0, dt=2.0,
    )

    assert set(result.summary_by_case_and_mode) == {
        ("continuous_range", "propagate_only"),
        ("continuous_range", "exact_transport_event_replay"),
        ("visibility_limited", "propagate_only"),
        ("visibility_limited", "exact_transport_event_replay"),
    }
    assert result.visibility_summary.overall.visibility_rate < 1.0
    for summary in result.summary_by_case_and_mode.values():
        assert summary.transition_timestamp < 120.0
        assert summary.mean_pre_transition_position_rmse > 0.0
        assert summary.mean_post_transition_position_rmse > 0.0
        assert summary.mean_pre_transition_velocity_rmse > 0.0
        assert summary.mean_post_transition_velocity_rmse > 0.0
        assert summary.psd_failure_count == 0
    exact = result.summary_by_case_and_mode[
        ("visibility_limited", "exact_transport_event_replay")
    ]
    assert exact.message_acceptance_rate == 1.0
    assert exact.message_rejection_count == 0


def test_dynamic_visibility_recovery_reports_first_reacquired_nis():
    result = run_v14_dynamic_visibility_experiment(
        seeds=1, duration=120.0, dt=2.0, transition_type="recovery",
    )

    exact = result.summary_by_case_and_mode[
        ("visibility_limited", "exact_transport_event_replay")
    ]
    assert exact.transition_type == "recovery"
    assert exact.transition_timestamp == 68.0
    assert exact.mean_transition_nis is not None
    assert exact.message_acceptance_rate == 1.0
    assert exact.psd_failure_count == 0


def test_dynamic_recovery_supports_range_and_range_rate_nis():
    result = run_v14_dynamic_visibility_experiment(
        seeds=1, duration=120.0, dt=2.0, transition_type="recovery",
        relative_modalities=("RANGE", "RANGE_RATE"),
    )

    exact = result.summary_by_case_and_mode[
        ("visibility_limited", "exact_transport_event_replay")
    ]
    assert exact.relative_modalities == ("RANGE", "RANGE_RATE")
    assert set(exact.mean_nis_by_modality) == {"RANGE", "RANGE_RATE"}
    assert set(exact.mean_transition_nis_by_modality) == {"RANGE", "RANGE_RATE"}
    assert exact.psd_failure_count == 0


def test_dynamic_recovery_supports_eci_az_el_nis():
    result = run_v14_dynamic_visibility_experiment(
        seeds=1, duration=120.0, dt=2.0, transition_type="recovery",
        relative_modalities=("RANGE", "RANGE_RATE", "AZ_EL"),
    )

    exact = result.summary_by_case_and_mode[
        ("visibility_limited", "exact_transport_event_replay")
    ]
    assert set(exact.mean_nis_by_modality) == {"RANGE", "RANGE_RATE", "AZ_EL"}
    assert set(exact.mean_transition_nis_by_modality) == {
        "RANGE", "RANGE_RATE", "AZ_EL",
    }
    assert exact.psd_failure_count == 0


def test_dynamic_recovery_supports_target_pointing_body_az_el_and_fov():
    result = run_v14_dynamic_visibility_experiment(
        seeds=1, duration=120.0, dt=2.0, transition_type="recovery",
        relative_modalities=("RANGE", "RANGE_RATE", "AZ_EL"),
        az_el_frame="BODY",
        az_el_field_of_view_half_angle=np.deg2rad(5.0),
    )

    exact = result.summary_by_case_and_mode[
        ("visibility_limited", "exact_transport_event_replay")
    ]
    assert exact.mean_nis_by_modality["AZ_EL"] > 0.0
    assert result.visibility_summary.by_modality["AZ_EL"].visible_count > 0
    assert exact.message_acceptance_rate == 1.0
    assert exact.psd_failure_count == 0


def test_dynamic_fov_recovery_is_driven_by_body_pointing_not_range():
    result = run_v14_dynamic_visibility_experiment(
        seeds=1, duration=120.0, dt=2.0, transition_type="recovery",
        visibility_driver="fov",
        relative_modalities=("RANGE", "RANGE_RATE", "AZ_EL"),
        az_el_frame="BODY",
        az_el_field_of_view_half_angle=np.deg2rad(5.0),
    )

    exact = result.summary_by_case_and_mode[
        ("visibility_limited", "exact_transport_event_replay")
    ]
    assert exact.transition_timestamp > 0.0
    assert result.visibility_summary.by_modality["RANGE"].visibility_rate == 1.0
    assert 0.0 < result.visibility_summary.by_modality["AZ_EL"].visibility_rate < 1.0
    assert set(
        result.visibility_summary.by_modality["AZ_EL"].rejection_counts
    ) == {"outside_fov"}
    assert (
        result.visibility_summary.by_modality["AZ_EL"]
        .rejection_counts["outside_fov"] > 0
    )
    assert exact.psd_failure_count == 0


def test_dynamic_fov_hysteresis_suppresses_jitter_chatter():
    common = dict(
        seeds=1, duration=120.0, dt=2.0, transition_type="recovery",
        visibility_driver="fov",
        relative_modalities=("RANGE", "RANGE_RATE", "AZ_EL"),
        az_el_frame="BODY",
        az_el_field_of_view_half_angle=np.deg2rad(5.0),
        fov_jitter_amplitude=np.deg2rad(0.8),
    )
    raw = run_v14_dynamic_visibility_experiment(**common)
    stabilized = run_v14_dynamic_visibility_experiment(
        **common, fov_hysteresis=np.deg2rad(0.2),
        fov_acquisition_epochs=2, fov_loss_epochs=2,
    )

    key = ("sat_a", "sat_b", "AZ_EL")
    raw_switches = (
        raw.visibility_summary.availability_switch_count_by_edge_and_modality[key]
    )
    stabilized_switches = (
        stabilized.visibility_summary
        .availability_switch_count_by_edge_and_modality[key]
    )
    assert raw_switches > 1
    assert stabilized_switches < raw_switches
    assert stabilized.summary_by_case_and_mode[
        ("visibility_limited", "exact_transport_event_replay")
    ].psd_failure_count == 0


def test_range_rate_sensitivity_uses_fixed_range_baseline():
    result = run_v14_range_rate_sensitivity(
        range_rate_sigmas=(0.02, 0.05), seeds=1, duration=120.0, dt=2.0,
    )

    assert result.range_only_summary.relative_modalities == ("RANGE",)
    assert set(result.summary_by_range_rate_sigma) == {0.02, 0.05}
    for summary in result.summary_by_range_rate_sigma.values():
        assert summary.relative_modalities == ("RANGE", "RANGE_RATE")
        assert summary.message_acceptance_rate == 1.0
        assert summary.psd_failure_count == 0


def test_az_el_sensitivity_uses_fixed_range_and_rate_baseline():
    result = run_v14_az_el_sensitivity(
        az_el_sigmas_degrees=(0.05, 0.1), seeds=1,
        duration=120.0, dt=2.0,
    )

    assert result.range_and_rate_summary.relative_modalities == (
        "RANGE", "RANGE_RATE",
    )
    assert set(result.summary_by_az_el_sigma_degrees) == {0.05, 0.1}
    for summary in result.summary_by_az_el_sigma_degrees.values():
        assert summary.relative_modalities == ("RANGE", "RANGE_RATE", "AZ_EL")
        assert "AZ_EL" in summary.mean_nis_by_modality
        assert summary.message_acceptance_rate == 1.0
        assert summary.psd_failure_count == 0


def test_attitude_error_consistency_compares_ignored_and_propagated_cases():
    result = run_v14_attitude_error_consistency(
        attitude_error_sigma_degrees=0.05, seeds=1,
        duration=120.0, dt=2.0,
    )

    assert result.ideal_attitude.mean_nis_by_modality["AZ_EL"] > 0.0
    assert result.ignored_attitude_uncertainty.mean_nis_by_modality["AZ_EL"] > 0.0
    assert result.propagated_attitude_uncertainty.mean_nis_by_modality["AZ_EL"] > 0.0
    for summary in (
        result.ideal_attitude,
        result.ignored_attitude_uncertainty,
        result.propagated_attitude_uncertainty,
    ):
        assert summary.message_acceptance_rate == 1.0
        assert summary.psd_failure_count == 0
