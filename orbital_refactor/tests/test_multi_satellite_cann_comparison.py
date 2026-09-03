import numpy as np

from experiments.multi_satellite_cann_comparison import (
    _build_scenario, _visibility_flags,
    audit_multi_satellite_geometry,
    run_multi_satellite_cann_comparison,
)
from orbital_core.coordinates import state_history_eci_to_spri
from scenarios.measurement_visibility import (
    VisibilityConfig, VisibilityTemporalFilterConfig,
)


def test_short_three_satellite_cann_comparison_runs_paired_paths():
    result = run_multi_satellite_cann_comparison(
        duration=8.0, seed=2,
        fault_by_node={"sat_01": {"radar": (4.0,)}},
    )
    summary = result["summary"]
    assert set(summary["baseline_local_position_rmse_m"]) == {
        "sat_01", "sat_02", "sat_03",
    }
    assert np.isfinite(summary["baseline_position_rmse_m"])
    assert np.isfinite(summary["cann_position_rmse_m"])
    assert set(summary["visibility_rate_by_node"]) == {
        "sat_01", "sat_02", "sat_03",
    }
    diagnostics = summary["fault_diagnostics"]["sat_01"]
    assert summary["injected_fault_count_by_node_modality"] == {
        "sat_01": {"radar": 1},
    }
    assert set(diagnostics) == {"baseline", "cann"}
    assert 0.0 <= diagnostics["baseline"]["mean_ci_weight"] <= 1.0
    assert np.isfinite(diagnostics["cann"]["local_position_rmse_m"])


def test_geometry_audit_does_not_run_filter_and_reports_three_nodes():
    audit = audit_multi_satellite_geometry(duration=20.0)
    assert set(audit) == {"sat_01", "sat_02", "sat_03"}
    assert all(item["minimum_range_km"] > 0.0 for item in audit.values())


def test_default_optical_visibility_matches_positive_spri_depth():
    timestamps = np.arange(0.0, 1800.1, 2.0)
    scenario = _build_scenario(timestamps)
    optical = VisibilityConfig(
        maximum_range=75e3, field_of_view_half_angle=np.deg2rad(89.0),
        boresight_axis=(0.0, 0.0, 1.0),
    )
    flags = _visibility_flags(
        scenario, "sat_03", {"OPTICAL": optical},
        {"OPTICAL": VisibilityTemporalFilterConfig(
            acquisition_epochs=2, loss_epochs=2,
        )},
    )["OPTICAL"]
    observer = scenario.observer_trajectories["sat_03"]
    spri = state_history_eci_to_spri(
        scenario.relative_state_eci_by_node["sat_03"],
        observer.q_eci2pri_history,
    )
    assert np.any(flags)
    assert np.all(spri[flags, 2] > 0.0)


def test_recovery_faults_use_first_actually_valid_samples_after_dropout():
    result = run_multi_satellite_cann_comparison(
        duration=12.0, seed=1,
        dropout_windows_by_node={
            "sat_01": {"RADAR": ((2.0, 6.0),)},
        },
        recovery_fault_samples=2,
    )
    summary = result["summary"]
    assert summary["recovery_fault_times_by_node_modality"] == {
        "sat_01": {"radar": (8.0, 10.0)},
    }
    assert summary["injected_fault_count_by_node_modality"] == {
        "sat_01": {"radar": 2},
    }


def test_confirmation_baseline_is_reported_separately_from_cann():
    result = run_multi_satellite_cann_comparison(
        duration=12.0, seed=3,
        dropout_windows_by_node={
            "sat_01": {"RADAR": ((2.0, 6.0),)},
        },
        recovery_fault_samples=2, include_confirmation_baseline=True,
    )
    summary = result["summary"]
    assert summary["confirmation_position_rmse_m"] is not None
    assert np.isclose(
        summary["cann_minus_confirmation_position_rmse_m"],
        summary["cann_position_rmse_m"]
        - summary["confirmation_position_rmse_m"],
    )
