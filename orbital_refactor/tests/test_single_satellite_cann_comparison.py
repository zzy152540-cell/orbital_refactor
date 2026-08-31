import numpy as np

from experiments.single_satellite_cann_comparison import run_single_satellite_cann_comparison


def test_short_three_modal_single_satellite_cann_comparison_runs():
    result = run_single_satellite_cann_comparison(
        duration=20.0, dt=2.0, outage_start=8.0, outage_end=12.0,
        cue_interval_samples=2,
    )
    assert result["summary"]["cann_valid_fraction"] == 1.0
    assert np.isfinite(result["summary"]["position_rmse_m"])
    assert result["summary"]["infrared_valid_count"] > 0
    assert result["summary"]["radar_valid_count"] > 0


def test_selective_outage_only_removes_requested_modality():
    result = run_single_satellite_cann_comparison(
        duration=20.0, dt=2.0, outage_start=8.0, outage_end=12.0,
        outage_modalities=("ir",),
    )
    summary = result["summary"]
    assert summary["outage_modalities"] == ["ir"]
    assert summary["infrared_valid_count"] < summary["radar_valid_count"]
    assert summary["position_rmse_outage_m"] is not None


def test_original_filter_runs_without_constructing_cann_sidecar():
    result = run_single_satellite_cann_comparison(
        duration=20.0, dt=2.0, outage_start=8.0, outage_end=12.0,
        enable_cann=False,
    )
    assert result["cann"] is None
    assert result["summary"]["cann_enabled"] is False
    assert result["summary"]["cann_phase_rmse_deg"] is None
