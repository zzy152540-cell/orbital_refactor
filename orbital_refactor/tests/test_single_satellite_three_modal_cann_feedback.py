import numpy as np

from experiments.single_satellite_three_modal_cann_feedback import (
    run_staggered_modality_outage_comparison,
    run_staggered_recovery_fault_comparison,
    run_single_satellite_three_modal_cann_feedback,
    write_single_satellite_three_modal_cann_feedback,
)


def test_three_modal_feedback_exposes_each_preprocessor():
    result = run_single_satellite_three_modal_cann_feedback(
        inject_faults=False, duration=12.0, dt=2.0,
        outage_start=4.0, outage_end=8.0,
    )
    summary = result["processed"]["summary"]
    assert summary["hybrid_cann_preprocess_ir"]
    assert summary["radar_cann_preprocess"]
    assert summary["optical_cann_preprocess"]


def test_three_modal_feedback_writer_creates_summary_and_comparison_plot(tmp_path):
    result = run_single_satellite_three_modal_cann_feedback(
        inject_faults=False, duration=12.0, dt=2.0,
        outage_start=4.0, outage_end=8.0,
    )
    paths = write_single_satellite_three_modal_cann_feedback(result, tmp_path)
    assert paths["summary"].is_file()
    assert paths["figure"].is_file()
    assert paths["figure"].stat().st_size > 0


def test_staggered_outage_comparison_includes_no_outage_reference():
    result = run_staggered_modality_outage_comparison(
        duration=18.0, dt=2.0,
        outage_windows={"opt": (2.0, 4.0), "ir": (8.0, 10.0),
                        "rad": (14.0, 16.0)},
    )
    assert "no_outage_reference" in result
    impact = result["summary"]["outage_impact_vs_no_outage"]
    assert set(impact) == {"opt", "ir", "rad"}
    assert all(
        np.isfinite(values["baseline"]["net_position_rmse_change_m"])
        for values in impact.values()
    )


def test_recovery_fault_comparison_schedules_first_post_outage_samples():
    result = run_staggered_recovery_fault_comparison(
        duration=18.0, dt=2.0,
        outage_windows={"ir": (4.0, 6.0), "rad": (10.0, 12.0)},
    )
    assert result["summary"]["recovery_fault_times_by_modality"] == {
        "ir": [8.0, 10.0], "rad": [14.0, 16.0],
    }
    assert set(result["summary"]["recovery_fault_impact"]) == {"ir", "rad"}
