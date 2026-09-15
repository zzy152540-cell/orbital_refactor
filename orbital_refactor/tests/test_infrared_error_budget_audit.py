from experiments.infrared_error_budget_audit import (
    InfraredErrorProfile,
    run_infrared_error_budget_audit,
)


def test_infrared_error_budget_keeps_scope_and_reports_bias():
    report = run_infrared_error_budget_audit(
        monte_carlo_samples=8, focal_lengths_pixels=(300.0,),
        peak_snrs=(1000.0,), profiles=(
            InfraredErrorProfile("baseline"),
            InfraredErrorProfile(
                "biased", fixed_pixel_bias_x=0.2,
                fixed_pixel_bias_y=-0.1,
            ),
        ),
    )
    assert len(report.records) == 4
    assert "Infrared focal-plane" in report.model_scope
    assert all(0.0 <= record.detection_fraction <= 1.0
               for record in report.records)
    assert all(record.angular_rmse_deg > 0.0 for record in report.records)
    biased = [record for record in report.records if record.profile == "biased"]
    assert any(abs(record.azimuth_bias_deg) > 0.0 for record in biased)
