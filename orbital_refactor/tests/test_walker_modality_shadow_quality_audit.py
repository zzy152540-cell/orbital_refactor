from experiments.walker_modality_shadow_quality_audit import (
    run_walker_modality_shadow_quality_audit,
)


def test_short_walker_shadow_quality_audit_covers_all_modalities():
    result = run_walker_modality_shadow_quality_audit(
        duration=2.0, dt=2.0, anchor_interval_samples=1,
    )
    assert set(result["summary"]) == {"INFRARED", "OPTICAL", "RADAR"}
    assert all(0.0 <= item.shadow_quality <= 1.0
               for item in result["records"])
    assert all(0.0 <= item.feedback_quality <= 1.0
               for item in result["records"])
