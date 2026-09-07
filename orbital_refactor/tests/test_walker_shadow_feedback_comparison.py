from cooperative.shadow_quality_feedback import ShadowQualityFeedbackConfig
from experiments.walker_shadow_feedback_comparison import (
    run_walker_shadow_feedback_comparison,
)


def test_walker_shadow_feedback_has_strict_disabled_path():
    result = run_walker_shadow_feedback_comparison(
        duration=2.0, dt=2.0,
        feedback_config=ShadowQualityFeedbackConfig(
            enabled=True, maximum_covariance_inflation=2.0,
        ),
    )
    assert result["disabled_identity_preserved"]
    assert result["inflation"]["maximum"] <= 2.0
    assert result["inflation"]["rejected_count"] == 0
    assert result["baseline"]["nis_count"] > 0
    assert result["feedback"]["nis_count"] > 0
