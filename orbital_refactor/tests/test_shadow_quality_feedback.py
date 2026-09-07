import numpy as np
import pytest

from brain_inspired.modality_shadow_quality import ModalityShadowQuality
from cooperative.shadow_quality_feedback import (
    ShadowQualityFeedbackConfig,
    apply_shadow_quality_feedback,
)
from interfaces.data_objects import ObservationMessage


def _observation():
    return ObservationMessage(
        message_id="m", observer_id="a", target_id="b", timestamp=2.0,
        modality="RADAR", measurement=np.zeros(2), covariance=np.eye(2),
    )


def _quality(value):
    return ModalityShadowQuality(
        information_id="m", timestamp=2.0, observer_id="a", target_id="b",
        modality="RADAR", navigation_discrepancy=0.0,
        navigation_quality=1.0, frontend_quality=1.0, shadow_quality=0.9,
        feedback_quality=value, valid=True,
    )


def test_disabled_feedback_preserves_original_observation_objects():
    source = _observation()
    result = apply_shadow_quality_feedback([source], [_quality(0.2)])
    assert result[0] is source


def test_feedback_inflates_covariance_with_a_bound():
    result = apply_shadow_quality_feedback(
        [_observation()], [_quality(0.2)],
        config=ShadowQualityFeedbackConfig(
            enabled=True, maximum_covariance_inflation=3.0,
        ),
    )[0]
    assert np.allclose(result.covariance, 3.0 * np.eye(2))
    assert result.metadata["shadow_feedback_delay_epochs"] == 1


def test_optional_rejection_is_disabled_by_default():
    accepted = apply_shadow_quality_feedback(
        [_observation()], [_quality(0.05)],
        config=ShadowQualityFeedbackConfig(enabled=True),
    )[0]
    rejected = apply_shadow_quality_feedback(
        [_observation()], [_quality(0.05)],
        config=ShadowQualityFeedbackConfig(
            enabled=True, rejection_threshold=0.1,
        ),
    )[0]
    assert accepted.valid_flag
    assert not rejected.valid_flag


def test_feedback_requires_matching_information_id():
    with pytest.raises(ValueError, match="Missing shadow quality"):
        apply_shadow_quality_feedback(
            [_observation()], [],
            config=ShadowQualityFeedbackConfig(enabled=True),
        )
