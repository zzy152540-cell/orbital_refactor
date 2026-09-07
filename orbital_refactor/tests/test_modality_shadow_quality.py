import numpy as np
import pytest

from brain_inspired.modality_shadow_quality import build_modality_shadow_quality
from brain_inspired.navigation_shadow_quality import NavigationShadowQualityHistory
from interfaces.data_objects import ObservationMessage


def _quality(node, direction, radial):
    size = len(direction)
    return NavigationShadowQualityHistory(
        node_id=node, timestamps=np.arange(size, dtype=float),
        direction_discrepancy=np.asarray(direction),
        radial_discrepancy=np.asarray(radial),
        consistency_quality=np.ones(size), freshness_quality=np.ones(size),
        shadow_quality=np.ones(size), feedback_quality=np.ones(size),
        valid=np.ones(size, dtype=bool),
        boundary_saturated=np.zeros(size, dtype=bool),
    )


def _observation(timestamp, modality, frontend=1.0):
    return ObservationMessage(
        message_id=f"a-b-{modality}-{timestamp}", observer_id="a",
        target_id="b", timestamp=timestamp, modality=modality,
        measurement=np.zeros(2), covariance=np.eye(2),
        metadata={"frontend_quality_score": frontend},
    )


def test_angle_and_radar_use_different_navigation_components():
    histories = {
        "a": _quality("a", [0.2], [4.0]),
        "b": _quality("b", [0.2], [4.0]),
    }
    values = build_modality_shadow_quality(
        observations=[_observation(0.0, "OPTICAL"),
                      _observation(0.0, "RADAR")],
        navigation_quality_by_node=histories,
    )
    by_modality = {value.modality: value for value in values}
    assert by_modality["OPTICAL"].navigation_discrepancy == pytest.approx(0.2)
    assert by_modality["RADAR"].navigation_discrepancy > 2.0


def test_frontend_quality_and_link_delay_are_applied():
    histories = {
        "a": _quality("a", [0.0, 0.0], [0.0, 0.0]),
        "b": _quality("b", [0.0, 0.0], [0.0, 0.0]),
    }
    values = build_modality_shadow_quality(
        observations=[_observation(0.0, "INFRARED", 0.8),
                      _observation(1.0, "INFRARED", 0.6)],
        navigation_quality_by_node=histories,
    )
    assert values[0].shadow_quality == pytest.approx(0.8)
    assert values[0].feedback_quality == pytest.approx(0.05)
    assert values[1].feedback_quality == pytest.approx(0.8)


def test_missing_endpoint_quality_is_rejected():
    with pytest.raises(ValueError, match="Missing navigation quality"):
        build_modality_shadow_quality(
            observations=[_observation(0.0, "RADAR")],
            navigation_quality_by_node={"a": _quality("a", [0.0], [0.0])},
        )
