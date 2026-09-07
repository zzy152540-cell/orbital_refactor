import numpy as np
import pytest

from brain_inspired.navigation_shadow_quality import build_navigation_shadow_quality
from brain_inspired.orbital_direction_runner import OrbitalDirectionHistory
from brain_inspired.orbital_radial_runner import OrbitalRadialHistory


def _direction(residual):
    size = len(residual)
    return OrbitalDirectionHistory(
        node_id="sat", timestamps=np.arange(size, dtype=float),
        source_phase=np.zeros(size), predicted_phase=np.zeros(size),
        anchored_phase=np.zeros(size), prediction_residual=np.asarray(residual),
        anchored_residual=np.zeros(size), bump_concentration=np.ones(size),
        bump_width=np.ones(size), valid=np.ones(size, dtype=bool),
        cue_applied=np.zeros(size, dtype=bool), anchor_gain=np.zeros(size),
        anchor_age=np.arange(size, dtype=float),
    )


def _radial(residual, saturated=None):
    size = len(residual)
    return OrbitalRadialHistory(
        node_id="sat", timestamps=np.arange(size, dtype=float),
        source_displacement=np.zeros(size), predicted_displacement=np.zeros(size),
        anchored_displacement=np.zeros(size), prediction_residual=np.asarray(residual),
        anchored_residual=np.zeros(size), bump_concentration=np.ones(size),
        bump_width=np.ones(size),
        saturated_at_boundary=np.zeros(size, dtype=bool) if saturated is None
        else np.asarray(saturated, dtype=bool),
        valid=np.ones(size, dtype=bool), cue_applied=np.zeros(size, dtype=bool),
        anchor_gain=np.zeros(size), anchor_age=np.arange(size, dtype=float),
        reference_radius=7_000_000.0,
    )


def test_shadow_quality_is_bounded_and_feedback_is_delayed():
    quality = build_navigation_shadow_quality(
        direction_by_node={"sat": _direction([0.0, 0.01, 0.02])},
        radial_by_node={"sat": _radial([0.0, 5.0, 20.0])},
    )["sat"]
    assert np.all((quality.shadow_quality >= 0.05)
                  & (quality.shadow_quality <= 1.0))
    assert quality.feedback_quality[0] == pytest.approx(0.05)
    assert np.allclose(quality.feedback_quality[1:], quality.shadow_quality[:-1])
    assert quality.shadow_quality[0] > quality.shadow_quality[-1]


def test_boundary_saturation_forces_minimum_quality():
    quality = build_navigation_shadow_quality(
        direction_by_node={"sat": _direction([0.0, 0.0])},
        radial_by_node={"sat": _radial([0.0, 0.0], [False, True])},
    )["sat"]
    assert quality.shadow_quality[1] == pytest.approx(0.05)


def test_shadow_quality_requires_aligned_nodes():
    with pytest.raises(ValueError, match="identical nodes"):
        build_navigation_shadow_quality(
            direction_by_node={"sat": _direction([0.0])}, radial_by_node={},
        )
