import numpy as np
import pytest

from brain_inspired.orbital_rt_grid_state import (
    OrbitalRTGridConfig,
    OrbitalRTGridState,
)
from orbital_core.dynamics import rk4_step_absolute


def _state():
    return np.array([7e6, 0.0, 0.0, 0.0, 7500.0, 0.0])


def test_rt_grid_predicts_and_soft_anchors_without_filter_feedback():
    reference = _state()
    state = reference.copy()
    state[0] += 100.0
    state[1] += 200.0
    grid = OrbitalRTGridState(node_id="sat")
    first = grid.initialize(
        timestamp=0.0, state_eci=state, reference_state_eci=reference,
    )
    next_reference = rk4_step_absolute(reference, 2.0)
    next_state = rk4_step_absolute(state, 2.0)
    predicted = grid.predict(
        timestamp=2.0, predicted_state_eci=next_state,
        reference_state_eci=next_reference,
    )
    anchored = grid.anchor(
        timestamp=2.0, posterior_state_eci=next_state,
        reference_state_eci=next_reference, confidence=0.8, trusted=True,
    )
    assert first.valid and predicted.valid and anchored.valid
    assert anchored.cue_applied
    assert anchored.anchor_gain == pytest.approx(0.2)
    assert anchored.anchor_age == 0.0
    assert not anchored.saturated_at_boundary


def test_rt_grid_reports_rate_bias_boundary_saturation():
    reference = _state()
    grid = OrbitalRTGridState(node_id="sat")
    grid.initialize(timestamp=0.0, state_eci=reference,
                    reference_state_eci=reference)
    output = grid.predict(
        timestamp=1.0, predicted_state_eci=reference,
        reference_state_eci=reference, rate_bias_rt=(1e6, 0.0),
    )
    assert output.saturated_at_boundary


def test_rt_grid_rolling_anchor_learns_opposite_rate_correction():
    reference = _state()
    grid = OrbitalRTGridState(
        node_id="sat",
        config=OrbitalRTGridConfig(
            rolling_bias_enabled=True, minimum_bias_baseline=10.0,
        ),
    )
    grid.initialize(timestamp=0.0, state_eci=reference,
                    reference_state_eci=reference)
    grid.predict(
        timestamp=10.0, predicted_state_eci=reference,
        reference_state_eci=reference, rate_bias_rt=(0.2, 0.0),
    )
    anchored = grid.anchor(
        timestamp=10.0, posterior_state_eci=reference,
        reference_state_eci=reference, confidence=1.0, trusted=True,
    )
    assert anchored.bias_update_applied
    assert anchored.rate_correction_rt[0] < 0.0
    assert abs(anchored.rate_correction_rt[0]) <= 2.0
