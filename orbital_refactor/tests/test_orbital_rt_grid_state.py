import numpy as np
import pytest

from brain_inspired.orbital_rt_grid_state import (
    OrbitalRTGridConfig,
    OrbitalRTGridState,
)
from brain_inspired.line_cann import LineCANNConfig
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


def test_rt_grid_rejects_large_anchor_before_cue_and_bias_update():
    reference = _state()
    outlier = reference.copy()
    outlier[0] += 500.0
    grid = OrbitalRTGridState(
        node_id="sat",
        config=OrbitalRTGridConfig(
            rolling_bias_enabled=True, minimum_bias_baseline=1.0,
            maximum_anchor_innovation=100.0,
        ),
    )
    grid.initialize(timestamp=0.0, state_eci=reference,
                    reference_state_eci=reference)
    grid.predict(timestamp=2.0, predicted_state_eci=reference,
                 reference_state_eci=reference)
    rejected = grid.anchor(
        timestamp=2.0, posterior_state_eci=outlier,
        reference_state_eci=reference, confidence=1.0, trusted=True,
    )
    assert rejected.anchor_rejected
    assert rejected.anchor_rejection_reason == "innovation_limit"
    assert not rejected.cue_applied
    assert not rejected.bias_update_applied
    assert np.allclose(rejected.rate_correction_rt, 0.0)


def test_rt_grid_rolling_reference_preserves_physical_output_on_rebase():
    reference = _state()
    axis = LineCANNConfig(
        num_neurons=81, minimum_value=-10.0,
        maximum_value=10.0, tuning_width=0.25,
    )
    grid = OrbitalRTGridState(
        node_id="sat",
        config=OrbitalRTGridConfig(
            radial=axis, along_track=axis,
            rolling_reference_enabled=True,
            rolling_reference_trigger_fraction=0.5,
        ),
    )
    first = grid.initialize(timestamp=0.0, state_eci=reference,
                            reference_state_eci=reference)
    output = grid.predict(
        timestamp=1.0, predicted_state_eci=reference,
        reference_state_eci=reference, rate_bias_rt=(6.0, 0.0),
    )
    assert first.decoded_rt[0] == 0.0
    assert output.reference_rebased.tolist() == [True, False]
    assert output.decoded_rt[0] == pytest.approx(6.0, abs=1e-6)
    assert output.reference_origin_rt[0] == pytest.approx(6.0, abs=1e-6)
    assert output.reference_rebase_count.tolist() == [1, 0]
    assert not output.saturated_at_boundary


def test_disabled_rolling_reference_is_strictly_equivalent_to_default():
    reference = _state()
    state = reference.copy()
    state[:2] += [100.0, 200.0]
    default = OrbitalRTGridState(node_id="sat")
    explicit_off = OrbitalRTGridState(
        node_id="sat",
        config=OrbitalRTGridConfig(rolling_reference_enabled=False),
    )
    first_outputs = [tracker.initialize(
        timestamp=0.0, state_eci=state, reference_state_eci=reference,
    ) for tracker in (default, explicit_off)]
    next_reference = rk4_step_absolute(reference, 2.0)
    next_state = rk4_step_absolute(state, 2.0)
    predicted_outputs = [tracker.predict(
        timestamp=2.0, predicted_state_eci=next_state,
        reference_state_eci=next_reference, rate_bias_rt=(0.1, 0.2),
    ) for tracker in (default, explicit_off)]
    anchor_outputs = [tracker.anchor(
        timestamp=2.0, posterior_state_eci=next_state,
        reference_state_eci=next_reference, confidence=0.8, trusted=True,
    ) for tracker in (default, explicit_off)]
    for pair in (first_outputs, predicted_outputs, anchor_outputs):
        assert np.array_equal(pair[0].decoded_rt, pair[1].decoded_rt)
        assert np.array_equal(pair[0].residual_rt, pair[1].residual_rt)
        assert np.array_equal(pair[0].radial_activity, pair[1].radial_activity)
        assert np.array_equal(
            pair[0].along_track_activity, pair[1].along_track_activity,
        )
