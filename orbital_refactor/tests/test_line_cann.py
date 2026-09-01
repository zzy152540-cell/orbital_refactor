import numpy as np

from brain_inspired.line_cann import (
    LineCANN, LineCANNConfig, decode_line_activity,
    decode_line_activity_peak_fit,
)


def test_line_decoder_recovers_gaussian_center():
    preferred = np.linspace(-1.0, 1.0, 201)
    firing = 1.0 + 39.0 * np.exp(-0.5 * ((preferred - 0.2) / 0.08) ** 2)
    value, concentration, width = decode_line_activity(
        firing, preferred, background_firing_rate=1.0,
    )
    assert abs(value - 0.2) < 1e-10
    assert concentration > 0.9
    assert 0.07 < width < 0.09


def test_line_cann_holds_static_value_and_tracks_rate():
    cann = LineCANN()
    initial = cann.reset(0.2)
    held = cann.step(0.0, 10.0)
    moved = cann.step(0.03, 10.0)
    assert abs(held.decoded_value - initial.decoded_value) < 1e-12
    assert abs(moved.decoded_value - 0.5) < 1e-5
    assert moved.valid


def test_line_cann_clamps_instead_of_wrapping_at_boundary():
    cann = LineCANN(LineCANNConfig(minimum_value=-1.0, maximum_value=1.0))
    cann.reset(0.9)
    output = cann.step(0.2, 1.0)
    assert output.saturated_at_boundary
    assert abs(output.decoded_value - 1.0) < 1e-10
    assert output.decoded_value != -0.9
    peak_fit = decode_line_activity_peak_fit(
        output.neural_activity, cann.preferred_value,
        background_firing_rate=cann.config.background_firing_rate,
    )
    assert abs(peak_fit - 1.0) < 1e-10


def test_line_cue_reduces_offset_without_advancing_time():
    cann = LineCANN()
    cann.reset(0.5, timestamp=3.0)
    before = abs(cann.output().decoded_value)
    output = cann.apply_value_cue(0.0, cue_gain=0.5)
    assert abs(output.decoded_value) < before
    assert output.timestamp == 3.0
