import numpy as np

from brain_inspired.ring_cann import (
    RingCANN,
    RingCANNConfig,
    decode_ring_activity,
    periodic_spectral_derivative,
)


def _circular_error(actual, expected):
    return (actual - expected + np.pi) % (2.0 * np.pi) - np.pi


def test_population_decoder_respects_ring_boundary():
    phase = 2.0 * np.pi * np.arange(180) / 180
    center = np.deg2rad(359.0)
    activity = 1.0 + 39.0 * np.exp(8.0 * (np.cos(phase - center) - 1.0))
    decoded, concentration, width = decode_ring_activity(activity, phase)
    assert abs(_circular_error(decoded, center)) < np.deg2rad(0.05)
    assert 0.7 < concentration < 0.9
    assert 0.0 < width < 1.0


def test_fft_recurrent_input_matches_matrix_reference():
    cann = RingCANN(RingCANNConfig(num_neurons=90))
    firing = 2.0 + np.sin(cann.preferred_phase) + 0.2 * np.cos(
        4.0 * cann.preferred_phase
    )
    for gamma in (0.0, -0.001, 0.001):
        assert np.allclose(
            cann.recurrent_input(firing, gamma),
            cann.recurrent_input_matrix_reference(firing, gamma),
            atol=1.0e-12,
        )


def test_static_and_derivative_kernels_have_required_symmetry():
    cann = RingCANN()
    negative = (-np.arange(cann.config.num_neurons)) % cann.config.num_neurons
    assert np.allclose(cann.static_kernel, cann.static_kernel[negative], atol=1e-13)
    assert np.allclose(
        cann.derivative_kernel, -cann.derivative_kernel[negative], atol=1e-13,
    )
    assert cann.static_kernel[0] > 0.0
    assert np.min(cann.static_kernel) < 0.0


def test_spectral_derivative_matches_periodic_analytic_function():
    phase = 2.0 * np.pi * np.arange(180) / 180
    values = np.cos(3.0 * phase)
    expected = -3.0 * np.sin(3.0 * phase)
    assert np.allclose(
        periodic_spectral_derivative(values), expected, atol=2.0e-13,
    )


def test_reset_forms_stable_single_bump_without_motion():
    cann = RingCANN()
    initial = np.deg2rad(37.0)
    reset = cann.reset(initial)
    assert reset.valid
    assert abs(_circular_error(reset.decoded_phase, initial)) < np.deg2rad(0.1)
    assert 0.70 < reset.bump_concentration < 0.90
    propagated = cann.step(0.0, 10.0)
    assert abs(
        _circular_error(propagated.decoded_phase, reset.decoded_phase)
    ) < np.deg2rad(0.1)
    assert abs(
        propagated.bump_width - reset.bump_width
    ) / reset.bump_width < 0.02


def test_positive_and_negative_velocity_move_bump_in_expected_direction():
    rate = np.deg2rad(10.0)
    for sign in (1.0, -1.0):
        cann = RingCANN()
        initial = np.deg2rad(120.0)
        cann.reset(initial)
        output = cann.step(sign * rate, 20.0)
        expected = initial + sign * rate * 20.0
        error = abs(_circular_error(output.decoded_phase, expected))
        assert error < np.deg2rad(1.0)
        assert error / (abs(rate) * 20.0) < 0.02
        assert 0.70 < output.bump_concentration < 0.90


def test_motion_wraps_from_359_degrees_through_zero():
    cann = RingCANN()
    initial = np.deg2rad(359.0)
    cann.reset(initial)
    output = cann.step(np.deg2rad(5.0), 1.0)
    assert abs(_circular_error(output.decoded_phase, np.deg2rad(4.0))) < np.deg2rad(0.5)


def test_external_cue_reduces_phase_offset():
    cann = RingCANN()
    true_phase = np.deg2rad(90.0)
    cann.reset(true_phase + np.deg2rad(20.0))
    before = abs(_circular_error(cann.output().decoded_phase, true_phase))
    after = cann.step(
        0.0, 0.5, external_phase_hint=true_phase,
    )
    assert abs(_circular_error(after.decoded_phase, true_phase)) < 0.3 * before
    assert after.valid


def test_discrete_phase_cue_does_not_advance_physical_timestamp():
    cann = RingCANN()
    cann.reset(np.deg2rad(30.0), timestamp=4.0)
    before = cann.timestamp
    output = cann.apply_phase_cue(np.deg2rad(31.0))
    assert output.timestamp == before
    assert output.internal_step_count > 0


def test_step_uses_exact_physical_time_coverage():
    cann = RingCANN(RingCANNConfig(internal_dt=0.001))
    cann.reset(0.0)
    output = cann.step(0.0, 0.0025)
    assert output.internal_step_count == 3
    assert output.timestamp == 0.0025
