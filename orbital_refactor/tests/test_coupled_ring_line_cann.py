import numpy as np

from brain_inspired.coupled_ring_line_cann import (
    CoupledRingLineCANN, CoupledRingLineCANNConfig,
)


def _difference(actual, expected):
    return (actual - expected + np.pi) % (2.0 * np.pi) - np.pi


def test_coupled_cann_learns_constant_rate_bias_from_phase_cues():
    cann = CoupledRingLineCANN(CoupledRingLineCANNConfig(
        minimum_bias_baseline=10.0,
    ))
    true_rate = np.deg2rad(0.1)
    measured_bias = np.deg2rad(0.003)
    initial = np.deg2rad(20.0)
    cann.initialize(phase=initial)
    output = None
    for index in range(1, 21):
        timestamp = float(index * 2)
        truth = initial + true_rate * timestamp
        output = cann.update(
            timestamp=timestamp,
            measured_phase_rate=true_rate + measured_bias,
            phase_hint=truth if index % 5 == 0 else None,
            phase_hint_valid=index % 5 == 0,
        )
    assert output.decoded_rate_bias < 0.0
    assert output.bias_observation_count > 0
    assert abs(_difference(output.decoded_phase, truth)) < np.deg2rad(0.2)


def test_coupled_cann_rejects_extreme_phase_cue():
    cann = CoupledRingLineCANN()
    cann.initialize(phase=0.0)
    output = cann.update(
        timestamp=2.0, measured_phase_rate=0.0,
        phase_hint=np.deg2rad(20.0), phase_hint_valid=True,
    )
    assert not output.cue_applied
    assert abs(output.decoded_rate_bias) < 1e-15


def test_rolling_anchor_starts_from_first_accepted_cue_and_reanchors():
    cann = CoupledRingLineCANN(CoupledRingLineCANNConfig(
        minimum_bias_baseline=10.0, bias_anchor_mode="rolling_cue",
    ))
    true_rate = np.deg2rad(0.1)
    measured_bias = np.deg2rad(0.003)
    truth_initial = np.deg2rad(20.0)
    cann.initialize(phase=truth_initial + np.deg2rad(2.0))
    outputs = []
    for index in range(1, 16):
        timestamp = float(index * 2)
        truth = truth_initial + true_rate * timestamp
        outputs.append(cann.update(
            timestamp=timestamp,
            measured_phase_rate=true_rate + measured_bias,
            phase_hint=truth if index % 5 == 0 else None,
            phase_hint_valid=index % 5 == 0,
        ))
    assert outputs[4].bias_observation_count == 0
    assert outputs[9].bias_observation_count == 1
    assert outputs[14].bias_observation_count == 2
    assert outputs[-1].decoded_rate_bias < 0.0


def test_hybrid_anchor_trusts_consistent_long_baseline():
    cann = CoupledRingLineCANN(CoupledRingLineCANNConfig(
        minimum_bias_baseline=10.0, bias_anchor_mode="hybrid_dual",
    ))
    rate = np.deg2rad(0.103)
    truth_rate = np.deg2rad(0.1)
    initial = np.deg2rad(20.0)
    cann.initialize(phase=initial)
    cann.update(
        timestamp=10.0, measured_phase_rate=rate,
        phase_hint=initial + truth_rate * 10.0, phase_hint_valid=True,
    )
    output = cann.update(
        timestamp=20.0, measured_phase_rate=rate,
        phase_hint=initial + truth_rate * 20.0, phase_hint_valid=True,
    )
    assert output.long_anchor_trusted is True
    assert output.bias_observation_count == 1


def test_hybrid_anchor_rejects_initial_phase_contamination():
    cann = CoupledRingLineCANN(CoupledRingLineCANNConfig(
        minimum_bias_baseline=10.0, bias_anchor_mode="hybrid_dual",
    ))
    rate = np.deg2rad(0.103)
    truth_rate = np.deg2rad(0.1)
    truth_initial = np.deg2rad(20.0)
    cann.initialize(phase=truth_initial + np.deg2rad(2.0))
    cann.update(
        timestamp=10.0, measured_phase_rate=rate,
        phase_hint=truth_initial + truth_rate * 10.0,
        phase_hint_valid=True,
    )
    output = cann.update(
        timestamp=20.0, measured_phase_rate=rate,
        phase_hint=truth_initial + truth_rate * 20.0,
        phase_hint_valid=True,
    )
    assert output.long_anchor_trusted is False
    assert output.bias_observation_count == 1
    assert output.decoded_rate_bias < 0.0
