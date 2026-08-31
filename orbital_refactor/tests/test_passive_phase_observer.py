import numpy as np
import pytest

from brain_inspired.passive_phase_observer import (
    PassiveRingCANNObserver,
    PeriodicStateInput,
)


def _circular_error(actual, expected):
    return (actual - expected + np.pi) % (2.0 * np.pi) - np.pi


def test_passive_observer_propagates_estimator_rate_without_feedback():
    observer = PassiveRingCANNObserver()
    initial = np.deg2rad(30.0)
    initialized = observer.initialize(
        phase=initial, timestamp=10.0, source_id="sat-01",
    )
    result = observer.update(PeriodicStateInput(
        timestamp=12.0, phase_rate=np.deg2rad(5.0), source_id="sat-01",
    ))
    assert initialized.cue_applied is False
    assert result.source_phase_hint is None
    assert result.phase_residual is None
    assert result.source_id == "sat-01"
    assert result.timestamp == 12.0
    assert abs(_circular_error(result.decoded_phase, np.deg2rad(40.0))) < np.deg2rad(0.5)


def test_passive_observer_applies_only_explicitly_valid_hint():
    observer = PassiveRingCANNObserver()
    observer.initialize(phase=np.deg2rad(110.0))
    ignored = observer.update(PeriodicStateInput(
        timestamp=0.1, phase_rate=0.0, phase_hint=np.deg2rad(90.0),
        phase_hint_valid=False,
    ))
    corrected = observer.update(PeriodicStateInput(
        timestamp=0.6, phase_rate=0.0, phase_hint=np.deg2rad(90.0),
        phase_hint_valid=True,
    ))
    assert ignored.cue_applied is False
    assert ignored.phase_residual is None
    assert corrected.cue_applied is True
    assert abs(corrected.phase_residual) < np.deg2rad(2.0)


def test_passive_observer_rejects_missing_initialization_and_time_reversal():
    observer = PassiveRingCANNObserver()
    with pytest.raises(RuntimeError, match="initialize"):
        observer.update(PeriodicStateInput(timestamp=1.0, phase_rate=0.0))
    observer.initialize(phase=0.0, timestamp=1.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        observer.update(PeriodicStateInput(timestamp=1.0, phase_rate=0.0))


def test_periodic_state_input_requires_valid_hint_to_be_present():
    sample = PeriodicStateInput(
        timestamp=1.0, phase_rate=0.0, phase_hint_valid=True,
    )
    with pytest.raises(ValueError, match="finite and present"):
        sample.validate()
