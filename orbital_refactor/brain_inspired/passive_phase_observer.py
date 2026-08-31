from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brain_inspired.ring_cann import CANNOutput, RingCANN, RingCANNConfig


@dataclass(frozen=True)
class PeriodicStateInput:
    """Estimator-derived scalar input for a passive ring-CANN observer."""

    timestamp: float
    phase_rate: float
    phase_hint: float | None = None
    phase_hint_valid: bool = False
    cue_gain: float | None = None
    source_id: str | None = None

    def validate(self) -> None:
        if not np.isfinite(self.timestamp):
            raise ValueError("Periodic-state timestamp must be finite.")
        if not np.isfinite(self.phase_rate):
            raise ValueError("Periodic-state phase rate must be finite.")
        if self.phase_hint_valid:
            if self.phase_hint is None or not np.isfinite(self.phase_hint):
                raise ValueError("A valid phase hint must be finite and present.")
        if self.cue_gain is not None:
            if not np.isfinite(self.cue_gain) or self.cue_gain < 0.0:
                raise ValueError("Cue gain must be finite and nonnegative.")


@dataclass(frozen=True)
class PassiveCANNObservation:
    timestamp: float
    decoded_phase: float
    source_phase_hint: float | None
    phase_residual: float | None
    cue_applied: bool
    bump_concentration: float
    bump_width: float
    valid: bool
    source_id: str | None = None


class PassiveRingCANNObserver:
    """Run a ring CANN beside an estimator without feeding back into it."""

    def __init__(self, config: RingCANNConfig = RingCANNConfig()) -> None:
        self._cann = RingCANN(config)
        self._last_timestamp: float | None = None

    @property
    def initialized(self) -> bool:
        return self._last_timestamp is not None

    @property
    def cann(self) -> RingCANN:
        return self._cann

    def initialize(
        self, *, phase: float, timestamp: float = 0.0,
        source_id: str | None = None,
    ) -> PassiveCANNObservation:
        output = self._cann.reset(phase, timestamp=timestamp)
        self._last_timestamp = float(timestamp)
        return _to_observation(
            output, phase_hint=float(phase), cue_applied=False,
            source_id=source_id,
        )

    def update(self, sample: PeriodicStateInput) -> PassiveCANNObservation:
        if self._last_timestamp is None:
            raise RuntimeError("PassiveRingCANNObserver.initialize is required.")
        sample.validate()
        dt = float(sample.timestamp - self._last_timestamp)
        if dt <= 0.0:
            raise ValueError("Periodic-state timestamps must be strictly increasing.")
        cue_applied = bool(sample.phase_hint_valid)
        phase_hint = float(sample.phase_hint) if cue_applied else None
        output = self._cann.step(
            float(sample.phase_rate), dt, external_phase_hint=phase_hint,
            cue_gain=sample.cue_gain,
        )
        self._last_timestamp = float(sample.timestamp)
        return _to_observation(
            output, phase_hint=phase_hint, cue_applied=cue_applied,
            source_id=sample.source_id,
        )


def _to_observation(
    output: CANNOutput, *, phase_hint: float | None, cue_applied: bool,
    source_id: str | None,
) -> PassiveCANNObservation:
    residual = None
    if phase_hint is not None:
        residual = float(
            (output.decoded_phase - phase_hint + np.pi) % (2.0 * np.pi) - np.pi
        )
    return PassiveCANNObservation(
        timestamp=output.timestamp,
        decoded_phase=output.decoded_phase,
        source_phase_hint=phase_hint,
        phase_residual=residual,
        cue_applied=cue_applied,
        bump_concentration=output.bump_concentration,
        bump_width=output.bump_width,
        valid=output.valid,
        source_id=source_id,
    )
