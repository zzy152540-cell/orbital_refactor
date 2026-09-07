from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from brain_inspired.orbital_phase_adapter import (
    OrbitalPlaneFrame,
    extract_orbital_phase_state,
)
from brain_inspired.ring_cann import CANNOutput, RingCANN, RingCANNConfig

Array = np.ndarray


@dataclass(frozen=True)
class OrbitalDirectionConfig:
    """Per-satellite orbital-phase CANN and bounded anchoring settings."""

    ring: RingCANNConfig = field(default_factory=RingCANNConfig)
    maximum_anchor_gain: float = 0.25
    anchor_relaxation_duration: float = 0.01

    def validate(self) -> None:
        self.ring.validate()
        if (
            not np.isfinite(self.maximum_anchor_gain)
            or not 0.0 <= self.maximum_anchor_gain <= 1.0
        ):
            raise ValueError("maximum_anchor_gain must lie in [0, 1].")
        if (
            not np.isfinite(self.anchor_relaxation_duration)
            or self.anchor_relaxation_duration <= 0.0
        ):
            raise ValueError(
                "anchor_relaxation_duration must be finite and positive."
            )


@dataclass(frozen=True)
class DirectionNavigationSnapshot:
    node_id: str
    timestamp: float
    decoded_phase: float
    bump_concentration: float
    bump_width: float
    valid: bool
    cue_applied: bool
    anchor_gain: float
    last_anchor_timestamp: float
    anchor_age: float
    source_phase: float | None
    phase_residual: float | None
    neural_activity: Array


class OrbitalDirectionState:
    """Persistent per-node direction state outside the probability filter.

    Prediction consumes only a phase rate and elapsed time.  Anchoring is a
    separate endpoint operation and never produces an estimator observation.
    The caller remains responsible for deciding whether a posterior is trusted.
    """

    def __init__(
        self, *, node_id: str, frame: OrbitalPlaneFrame,
        config: OrbitalDirectionConfig | None = None,
    ) -> None:
        self.node_id = str(node_id)
        self.frame = frame
        self.config = config or OrbitalDirectionConfig()
        self.config.validate()
        self._cann = RingCANN(self.config.ring)
        self._initialized = False
        self._last_anchor_timestamp: float | None = None

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize_from_state(
        self, *, timestamp: float, state_eci: Array,
    ) -> DirectionNavigationSnapshot:
        phase = extract_orbital_phase_state(
            timestamp=timestamp, state_eci=state_eci, frame=self.frame,
            source_id=self.node_id,
        )
        output = self._cann.reset(
            phase.argument_of_latitude, timestamp=timestamp,
        )
        self._initialized = True
        self._last_anchor_timestamp = float(timestamp)
        return self._snapshot(
            output, cue_applied=False, anchor_gain=0.0,
            source_phase=phase.argument_of_latitude,
        )

    def predict_from_state(
        self, *, timestamp: float, predicted_state_eci: Array,
        phase_rate_bias: float = 0.0,
    ) -> DirectionNavigationSnapshot:
        """Advance to ``timestamp`` without assimilating a phase cue."""
        self._require_initialized()
        phase = extract_orbital_phase_state(
            timestamp=timestamp, state_eci=predicted_state_eci,
            frame=self.frame, source_id=self.node_id,
        )
        delta_time = float(timestamp - self._cann.timestamp)
        if not np.isfinite(delta_time) or delta_time <= 0.0:
            raise ValueError("Direction timestamps must be strictly increasing.")
        if not np.isfinite(phase_rate_bias):
            raise ValueError("phase_rate_bias must be finite.")
        output = self._cann.step(
            phase.argument_of_latitude_rate + float(phase_rate_bias),
            delta_time,
        )
        return self._snapshot(
            output, cue_applied=False, anchor_gain=0.0,
            source_phase=phase.argument_of_latitude,
        )

    def anchor_from_state(
        self, *, timestamp: float, posterior_state_eci: Array,
        confidence: float, trusted: bool,
    ) -> DirectionNavigationSnapshot:
        """Apply a bounded same-time cue selected by an external trust policy."""
        self._require_initialized()
        if not np.isclose(timestamp, self._cann.timestamp, rtol=0.0, atol=1e-12):
            raise ValueError("An anchor must match the current direction timestamp.")
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Anchor confidence must lie in [0, 1].")
        phase = extract_orbital_phase_state(
            timestamp=timestamp, state_eci=posterior_state_eci,
            frame=self.frame, source_id=self.node_id,
        )
        gain = (
            self.config.maximum_anchor_gain * float(confidence)
            if trusted else 0.0
        )
        if gain > 0.0:
            output = self._cann.apply_phase_cue(
                phase.argument_of_latitude, cue_gain=gain,
                relaxation_duration=self.config.anchor_relaxation_duration,
            )
            self._last_anchor_timestamp = float(timestamp)
        else:
            output = self._cann.output()
        return self._snapshot(
            output, cue_applied=gain > 0.0, anchor_gain=gain,
            source_phase=phase.argument_of_latitude,
        )

    def _snapshot(
        self, output: CANNOutput, *, cue_applied: bool, anchor_gain: float,
        source_phase: float | None,
    ) -> DirectionNavigationSnapshot:
        last_anchor = float(self._last_anchor_timestamp)
        residual = (
            _circular_difference(output.decoded_phase, source_phase)
            if source_phase is not None else None
        )
        return DirectionNavigationSnapshot(
            node_id=self.node_id, timestamp=float(output.timestamp),
            decoded_phase=float(output.decoded_phase),
            bump_concentration=float(output.bump_concentration),
            bump_width=float(output.bump_width), valid=bool(output.valid),
            cue_applied=bool(cue_applied), anchor_gain=float(anchor_gain),
            last_anchor_timestamp=last_anchor,
            anchor_age=float(output.timestamp - last_anchor),
            source_phase=(
                None if source_phase is None else float(source_phase)
            ),
            phase_residual=residual,
            neural_activity=output.neural_activity.copy(),
        )

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "OrbitalDirectionState.initialize_from_state is required."
            )


def _circular_difference(left: float, right: float) -> float:
    return float((left - right + np.pi) % (2.0 * np.pi) - np.pi)
