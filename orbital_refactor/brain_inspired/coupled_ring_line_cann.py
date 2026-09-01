from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from brain_inspired.line_cann import LineCANN, LineCANNConfig, LineCANNOutput
from brain_inspired.ring_cann import CANNOutput, RingCANN, RingCANNConfig


@dataclass(frozen=True)
class CoupledRingLineCANNConfig:
    ring: RingCANNConfig = field(default_factory=RingCANNConfig)
    line: LineCANNConfig = field(default_factory=lambda: LineCANNConfig(
        minimum_value=np.deg2rad(-0.05),
        maximum_value=np.deg2rad(0.05),
        tuning_width=np.deg2rad(0.003),
        cue_gain=0.1,
    ))
    phase_cue_gain: float = 0.05
    phase_gate: float = np.deg2rad(3.0)
    minimum_bias_baseline: float = 120.0
    bias_consistency_scale: float = np.deg2rad(0.01)
    anchor_agreement_scale: float = np.deg2rad(0.002)
    bias_anchor_mode: Literal[
        "fixed_initial", "rolling_cue", "hybrid_dual",
    ] = "fixed_initial"

    def validate(self) -> None:
        self.ring.validate()
        self.line.validate()
        if not np.isfinite(self.phase_cue_gain) or self.phase_cue_gain < 0.0:
            raise ValueError("Coupled phase cue gain must be finite and nonnegative.")
        if not np.isfinite(self.phase_gate) or self.phase_gate <= 0.0:
            raise ValueError("Coupled phase gate must be finite and positive.")
        if not np.isfinite(self.minimum_bias_baseline) or self.minimum_bias_baseline <= 0.0:
            raise ValueError("Bias baseline must be finite and positive.")
        if not np.isfinite(self.bias_consistency_scale) or self.bias_consistency_scale <= 0.0:
            raise ValueError("Bias consistency scale must be finite and positive.")
        if not np.isfinite(self.anchor_agreement_scale) or self.anchor_agreement_scale <= 0.0:
            raise ValueError("Anchor agreement scale must be finite and positive.")
        if self.bias_anchor_mode not in {
            "fixed_initial", "rolling_cue", "hybrid_dual",
        }:
            raise ValueError("Unknown coupled CANN bias anchor mode.")


@dataclass(frozen=True)
class CoupledRingLineCANNOutput:
    timestamp: float
    decoded_phase: float
    decoded_rate_bias: float
    phase_innovation: float | None
    cue_applied: bool
    bias_cue_applied: bool
    bias_confidence: float
    bias_observation_count: int
    long_anchor_trusted: bool | None
    ring_output: CANNOutput
    line_output: LineCANNOutput


class CoupledRingLineCANN:
    """Ring phase attractor coupled to a bounded line bias attractor."""

    def __init__(
        self, config: CoupledRingLineCANNConfig = CoupledRingLineCANNConfig(),
    ) -> None:
        config.validate()
        self.config = config
        self.ring = RingCANN(config.ring)
        self.line = LineCANN(config.line)
        self._last_timestamp: float | None = None
        self._last_cue_timestamp: float | None = None
        self._last_cue_phase: float | None = None
        self._integrated_measured_rate = 0.0
        self._cumulative_measured_phase = 0.0
        self._bias_anchor_timestamp: float | None = None
        self._bias_anchor_phase: float | None = None
        self._bias_anchor_integral: float | None = None
        self._rolling_anchor_timestamp: float | None = None
        self._rolling_anchor_phase: float | None = None
        self._rolling_anchor_integral: float | None = None
        self._long_anchor_trusted: bool | None = None
        self._bias_observation_count = 0

    def initialize(
        self, *, phase: float, timestamp: float = 0.0,
        rate_bias: float = 0.0,
    ) -> CoupledRingLineCANNOutput:
        ring = self.ring.reset(phase, timestamp=timestamp)
        line = self.line.reset(rate_bias, timestamp=timestamp)
        self._last_timestamp = float(timestamp)
        self._last_cue_timestamp = float(timestamp)
        self._last_cue_phase = float(phase) % (2.0 * np.pi)
        self._integrated_measured_rate = 0.0
        self._cumulative_measured_phase = 0.0
        if self.config.bias_anchor_mode in {"fixed_initial", "hybrid_dual"}:
            self._set_bias_anchor(timestamp, phase)
        else:
            self._clear_bias_anchor()
        self._clear_rolling_anchor()
        self._long_anchor_trusted = None
        self._bias_observation_count = 0
        return self._output(
            ring, line, innovation=None, cue_applied=False,
            bias_cue_applied=False, bias_confidence=0.0,
        )

    def update(
        self, *, timestamp: float, measured_phase_rate: float,
        phase_hint: float | None = None, phase_hint_valid: bool = False,
    ) -> CoupledRingLineCANNOutput:
        if self._last_timestamp is None:
            raise RuntimeError("CoupledRingLineCANN.initialize is required.")
        dt = float(timestamp - self._last_timestamp)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("Coupled CANN timestamps must be strictly increasing.")
        if not np.isfinite(measured_phase_rate):
            raise ValueError("Measured phase rate must be finite.")
        self._integrated_measured_rate += float(measured_phase_rate) * dt
        self._cumulative_measured_phase += float(measured_phase_rate) * dt
        bias = self.line.output().decoded_value
        ring = self.ring.step(float(measured_phase_rate) + bias, dt)
        line = self.line.step(0.0, dt)
        innovation = None
        cue_applied = False
        bias_cue_applied = False
        bias_confidence = 0.0
        if phase_hint_valid:
            if phase_hint is None or not np.isfinite(phase_hint):
                raise ValueError("A valid coupled phase hint must be finite.")
            innovation = _wrapped_difference(float(phase_hint), ring.decoded_phase)
            if abs(innovation) <= self.config.phase_gate:
                cue_applied = True
                ring = self.ring.apply_phase_cue(
                    float(phase_hint), cue_gain=self.config.phase_cue_gain,
                )
                bias_observation = self._select_bias_observation(
                    timestamp=float(timestamp), phase=float(phase_hint),
                )
                if bias_observation is not None:
                    discrepancy = abs(bias_observation - bias)
                    bias_confidence = float(np.exp(
                        -0.5 * (
                            discrepancy / self.config.bias_consistency_scale
                        ) ** 2
                    ))
                    effective_gain = self.config.line.cue_gain * bias_confidence
                    line = self.line.apply_value_cue(
                        bias_observation, cue_gain=effective_gain,
                    )
                    bias_cue_applied = effective_gain > 0.0
                    self._bias_observation_count += 1
                self._last_cue_timestamp = float(timestamp)
                self._last_cue_phase = float(phase_hint) % (2.0 * np.pi)
                self._integrated_measured_rate = 0.0
        self._last_timestamp = float(timestamp)
        return self._output(
            ring, line, innovation=innovation, cue_applied=cue_applied,
            bias_cue_applied=bias_cue_applied,
            bias_confidence=bias_confidence,
        )

    def _set_bias_anchor(self, timestamp: float, phase: float) -> None:
        self._bias_anchor_timestamp = float(timestamp)
        self._bias_anchor_phase = float(phase) % (2.0 * np.pi)
        self._bias_anchor_integral = float(self._cumulative_measured_phase)

    def _clear_bias_anchor(self) -> None:
        self._bias_anchor_timestamp = None
        self._bias_anchor_phase = None
        self._bias_anchor_integral = None

    def _set_rolling_anchor(self, timestamp: float, phase: float) -> None:
        self._rolling_anchor_timestamp = float(timestamp)
        self._rolling_anchor_phase = float(phase) % (2.0 * np.pi)
        self._rolling_anchor_integral = float(self._cumulative_measured_phase)

    def _clear_rolling_anchor(self) -> None:
        self._rolling_anchor_timestamp = None
        self._rolling_anchor_phase = None
        self._rolling_anchor_integral = None

    def _anchor_observation(
        self, timestamp: float, phase: float, *, anchor_timestamp: float,
        anchor_phase: float, anchor_integral: float,
    ) -> float:
        interval = float(timestamp - anchor_timestamp)
        measured_delta = self._cumulative_measured_phase - anchor_integral
        predicted = (anchor_phase + measured_delta) % (2.0 * np.pi)
        return _wrapped_difference(phase, predicted) / interval

    def _select_bias_observation(
        self, *, timestamp: float, phase: float,
    ) -> float | None:
        mode = self.config.bias_anchor_mode
        if mode == "fixed_initial":
            interval = float(timestamp - self._bias_anchor_timestamp)
            if interval < self.config.minimum_bias_baseline:
                return None
            return self._anchor_observation(
                timestamp, phase, anchor_timestamp=self._bias_anchor_timestamp,
                anchor_phase=self._bias_anchor_phase,
                anchor_integral=self._bias_anchor_integral,
            )
        if mode == "rolling_cue":
            if self._bias_anchor_timestamp is None:
                self._set_bias_anchor(timestamp, phase)
                return None
            interval = float(timestamp - self._bias_anchor_timestamp)
            if interval < self.config.minimum_bias_baseline:
                return None
            observation = self._anchor_observation(
                timestamp, phase, anchor_timestamp=self._bias_anchor_timestamp,
                anchor_phase=self._bias_anchor_phase,
                anchor_integral=self._bias_anchor_integral,
            )
            self._set_bias_anchor(timestamp, phase)
            return observation

        if self._rolling_anchor_timestamp is None:
            self._set_rolling_anchor(timestamp, phase)
            return None
        rolling_interval = float(timestamp - self._rolling_anchor_timestamp)
        if rolling_interval >= self.config.minimum_bias_baseline:
            long_observation = self._anchor_observation(
                timestamp, phase, anchor_timestamp=self._bias_anchor_timestamp,
                anchor_phase=self._bias_anchor_phase,
                anchor_integral=self._bias_anchor_integral,
            )
            rolling_observation = self._anchor_observation(
                timestamp, phase,
                anchor_timestamp=self._rolling_anchor_timestamp,
                anchor_phase=self._rolling_anchor_phase,
                anchor_integral=self._rolling_anchor_integral,
            )
            anchors_agree = bool(abs(
                long_observation - rolling_observation
            ) <= self.config.anchor_agreement_scale)
            if self._long_anchor_trusted is None or not anchors_agree:
                self._long_anchor_trusted = anchors_agree
            self._set_rolling_anchor(timestamp, phase)
            return (
                long_observation if self._long_anchor_trusted
                else rolling_observation
            )
        if self._long_anchor_trusted:
            return self._anchor_observation(
                timestamp, phase, anchor_timestamp=self._bias_anchor_timestamp,
                anchor_phase=self._bias_anchor_phase,
                anchor_integral=self._bias_anchor_integral,
            )
        return None

    def _output(
        self, ring, line, *, innovation, cue_applied,
        bias_cue_applied, bias_confidence,
    ):
        return CoupledRingLineCANNOutput(
            timestamp=float(ring.timestamp), decoded_phase=ring.decoded_phase,
            decoded_rate_bias=line.decoded_value,
            phase_innovation=innovation, cue_applied=bool(cue_applied),
            bias_cue_applied=bool(bias_cue_applied),
            bias_confidence=float(bias_confidence),
            bias_observation_count=int(self._bias_observation_count),
            long_anchor_trusted=self._long_anchor_trusted,
            ring_output=ring, line_output=line,
        )


def _wrapped_difference(actual: float, expected: float) -> float:
    return float((actual - expected + np.pi) % (2.0 * np.pi) - np.pi)
