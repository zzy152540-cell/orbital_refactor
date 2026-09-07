from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from brain_inspired.line_cann import LineCANN, LineCANNConfig
from brain_inspired.orbital_rt_adapter import extract_orbital_rt_offset

Array = np.ndarray


def _default_axis():
    return LineCANNConfig(
        num_neurons=81, minimum_value=-10_000.0,
        maximum_value=10_000.0, tuning_width=250.0,
    )


@dataclass(frozen=True)
class OrbitalRTGridConfig:
    radial: LineCANNConfig = field(default_factory=_default_axis)
    along_track: LineCANNConfig = field(default_factory=_default_axis)
    maximum_anchor_gain: float = 0.25
    rolling_bias_enabled: bool = False
    rolling_bias_gain: float = 0.35
    minimum_bias_baseline: float = 30.0
    maximum_rate_correction: float = 2.0
    bias_rate_deadband: float = 0.01
    maximum_anchor_innovation: float = 100.0

    def validate(self):
        self.radial.validate()
        self.along_track.validate()
        if (not np.isfinite(self.maximum_anchor_gain)
                or not 0.0 <= self.maximum_anchor_gain <= 1.0):
            raise ValueError("maximum_anchor_gain must lie in [0, 1].")
        if (not np.isfinite(self.rolling_bias_gain)
                or not 0.0 <= self.rolling_bias_gain <= 1.0):
            raise ValueError("rolling_bias_gain must lie in [0, 1].")
        if (not np.isfinite(self.minimum_bias_baseline)
                or self.minimum_bias_baseline <= 0.0):
            raise ValueError("minimum_bias_baseline must be positive.")
        if (not np.isfinite(self.maximum_rate_correction)
                or self.maximum_rate_correction < 0.0):
            raise ValueError("maximum_rate_correction must be nonnegative.")
        if (not np.isfinite(self.bias_rate_deadband)
                or self.bias_rate_deadband < 0.0):
            raise ValueError("bias_rate_deadband must be nonnegative.")
        if (not np.isfinite(self.maximum_anchor_innovation)
                or self.maximum_anchor_innovation <= 0.0):
            raise ValueError("maximum_anchor_innovation must be positive.")


@dataclass(frozen=True)
class OrbitalRTGridSnapshot:
    node_id: str
    timestamp: float
    decoded_rt: Array
    source_rt: Array
    residual_rt: Array
    valid: bool
    saturated_at_boundary: bool
    cue_applied: bool
    anchor_gain: float
    anchor_age: float
    rate_correction_rt: Array
    bias_update_applied: bool
    anchor_innovation_norm: float
    anchor_rejected: bool
    anchor_rejection_reason: str
    radial_activity: Array
    along_track_activity: Array


class OrbitalRTGridState:
    """Independent R/T Line CANN prototype around a propagated reference."""

    def __init__(self, *, node_id: str, config: OrbitalRTGridConfig | None = None):
        self.node_id = str(node_id)
        self.config = config or OrbitalRTGridConfig()
        self.config.validate()
        self._radial = LineCANN(self.config.radial)
        self._along_track = LineCANN(self.config.along_track)
        self._initialized = False
        self._last_anchor_timestamp: float | None = None
        self._bias_anchor_timestamp: float | None = None
        self._rate_correction = np.zeros(2)

    def initialize(
        self, *, timestamp: float, state_eci: Array, reference_state_eci: Array,
    ) -> OrbitalRTGridSnapshot:
        offset = extract_orbital_rt_offset(
            timestamp=timestamp, state_eci=state_eci,
            reference_state_eci=reference_state_eci, source_id=self.node_id,
        )
        radial = self._radial.reset(offset.radial_position, timestamp=timestamp)
        along = self._along_track.reset(
            offset.along_track_position, timestamp=timestamp,
        )
        self._initialized = True
        self._last_anchor_timestamp = float(timestamp)
        self._bias_anchor_timestamp = float(timestamp)
        self._rate_correction.fill(0.0)
        return self._snapshot(radial, along, offset, cue_applied=False, gain=0.0,
                              bias_update_applied=False,
                              anchor_innovation_norm=0.0,
                              anchor_rejected=False,
                              anchor_rejection_reason="")

    def predict(
        self, *, timestamp: float, predicted_state_eci: Array,
        reference_state_eci: Array, rate_bias_rt=(0.0, 0.0),
    ) -> OrbitalRTGridSnapshot:
        self._require_initialized()
        offset = extract_orbital_rt_offset(
            timestamp=timestamp, state_eci=predicted_state_eci,
            reference_state_eci=reference_state_eci, source_id=self.node_id,
        )
        bias = np.asarray(rate_bias_rt, dtype=float).reshape(-1)
        if bias.shape != (2,) or np.any(~np.isfinite(bias)):
            raise ValueError("rate_bias_rt must be a finite 2-vector.")
        dt = float(timestamp - self._radial.timestamp)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("RT grid timestamps must be strictly increasing.")
        radial = self._radial.step(
            offset.radial_rate + bias[0] + self._rate_correction[0], dt,
        )
        along = self._along_track.step(
            offset.along_track_rate + bias[1] + self._rate_correction[1], dt,
        )
        return self._snapshot(radial, along, offset, cue_applied=False, gain=0.0,
                              bias_update_applied=False,
                              anchor_innovation_norm=0.0,
                              anchor_rejected=False,
                              anchor_rejection_reason="")

    def anchor(
        self, *, timestamp: float, posterior_state_eci: Array,
        reference_state_eci: Array, confidence: float, trusted: bool,
    ) -> OrbitalRTGridSnapshot:
        self._require_initialized()
        if not np.isclose(timestamp, self._radial.timestamp, rtol=0.0, atol=1e-12):
            raise ValueError("An anchor must match the current RT grid timestamp.")
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Anchor confidence must lie in [0, 1].")
        offset = extract_orbital_rt_offset(
            timestamp=timestamp, state_eci=posterior_state_eci,
            reference_state_eci=reference_state_eci, source_id=self.node_id,
        )
        decoded_before_anchor = np.array([
            self._radial.output().decoded_value,
            self._along_track.output().decoded_value,
        ])
        source_rt = np.array([
            offset.radial_position, offset.along_track_position,
        ])
        innovation_norm = float(np.linalg.norm(source_rt - decoded_before_anchor))
        rejected = bool(
            trusted and innovation_norm > self.config.maximum_anchor_innovation
        )
        accepted = bool(trusted and not rejected)
        gain = self.config.maximum_anchor_gain * confidence if accepted else 0.0
        bias_updated = self._update_rate_correction(
            timestamp=float(timestamp), source_rt=source_rt, trusted=accepted,
        )
        if gain > 0.0:
            radial = self._radial.apply_value_cue(
                offset.radial_position, cue_gain=gain,
            )
            along = self._along_track.apply_value_cue(
                offset.along_track_position, cue_gain=gain,
            )
            self._last_anchor_timestamp = float(timestamp)
        else:
            radial, along = self._radial.output(), self._along_track.output()
        return self._snapshot(radial, along, offset,
                              cue_applied=gain > 0.0, gain=gain,
                              bias_update_applied=bias_updated,
                              anchor_innovation_norm=innovation_norm,
                              anchor_rejected=rejected,
                              anchor_rejection_reason=(
                                  "innovation_limit" if rejected else ""
                              ))

    def _update_rate_correction(self, *, timestamp, source_rt, trusted):
        if not trusted or not self.config.rolling_bias_enabled:
            return False
        if self._bias_anchor_timestamp is None:
            self._bias_anchor_timestamp = timestamp
            return False
        elapsed = timestamp - self._bias_anchor_timestamp
        if elapsed < self.config.minimum_bias_baseline:
            return False
        decoded = np.array([
            self._radial.output().decoded_value,
            self._along_track.output().decoded_value,
        ])
        innovation_rate = (source_rt - decoded) / elapsed
        significant = np.abs(innovation_rate) > self.config.bias_rate_deadband
        if not np.any(significant):
            self._bias_anchor_timestamp = timestamp
            return False
        self._rate_correction[significant] += (
            self.config.rolling_bias_gain * innovation_rate[significant]
        )
        limit = self.config.maximum_rate_correction
        self._rate_correction = np.clip(self._rate_correction, -limit, limit)
        self._bias_anchor_timestamp = timestamp
        return True

    def _snapshot(self, radial, along, offset, *, cue_applied, gain,
                  bias_update_applied, anchor_innovation_norm,
                  anchor_rejected, anchor_rejection_reason):
        decoded = np.array([radial.decoded_value, along.decoded_value])
        source = np.array([offset.radial_position, offset.along_track_position])
        return OrbitalRTGridSnapshot(
            node_id=self.node_id, timestamp=float(radial.timestamp),
            decoded_rt=decoded, source_rt=source, residual_rt=decoded - source,
            valid=bool(radial.valid and along.valid),
            saturated_at_boundary=bool(
                radial.saturated_at_boundary or along.saturated_at_boundary
            ),
            cue_applied=bool(cue_applied), anchor_gain=float(gain),
            anchor_age=float(radial.timestamp - self._last_anchor_timestamp),
            rate_correction_rt=self._rate_correction.copy(),
            bias_update_applied=bool(bias_update_applied),
            anchor_innovation_norm=float(anchor_innovation_norm),
            anchor_rejected=bool(anchor_rejected),
            anchor_rejection_reason=str(anchor_rejection_reason),
            radial_activity=radial.neural_activity.copy(),
            along_track_activity=along.neural_activity.copy(),
        )

    def _require_initialized(self):
        if not self._initialized:
            raise RuntimeError("OrbitalRTGridState.initialize is required.")
