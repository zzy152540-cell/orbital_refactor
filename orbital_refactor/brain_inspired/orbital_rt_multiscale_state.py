from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from brain_inspired.multiscale_line_cann import (
    MultiScaleLineCANN,
    MultiScaleLineCANNConfig,
)
from brain_inspired.orbital_rt_adapter import extract_orbital_rt_offset

Array = np.ndarray


@dataclass(frozen=True)
class OrbitalRTMultiScaleConfig:
    radial: MultiScaleLineCANNConfig = field(default_factory=MultiScaleLineCANNConfig)
    along_track: MultiScaleLineCANNConfig = field(
        default_factory=MultiScaleLineCANNConfig,
    )
    maximum_anchor_gain: float = 0.25
    maximum_anchor_innovation: float = 100.0

    def validate(self):
        self.radial.validate()
        self.along_track.validate()
        if (not np.isfinite(self.maximum_anchor_gain)
                or not 0.0 <= self.maximum_anchor_gain <= 1.0):
            raise ValueError("maximum_anchor_gain must lie in [0, 1].")
        if (not np.isfinite(self.maximum_anchor_innovation)
                or self.maximum_anchor_innovation <= 0.0):
            raise ValueError("maximum_anchor_innovation must be positive.")


@dataclass(frozen=True)
class OrbitalRTMultiScaleSnapshot:
    node_id: str
    timestamp: float
    decoded_rt: Array
    source_rt: Array
    residual_rt: Array
    coarse_center_rt: Array
    fine_residual_rt: Array
    cross_scale_disagreement_rt: Array
    coarse_cell_changed: bool
    valid: bool
    saturated_at_boundary: bool
    cue_applied: bool
    anchor_gain: float
    anchor_innovation_norm: float
    anchor_rejected: bool
    radial_coarse_activity: Array
    radial_fine_activity: Array
    along_coarse_activity: Array
    along_fine_activity: Array


class OrbitalRTMultiScaleState:
    """Independent hierarchical R/T sidecar; never updates the estimator."""

    def __init__(self, *, node_id: str, config=None):
        self.node_id = str(node_id)
        self.config = config or OrbitalRTMultiScaleConfig()
        self.config.validate()
        self._radial = MultiScaleLineCANN(self.config.radial)
        self._along = MultiScaleLineCANN(self.config.along_track)
        self._initialized = False

    def initialize(self, *, timestamp, state_eci, reference_state_eci):
        offset = self._offset(timestamp, state_eci, reference_state_eci)
        radial = self._radial.initialize(offset.radial_position, timestamp=timestamp)
        along = self._along.initialize(offset.along_track_position, timestamp=timestamp)
        self._initialized = True
        return self._snapshot(radial, along, offset)

    def predict(self, *, timestamp, predicted_state_eci, reference_state_eci,
                rate_bias_rt=(0.0, 0.0)):
        self._require_initialized()
        offset = self._offset(timestamp, predicted_state_eci, reference_state_eci)
        bias = np.asarray(rate_bias_rt, dtype=float).reshape(-1)
        if bias.shape != (2,) or np.any(~np.isfinite(bias)):
            raise ValueError("rate_bias_rt must be a finite 2-vector.")
        dt = float(timestamp - self._radial.output().timestamp)
        radial = self._radial.step(offset.radial_rate + bias[0], dt)
        along = self._along.step(offset.along_track_rate + bias[1], dt)
        return self._snapshot(radial, along, offset)

    def anchor(self, *, timestamp, posterior_state_eci, reference_state_eci,
               confidence, trusted):
        self._require_initialized()
        if not np.isclose(
            timestamp, self._radial.output().timestamp, rtol=0.0, atol=1e-12,
        ):
            raise ValueError("An anchor must match the current RT state timestamp.")
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Anchor confidence must lie in [0, 1].")
        offset = self._offset(timestamp, posterior_state_eci, reference_state_eci)
        source = np.array([offset.radial_position, offset.along_track_position])
        current = np.array([
            self._radial.output().decoded_value,
            self._along.output().decoded_value,
        ])
        innovation = float(np.linalg.norm(source - current))
        rejected = bool(trusted and innovation > self.config.maximum_anchor_innovation)
        gain = (self.config.maximum_anchor_gain * float(confidence)
                if trusted and not rejected else 0.0)
        if gain > 0.0:
            radial = self._radial.apply_value_cue(source[0], cue_gain=gain)
            along = self._along.apply_value_cue(source[1], cue_gain=gain)
        else:
            radial, along = self._radial.output(), self._along.output()
        return self._snapshot(
            radial, along, offset, cue_applied=gain > 0.0,
            anchor_gain=gain, anchor_innovation_norm=innovation,
            anchor_rejected=rejected,
        )

    def _offset(self, timestamp, state, reference):
        return extract_orbital_rt_offset(
            timestamp=timestamp, state_eci=state,
            reference_state_eci=reference, source_id=self.node_id,
        )

    def _snapshot(self, radial, along, offset, *, cue_applied=False,
                  anchor_gain=0.0, anchor_innovation_norm=0.0,
                  anchor_rejected=False):
        decoded = np.array([radial.decoded_value, along.decoded_value])
        source = np.array([offset.radial_position, offset.along_track_position])
        return OrbitalRTMultiScaleSnapshot(
            node_id=self.node_id, timestamp=radial.timestamp,
            decoded_rt=decoded, source_rt=source, residual_rt=decoded - source,
            coarse_center_rt=np.array([
                radial.coarse_cell_center, along.coarse_cell_center,
            ]),
            fine_residual_rt=np.array([radial.fine_residual, along.fine_residual]),
            cross_scale_disagreement_rt=np.array([
                radial.cross_scale_disagreement,
                along.cross_scale_disagreement,
            ]),
            coarse_cell_changed=bool(
                radial.coarse_cell_changed or along.coarse_cell_changed
            ),
            valid=bool(radial.valid and along.valid),
            saturated_at_boundary=bool(
                radial.saturated_at_boundary or along.saturated_at_boundary
            ),
            cue_applied=bool(cue_applied), anchor_gain=float(anchor_gain),
            anchor_innovation_norm=float(anchor_innovation_norm),
            anchor_rejected=bool(anchor_rejected),
            radial_coarse_activity=radial.coarse_output.neural_activity.copy(),
            radial_fine_activity=radial.fine_output.neural_activity.copy(),
            along_coarse_activity=along.coarse_output.neural_activity.copy(),
            along_fine_activity=along.fine_output.neural_activity.copy(),
        )

    def _require_initialized(self):
        if not self._initialized:
            raise RuntimeError("OrbitalRTMultiScaleState.initialize is required.")
