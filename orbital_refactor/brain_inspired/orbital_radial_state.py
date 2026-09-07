from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from brain_inspired.line_cann import LineCANN, LineCANNConfig, LineCANNOutput
from brain_inspired.orbital_phase_adapter import (
    OrbitalPlaneFrame,
    extract_orbital_phase_state,
)

Array = np.ndarray


@dataclass(frozen=True)
class OrbitalRadialConfig:
    """Per-satellite radial-displacement Line CANN settings, in metres."""

    line: LineCANNConfig = field(default_factory=lambda: LineCANNConfig(
        num_neurons=81, minimum_value=-20_000.0,
        maximum_value=20_000.0, tuning_width=500.0,
    ))
    maximum_anchor_gain: float = 0.25

    def validate(self) -> None:
        self.line.validate()
        if (not np.isfinite(self.maximum_anchor_gain)
                or not 0.0 <= self.maximum_anchor_gain <= 1.0):
            raise ValueError("maximum_anchor_gain must lie in [0, 1].")


@dataclass(frozen=True)
class RadialNavigationSnapshot:
    node_id: str
    timestamp: float
    decoded_displacement: float
    source_displacement: float | None
    displacement_residual: float | None
    reference_radius: float
    bump_concentration: float
    bump_width: float
    saturated_at_boundary: bool
    valid: bool
    cue_applied: bool
    anchor_gain: float
    last_anchor_timestamp: float
    anchor_age: float
    neural_activity: Array


class OrbitalRadialState:
    """Persistent radial Line CANN outside the probabilistic estimator."""

    def __init__(
        self, *, node_id: str, frame: OrbitalPlaneFrame,
        config: OrbitalRadialConfig | None = None,
    ) -> None:
        self.node_id = str(node_id)
        self.frame = frame
        self.config = config or OrbitalRadialConfig()
        self.config.validate()
        self._cann = LineCANN(self.config.line)
        self._reference_radius: float | None = None
        self._last_anchor_timestamp: float | None = None

    @property
    def initialized(self) -> bool:
        return self._reference_radius is not None

    @property
    def reference_radius(self) -> float:
        self._require_initialized()
        return float(self._reference_radius)

    def initialize_from_state(
        self, *, timestamp: float, state_eci: Array,
    ) -> RadialNavigationSnapshot:
        radial = extract_orbital_phase_state(
            timestamp=timestamp, state_eci=state_eci, frame=self.frame,
            source_id=self.node_id,
        )
        self._reference_radius = radial.in_plane_radius
        self._last_anchor_timestamp = float(timestamp)
        output = self._cann.reset(0.0, timestamp=timestamp)
        return self._snapshot(output, cue_applied=False, anchor_gain=0.0,
                              source_displacement=0.0)

    def predict_from_state(
        self, *, timestamp: float, predicted_state_eci: Array,
        radial_rate_bias: float = 0.0,
    ) -> RadialNavigationSnapshot:
        self._require_initialized()
        radial = extract_orbital_phase_state(
            timestamp=timestamp, state_eci=predicted_state_eci,
            frame=self.frame, source_id=self.node_id,
        )
        dt = float(timestamp - self._cann.timestamp)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("Radial timestamps must be strictly increasing.")
        if not np.isfinite(radial_rate_bias):
            raise ValueError("radial_rate_bias must be finite.")
        output = self._cann.step(
            radial.in_plane_radius_rate + float(radial_rate_bias), dt,
        )
        source = radial.in_plane_radius - self.reference_radius
        return self._snapshot(output, cue_applied=False, anchor_gain=0.0,
                              source_displacement=source)

    def anchor_from_state(
        self, *, timestamp: float, posterior_state_eci: Array,
        confidence: float, trusted: bool,
    ) -> RadialNavigationSnapshot:
        self._require_initialized()
        if not np.isclose(timestamp, self._cann.timestamp, rtol=0.0, atol=1e-12):
            raise ValueError("An anchor must match the current radial timestamp.")
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Anchor confidence must lie in [0, 1].")
        radial = extract_orbital_phase_state(
            timestamp=timestamp, state_eci=posterior_state_eci,
            frame=self.frame, source_id=self.node_id,
        )
        source = radial.in_plane_radius - self.reference_radius
        gain = self.config.maximum_anchor_gain * confidence if trusted else 0.0
        if gain > 0.0:
            output = self._cann.apply_value_cue(source, cue_gain=gain)
            self._last_anchor_timestamp = float(timestamp)
        else:
            output = self._cann.output()
        return self._snapshot(output, cue_applied=gain > 0.0,
                              anchor_gain=gain, source_displacement=source)

    def _snapshot(
        self, output: LineCANNOutput, *, cue_applied: bool,
        anchor_gain: float, source_displacement: float | None,
    ) -> RadialNavigationSnapshot:
        residual = (None if source_displacement is None else
                    float(output.decoded_value - source_displacement))
        return RadialNavigationSnapshot(
            node_id=self.node_id, timestamp=float(output.timestamp),
            decoded_displacement=float(output.decoded_value),
            source_displacement=(None if source_displacement is None else
                                 float(source_displacement)),
            displacement_residual=residual,
            reference_radius=self.reference_radius,
            bump_concentration=float(output.bump_concentration),
            bump_width=float(output.bump_width),
            saturated_at_boundary=bool(output.saturated_at_boundary),
            valid=bool(output.valid), cue_applied=bool(cue_applied),
            anchor_gain=float(anchor_gain),
            last_anchor_timestamp=float(self._last_anchor_timestamp),
            anchor_age=float(output.timestamp - self._last_anchor_timestamp),
            neural_activity=output.neural_activity.copy(),
        )

    def _require_initialized(self) -> None:
        if not self.initialized:
            raise RuntimeError("OrbitalRadialState.initialize_from_state is required.")
