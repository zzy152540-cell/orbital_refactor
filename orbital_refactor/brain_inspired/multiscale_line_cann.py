from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from brain_inspired.line_cann import LineCANN, LineCANNConfig, LineCANNOutput

Array = np.ndarray


def _coarse_config():
    return LineCANNConfig(
        num_neurons=81, minimum_value=-50_000.0,
        maximum_value=50_000.0, tuning_width=1_250.0,
    )


def _fine_config():
    return LineCANNConfig(
        num_neurons=81, minimum_value=-1_000.0,
        maximum_value=1_000.0, tuning_width=25.0,
    )


@dataclass(frozen=True)
class MultiScaleLineCANNConfig:
    coarse: LineCANNConfig = field(default_factory=_coarse_config)
    fine: LineCANNConfig = field(default_factory=_fine_config)

    def validate(self):
        self.coarse.validate()
        self.fine.validate()
        coarse_spacing = (
            (self.coarse.maximum_value - self.coarse.minimum_value)
            / (self.coarse.num_neurons - 1)
        )
        fine_half_span = 0.5 * (
            self.fine.maximum_value - self.fine.minimum_value
        )
        if fine_half_span < 0.5 * coarse_spacing:
            raise ValueError(
                "Fine scale must cover at least half one coarse-cell spacing."
            )


@dataclass(frozen=True)
class MultiScaleLineCANNOutput:
    timestamp: float
    decoded_value: float
    coarse_cell_center: float
    fine_residual: float
    cross_scale_disagreement: float
    coarse_cell_changed: bool
    saturated_at_boundary: bool
    valid: bool
    coarse_output: LineCANNOutput
    fine_output: LineCANNOutput


class MultiScaleLineCANN:
    """Hierarchical coarse-cell and local-residual Line CANN prototype."""

    def __init__(
        self, config: MultiScaleLineCANNConfig = MultiScaleLineCANNConfig(),
    ):
        config.validate()
        self.config = config
        self.coarse = LineCANN(config.coarse)
        self.fine = LineCANN(config.fine)
        self._coarse_center = 0.0
        self._initialized = False

    def initialize(self, value: float, *, timestamp: float = 0.0):
        coarse = self.coarse.reset(value, timestamp=timestamp)
        self._coarse_center = self._nearest_coarse_center(coarse.decoded_value)
        fine = self.fine.reset(
            float(value) - self._coarse_center, timestamp=timestamp,
        )
        self._initialized = True
        return self._output(coarse, fine, coarse_cell_changed=False)

    def step(self, value_rate: float, dt: float):
        self._require_initialized()
        if not np.isfinite(value_rate) or not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("Rate and step duration must be finite and dt positive.")
        previous_value = self.output().decoded_value
        target_value = previous_value + float(value_rate) * float(dt)
        coarse = self.coarse.step(value_rate, dt)
        new_center = self._nearest_coarse_center(coarse.decoded_value)
        changed = not np.isclose(new_center, self._coarse_center)
        if changed:
            self._coarse_center = new_center
            fine = self.fine.reset(
                target_value - new_center, timestamp=coarse.timestamp,
            )
        else:
            fine = self.fine.step(value_rate, dt)
        return self._output(coarse, fine, coarse_cell_changed=changed)

    def apply_value_cue(self, value_hint: float, *, cue_gain: float = 0.25):
        self._require_initialized()
        if not np.isfinite(value_hint) or not np.isfinite(cue_gain):
            raise ValueError("Cue value and gain must be finite.")
        if not 0.0 <= cue_gain <= 1.0:
            raise ValueError("Cue gain must lie in [0, 1].")
        current_value = self.output().decoded_value
        coarse = self.coarse.apply_value_cue(value_hint, cue_gain=cue_gain)
        new_center = self._nearest_coarse_center(coarse.decoded_value)
        changed = not np.isclose(new_center, self._coarse_center)
        if changed:
            self._coarse_center = new_center
            self.fine.reset(
                current_value - new_center, timestamp=coarse.timestamp,
            )
        fine = self.fine.apply_value_cue(
            float(value_hint) - new_center, cue_gain=cue_gain,
        )
        return self._output(coarse, fine, coarse_cell_changed=changed)

    def output(self):
        self._require_initialized()
        return self._output(
            self.coarse.output(), self.fine.output(), coarse_cell_changed=False,
        )

    def _output(self, coarse, fine, *, coarse_cell_changed):
        decoded = self._coarse_center + fine.decoded_value
        return MultiScaleLineCANNOutput(
            timestamp=float(coarse.timestamp), decoded_value=float(decoded),
            coarse_cell_center=float(self._coarse_center),
            fine_residual=float(fine.decoded_value),
            cross_scale_disagreement=float(coarse.decoded_value - decoded),
            coarse_cell_changed=bool(coarse_cell_changed),
            saturated_at_boundary=bool(
                coarse.saturated_at_boundary or fine.saturated_at_boundary
            ),
            valid=bool(coarse.valid and fine.valid and np.isfinite(decoded)),
            coarse_output=coarse, fine_output=fine,
        )

    def _nearest_coarse_center(self, value):
        index = int(np.argmin(np.abs(self.coarse.preferred_value - float(value))))
        return float(self.coarse.preferred_value[index])

    def _require_initialized(self):
        if not self._initialized:
            raise RuntimeError("MultiScaleLineCANN.initialize is required.")
