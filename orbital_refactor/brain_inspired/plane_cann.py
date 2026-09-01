from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from brain_inspired.line_cann import LineCANN, LineCANNConfig, LineCANNOutput


@dataclass(frozen=True)
class PlaneCANNConfig:
    """Separable engineering baseline for a bounded two-dimensional CANN."""

    x_axis: LineCANNConfig = field(default_factory=LineCANNConfig)
    y_axis: LineCANNConfig = field(default_factory=LineCANNConfig)

    def validate(self) -> None:
        self.x_axis.validate()
        self.y_axis.validate()


@dataclass(frozen=True)
class PlaneCANNOutput:
    timestamp: float
    decoded_position: np.ndarray
    neural_activity: np.ndarray
    bump_concentration: float
    bump_width: np.ndarray
    saturated_at_boundary: bool
    valid: bool
    x_output: LineCANNOutput
    y_output: LineCANNOutput


class PlaneCANN:
    """Bounded 2-D attractor represented by separable population bumps."""

    def __init__(self, config: PlaneCANNConfig = PlaneCANNConfig()) -> None:
        config.validate()
        self.config = config
        self.x_axis = LineCANN(config.x_axis)
        self.y_axis = LineCANN(config.y_axis)

    def reset(self, position, *, timestamp=0.0) -> PlaneCANNOutput:
        value = np.asarray(position, dtype=float).reshape(2)
        return self._output(
            self.x_axis.reset(value[0], timestamp=timestamp),
            self.y_axis.reset(value[1], timestamp=timestamp),
        )

    def step(self, velocity, dt) -> PlaneCANNOutput:
        rate = np.asarray(velocity, dtype=float).reshape(2)
        return self._output(
            self.x_axis.step(rate[0], dt), self.y_axis.step(rate[1], dt),
        )

    def apply_position_cue(self, position, *, cue_gain=None) -> PlaneCANNOutput:
        value = np.asarray(position, dtype=float).reshape(2)
        return self._output(
            self.x_axis.apply_value_cue(value[0], cue_gain=cue_gain),
            self.y_axis.apply_value_cue(value[1], cue_gain=cue_gain),
        )

    def _output(self, x_output, y_output):
        x_activity = np.maximum(
            x_output.neural_activity
            - self.config.x_axis.background_firing_rate, 0.0,
        )
        y_activity = np.maximum(
            y_output.neural_activity
            - self.config.y_axis.background_firing_rate, 0.0,
        )
        activity = np.outer(y_activity, x_activity)
        return PlaneCANNOutput(
            timestamp=float(x_output.timestamp),
            decoded_position=np.array([
                x_output.decoded_value, y_output.decoded_value,
            ]),
            neural_activity=activity,
            bump_concentration=float(np.sqrt(
                x_output.bump_concentration * y_output.bump_concentration
            )),
            bump_width=np.array([x_output.bump_width, y_output.bump_width]),
            saturated_at_boundary=bool(
                x_output.saturated_at_boundary
                or y_output.saturated_at_boundary
            ),
            valid=bool(x_output.valid and y_output.valid),
            x_output=x_output, y_output=y_output,
        )
