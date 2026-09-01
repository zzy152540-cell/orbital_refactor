from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class LineCANNConfig:
    """Numerical baseline for a bounded one-dimensional attractor manifold."""

    num_neurons: int = 121
    minimum_value: float = -1.0
    maximum_value: float = 1.0
    tuning_width: float = 0.08
    background_firing_rate: float = 1.0
    peak_firing_rate: float = 40.0
    cue_gain: float = 0.25

    def validate(self) -> None:
        finite = (
            self.minimum_value, self.maximum_value, self.tuning_width,
            self.background_firing_rate, self.peak_firing_rate, self.cue_gain,
        )
        if self.num_neurons < 3 or np.any(~np.isfinite(finite)):
            raise ValueError("Line CANN parameters must be finite and nontrivial.")
        if self.maximum_value <= self.minimum_value:
            raise ValueError("Line CANN bounds must be strictly ordered.")
        if self.tuning_width <= 0.0:
            raise ValueError("Line CANN tuning width must be positive.")
        if self.background_firing_rate < 0.0:
            raise ValueError("Line CANN background firing must be nonnegative.")
        if self.peak_firing_rate <= self.background_firing_rate:
            raise ValueError("Line CANN peak firing must exceed background firing.")
        if not 0.0 <= self.cue_gain <= 1.0:
            raise ValueError("Line CANN cue gain must lie in [0, 1].")


@dataclass(frozen=True)
class LineCANNOutput:
    timestamp: float
    decoded_value: float
    neural_activity: Array
    bump_concentration: float
    bump_width: float
    saturated_at_boundary: bool
    valid: bool


class LineCANN:
    """Discrete engineering CANN on a bounded, nonperiodic line.

    This first baseline preserves a translated population bump directly.  It
    deliberately avoids changing any estimator state and is intended for
    analytic validation before recurrent Ring-Line coupling is introduced.
    """

    def __init__(self, config: LineCANNConfig = LineCANNConfig()) -> None:
        config.validate()
        self.config = config
        self.preferred_value = np.linspace(
            config.minimum_value, config.maximum_value, config.num_neurons,
        )
        self.firing_rate = np.full(
            config.num_neurons, config.background_firing_rate, dtype=float,
        )
        self.timestamp = 0.0
        self._initialized = False
        self._saturated = False

    def reset(self, initial_value: float, *, timestamp: float = 0.0) -> LineCANNOutput:
        value, saturated = self._bounded(initial_value)
        if not np.isfinite(timestamp):
            raise ValueError("Line CANN timestamp must be finite.")
        self.firing_rate = self._target_firing(value)
        self.timestamp = float(timestamp)
        self._initialized = True
        self._saturated = saturated
        return self.output()

    def step(self, value_rate: float, dt: float) -> LineCANNOutput:
        if not self._initialized:
            raise RuntimeError("LineCANN.reset must be called before step.")
        if not np.isfinite(value_rate):
            raise ValueError("Line CANN value rate must be finite.")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("Line CANN step duration must be finite and positive.")
        centroid, _, _ = decode_line_activity(
            self.firing_rate, self.preferred_value,
            background_firing_rate=self.config.background_firing_rate,
        )
        current = decode_line_activity_boundary_corrected(
            self.firing_rate, self.preferred_value,
            background_firing_rate=self.config.background_firing_rate,
            centroid=centroid,
        )
        target, saturated = self._bounded(current + float(value_rate) * float(dt))
        self.firing_rate = self._target_firing(target)
        self.timestamp += float(dt)
        self._saturated = saturated
        return self.output()

    def apply_value_cue(
        self, value_hint: float, *, cue_gain: float | None = None,
    ) -> LineCANNOutput:
        if not self._initialized:
            raise RuntimeError("LineCANN.reset must be called before cue assimilation.")
        hint, saturated = self._bounded(value_hint)
        gain = self.config.cue_gain if cue_gain is None else float(cue_gain)
        if not np.isfinite(gain) or not 0.0 <= gain <= 1.0:
            raise ValueError("Line CANN cue gain must lie in [0, 1].")
        background = self.config.background_firing_rate
        existing = np.maximum(self.firing_rate - background, 0.0)
        cue = np.maximum(self._target_firing(hint) - background, 0.0)
        self.firing_rate = background + (1.0 - gain) * existing + gain * cue
        self._saturated = saturated
        return self.output()

    def output(self) -> LineCANNOutput:
        centroid, concentration, width = decode_line_activity(
            self.firing_rate, self.preferred_value,
            background_firing_rate=self.config.background_firing_rate,
        )
        value = decode_line_activity_boundary_corrected(
            self.firing_rate, self.preferred_value,
            background_firing_rate=self.config.background_firing_rate,
            centroid=centroid,
        )
        valid = bool(
            np.all(np.isfinite(self.firing_rate))
            and np.all(self.firing_rate >= 0.0)
            and np.isfinite(value) and np.isfinite(width)
        )
        return LineCANNOutput(
            timestamp=float(self.timestamp), decoded_value=value,
            neural_activity=self.firing_rate.copy(),
            bump_concentration=concentration, bump_width=width,
            saturated_at_boundary=bool(self._saturated), valid=valid,
        )

    def _target_firing(self, value: float) -> Array:
        delta = (self.preferred_value - value) / self.config.tuning_width
        return self.config.background_firing_rate + (
            self.config.peak_firing_rate - self.config.background_firing_rate
        ) * np.exp(-0.5 * delta * delta)

    def _bounded(self, value: float) -> tuple[float, bool]:
        if not np.isfinite(value):
            raise ValueError("Line CANN value must be finite.")
        bounded = float(np.clip(
            value, self.config.minimum_value, self.config.maximum_value,
        ))
        return bounded, bounded != float(value)


def decode_line_activity(
    firing_rate: Array, preferred_value: Array, *,
    background_firing_rate: float = 0.0,
) -> tuple[float, float, float]:
    firing = np.asarray(firing_rate, dtype=float).reshape(-1)
    preferred = np.asarray(preferred_value, dtype=float).reshape(-1)
    if firing.shape != preferred.shape or not firing.size:
        raise ValueError("Line firing rates and preferred values must align.")
    if np.any(~np.isfinite(firing)) or np.any(firing < 0.0):
        raise ValueError("Line firing rates must be finite and nonnegative.")
    activity = np.maximum(firing - float(background_firing_rate), 0.0)
    total = float(activity.sum())
    if total <= 0.0:
        raise ValueError("Line activity must rise above background firing.")
    weights = activity / total
    decoded = float(weights @ preferred)
    width = float(np.sqrt(weights @ (preferred - decoded) ** 2))
    span = float(preferred[-1] - preferred[0])
    concentration = float(np.clip(1.0 - 2.0 * width / span, 0.0, 1.0))
    return decoded, concentration, width


def decode_line_activity_peak_fit(
    firing_rate: Array, preferred_value: Array, *,
    background_firing_rate: float = 0.0, fit_radius: int = 2,
) -> float:
    """Decode a single Gaussian-like bump by fitting its local log profile.

    A one-sided fit remains available at a line boundary, avoiding the inward
    centroid bias caused by truncating the activity bump outside the domain.
    """
    firing = np.asarray(firing_rate, dtype=float).reshape(-1)
    preferred = np.asarray(preferred_value, dtype=float).reshape(-1)
    if firing.shape != preferred.shape or firing.size < 3:
        raise ValueError("Line firing rates and preferred values must align.")
    activity = firing - float(background_firing_rate)
    if np.any(~np.isfinite(activity)) or float(np.max(activity)) <= 0.0:
        raise ValueError("Line activity must be finite and rise above background.")
    peak = int(np.argmax(activity))
    radius = max(int(fit_radius), 1)
    start = max(0, peak - radius)
    stop = min(activity.size, peak + radius + 1)
    if stop - start < 3:
        if start == 0:
            stop = min(activity.size, 3)
        else:
            start = max(0, activity.size - 3)
    x = preferred[start:stop]
    y = np.log(np.maximum(activity[start:stop], np.max(activity) * 1.0e-12))
    quadratic, linear, _ = np.polyfit(x, y, 2)
    if not np.isfinite(quadratic) or quadratic >= 0.0:
        return float(preferred[peak])
    decoded = -linear / (2.0 * quadratic)
    return float(np.clip(decoded, preferred[0], preferred[-1]))


def decode_line_activity_boundary_corrected(
    firing_rate: Array, preferred_value: Array, *,
    background_firing_rate: float = 0.0, boundary_bins: int = 8,
    centroid: float | None = None,
) -> float:
    """Use smooth centroid decoding internally and a local fit at boundaries."""
    firing = np.asarray(firing_rate, dtype=float).reshape(-1)
    preferred = np.asarray(preferred_value, dtype=float).reshape(-1)
    activity = firing - float(background_firing_rate)
    peak = int(np.argmax(activity))
    bins = max(int(boundary_bins), 1)
    if peak < bins or peak >= firing.size - bins:
        return decode_line_activity_peak_fit(
            firing, preferred,
            background_firing_rate=background_firing_rate,
            fit_radius=bins,
        )
    if centroid is None:
        centroid, _, _ = decode_line_activity(
            firing, preferred,
            background_firing_rate=background_firing_rate,
        )
    return float(centroid)
