from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray
TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class RingCANNConfig:
    """Zhang-1996 ring-attractor parameters and numerical controls."""

    num_neurons: int = 180
    tau: float = 0.010
    internal_dt: float = 0.001
    initialization_duration: float = 0.5
    activation_scale: float = 6.34
    activation_power: float = 0.8
    activation_slope: float = 10.0
    activation_bias: float = 0.5
    tuning_sharpness: float = 8.0
    background_firing_rate: float = 1.0
    peak_firing_rate: float = 40.0
    normalized_regularization: float = 1.0e-3
    cue_gain: float = 0.25

    def validate(self) -> None:
        if self.num_neurons < 3:
            raise ValueError("A ring CANN requires at least three neurons.")
        positive = (
            self.tau, self.internal_dt, self.initialization_duration,
            self.activation_scale, self.activation_power,
            self.activation_slope, self.tuning_sharpness,
            self.background_firing_rate, self.peak_firing_rate,
            self.normalized_regularization,
        )
        if not np.all(np.isfinite(positive)) or min(positive) <= 0.0:
            raise ValueError("Ring CANN parameters must be finite and positive.")
        if self.internal_dt > 0.2 * self.tau:
            raise ValueError("Forward Euler requires internal_dt <= 0.2 * tau.")
        if self.peak_firing_rate <= self.background_firing_rate:
            raise ValueError("Peak firing rate must exceed background firing.")
        if not np.isfinite(self.activation_bias):
            raise ValueError("Activation bias must be finite.")
        if not np.isfinite(self.cue_gain) or self.cue_gain < 0.0:
            raise ValueError("Cue gain must be finite and nonnegative.")


@dataclass(frozen=True)
class CANNOutput:
    timestamp: float
    decoded_phase: float
    neural_activity: Array
    bump_concentration: float
    bump_width: float
    valid: bool
    internal_step_count: int


class RingCANN:
    """Discrete engineering reproduction of Zhang's 1996 HD ring CANN."""

    def __init__(self, config: RingCANNConfig = RingCANNConfig()) -> None:
        config.validate()
        self.config = config
        self.preferred_phase = (
            TWO_PI * np.arange(config.num_neurons, dtype=float)
            / config.num_neurons
        )
        self.target_firing_profile = self._target_firing(
            self.preferred_phase
        )
        self.target_input_profile = self.inverse_activation(
            self.target_firing_profile
        )
        self.static_kernel, self.derivative_kernel = self._build_kernels()
        self._static_kernel_fft = np.fft.fft(self.static_kernel)
        self._derivative_kernel_fft = np.fft.fft(self.derivative_kernel)
        self.input_state = np.zeros(config.num_neurons, dtype=float)
        self.firing_rate = np.zeros(config.num_neurons, dtype=float)
        self._failed_neuron_mask = np.zeros(config.num_neurons, dtype=bool)
        self.timestamp = 0.0
        self._initialized = False

    def activation(self, input_state: Array) -> Array:
        values = np.asarray(input_state, dtype=float)
        argument = self.config.activation_slope * (
            values + self.config.activation_bias
        )
        softplus = np.logaddexp(0.0, argument)
        return self.config.activation_scale * np.power(
            softplus, self.config.activation_power
        )

    def inverse_activation(self, firing_rate: Array) -> Array:
        values = np.asarray(firing_rate, dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("Firing rates must be finite and positive.")
        transformed = np.power(
            values / self.config.activation_scale,
            1.0 / self.config.activation_power,
        )
        log_expm1 = transformed + np.log(-np.expm1(-transformed))
        return (
            log_expm1 / self.config.activation_slope
            - self.config.activation_bias
        )

    def reset(self, initial_phase: float, *, timestamp: float = 0.0) -> CANNOutput:
        phase = _wrapped_phase(initial_phase)
        if not np.isfinite(timestamp):
            raise ValueError("CANN timestamp must be finite.")
        delta = _wrapped_signed(self.preferred_phase - phase)
        self.input_state = self.inverse_activation(self._target_firing(delta))
        self.firing_rate = self.activation(self.input_state)
        self._enforce_neuron_failures()
        self.timestamp = float(timestamp)
        initialization_steps = self._integrate(
            phase_rate=0.0,
            duration=self.config.initialization_duration,
            external_phase_hint=None,
            cue_gain=0.0,
        )
        self._initialized = True
        return self.output(internal_step_count=initialization_steps)

    def step(
        self,
        phase_rate: float,
        dt: float,
        *,
        external_phase_hint: float | None = None,
        cue_gain: float | None = None,
    ) -> CANNOutput:
        if not self._initialized:
            raise RuntimeError("RingCANN.reset must be called before step.")
        if not np.isfinite(phase_rate):
            raise ValueError("Phase rate must be finite.")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("CANN step duration must be finite and positive.")
        gain = self.config.cue_gain if cue_gain is None else float(cue_gain)
        if not np.isfinite(gain) or gain < 0.0:
            raise ValueError("Cue gain must be finite and nonnegative.")
        if external_phase_hint is not None:
            external_phase_hint = _wrapped_phase(external_phase_hint)
        internal_steps = self._integrate(
            phase_rate=float(phase_rate), duration=float(dt),
            external_phase_hint=external_phase_hint, cue_gain=gain,
        )
        self.timestamp += float(dt)
        return self.output(internal_step_count=internal_steps)

    def apply_phase_cue(
        self, phase_hint: float, *, cue_gain: float | None = None,
        relaxation_duration: float | None = None,
    ) -> CANNOutput:
        """Assimilate an endpoint phase cue without advancing physical time.

        This is the discrete-measurement counterpart of ``step``.  It avoids
        applying a cue observed at the end of an estimator interval throughout
        that preceding interval, which would introduce an acausal phase lead.
        """

        if not self._initialized:
            raise RuntimeError("RingCANN.reset must be called before cue assimilation.")
        hint = _wrapped_phase(phase_hint)
        gain = self.config.cue_gain if cue_gain is None else float(cue_gain)
        duration = (
            self.config.initialization_duration if relaxation_duration is None
            else float(relaxation_duration)
        )
        if not np.isfinite(gain) or gain < 0.0:
            raise ValueError("Cue gain must be finite and nonnegative.")
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("Cue relaxation duration must be finite and positive.")
        internal_steps = self._integrate(
            phase_rate=0.0, duration=duration,
            external_phase_hint=hint, cue_gain=gain,
        )
        return self.output(internal_step_count=internal_steps)

    def output(self, *, internal_step_count: int = 0) -> CANNOutput:
        phase, concentration, width = decode_ring_activity(
            self.firing_rate, self.preferred_phase
        )
        valid = bool(
            np.all(np.isfinite(self.input_state))
            and np.all(np.isfinite(self.firing_rate))
            and np.isfinite(phase)
            and np.isfinite(concentration)
            and np.isfinite(width)
            and self.firing_rate.sum() > 0.0
        )
        return CANNOutput(
            timestamp=float(self.timestamp), decoded_phase=phase,
            neural_activity=self.firing_rate.copy(),
            bump_concentration=concentration, bump_width=width,
            valid=valid, internal_step_count=int(internal_step_count),
        )

    def apply_transient_perturbation(
        self, *, additive_input: Array | None = None,
        silenced_neuron_mask: Array | None = None,
    ) -> CANNOutput:
        """Perturb the current neural state once without advancing physical time."""

        if not self._initialized:
            raise RuntimeError("RingCANN.reset must be called before perturbation.")
        if additive_input is not None:
            perturbation = np.asarray(additive_input, dtype=float)
            if (
                perturbation.shape != (self.config.num_neurons,)
                or np.any(~np.isfinite(perturbation))
            ):
                raise ValueError("Additive input must be a finite ring vector.")
            self.input_state = self.input_state + perturbation
        if silenced_neuron_mask is not None:
            mask = np.asarray(silenced_neuron_mask, dtype=bool)
            if mask.shape != (self.config.num_neurons,):
                raise ValueError("Silenced-neuron mask has the wrong ring dimension.")
            background_input = float(self.inverse_activation(np.array([
                self.config.background_firing_rate,
            ]))[0])
            self.input_state[mask] = background_input
        if additive_input is None and silenced_neuron_mask is None:
            raise ValueError("At least one transient perturbation is required.")
        self.firing_rate = self.activation(self.input_state)
        self._enforce_neuron_failures()
        return self.output()

    def set_neuron_failure_mask(self, failed_neuron_mask: Array) -> CANNOutput:
        """Persistently disable selected neurons until the mask is replaced."""

        if not self._initialized:
            raise RuntimeError("RingCANN.reset must be called before neuron failure.")
        mask = np.asarray(failed_neuron_mask, dtype=bool)
        if mask.shape != (self.config.num_neurons,):
            raise ValueError("Failed-neuron mask has the wrong ring dimension.")
        if np.all(mask):
            raise ValueError("At least one ring neuron must remain available.")
        self._failed_neuron_mask = mask.copy()
        self._enforce_neuron_failures()
        return self.output()

    def recurrent_input(self, firing_rate: Array, gamma: float = 0.0) -> Array:
        values = np.asarray(firing_rate, dtype=float)
        if values.shape != (self.config.num_neurons,):
            raise ValueError("Firing activity has the wrong ring dimension.")
        kernel_fft = (
            self._static_kernel_fft + float(gamma) * self._derivative_kernel_fft
        )
        return np.fft.ifft(
            kernel_fft * np.fft.fft(values)
        ).real / self.config.num_neurons

    def recurrent_input_matrix_reference(
        self, firing_rate: Array, gamma: float = 0.0,
    ) -> Array:
        values = np.asarray(firing_rate, dtype=float)
        if values.shape != (self.config.num_neurons,):
            raise ValueError("Firing activity has the wrong ring dimension.")
        indices = (
            np.arange(self.config.num_neurons)[:, None]
            - np.arange(self.config.num_neurons)[None, :]
        ) % self.config.num_neurons
        kernel = self.static_kernel + float(gamma) * self.derivative_kernel
        return kernel[indices] @ values / self.config.num_neurons

    def _integrate(
        self, *, phase_rate: float, duration: float,
        external_phase_hint: float | None, cue_gain: float,
    ) -> int:
        full_steps = int(np.floor(duration / self.config.internal_dt))
        remainder = duration - full_steps * self.config.internal_dt
        steps = [self.config.internal_dt] * full_steps
        tolerance = 16.0 * np.finfo(float).eps * max(1.0, duration)
        if remainder > tolerance:
            steps.append(remainder)
        gamma = -self.config.tau * phase_rate
        external_input = 0.0
        if external_phase_hint is not None and cue_gain > 0.0:
            delta = _wrapped_signed(
                self.preferred_phase - external_phase_hint
            )
            external_input = cue_gain * self.inverse_activation(
                self._target_firing(delta)
            )
        for internal_step in steps:
            recurrent = self.recurrent_input(self.firing_rate, gamma)
            derivative = (
                -self.input_state + recurrent + external_input
            ) / self.config.tau
            self.input_state = self.input_state + internal_step * derivative
            self.firing_rate = self.activation(self.input_state)
            self._enforce_neuron_failures()
        return len(steps)

    def _enforce_neuron_failures(self) -> None:
        if not np.any(self._failed_neuron_mask):
            return
        background_input = float(self.inverse_activation(np.array([
            self.config.background_firing_rate,
        ]))[0])
        self.input_state[self._failed_neuron_mask] = background_input
        self.firing_rate[self._failed_neuron_mask] = 0.0

    def _target_firing(self, phase_difference: Array) -> Array:
        config = self.config
        amplitude = (
            (config.peak_firing_rate - config.background_firing_rate)
            * np.exp(-config.tuning_sharpness)
        )
        return (
            config.background_firing_rate
            + amplitude * np.exp(
                config.tuning_sharpness * np.cos(phase_difference)
            )
        )

    def _build_kernels(self) -> tuple[Array, Array]:
        count = self.config.num_neurons
        firing_hat = np.fft.fft(self.target_firing_profile) / count
        input_hat = np.fft.fft(self.target_input_profile) / count
        regularization = (
            self.config.normalized_regularization
            * np.max(np.abs(firing_hat) ** 2)
        )
        kernel_hat = (
            input_hat * np.conj(firing_hat)
            / (np.abs(firing_hat) ** 2 + regularization)
        )
        kernel = np.fft.ifft(count * kernel_hat).real
        negative_indices = (-np.arange(count)) % count
        kernel = 0.5 * (kernel + kernel[negative_indices])

        derivative = periodic_spectral_derivative(kernel)
        derivative = 0.5 * (
            derivative - derivative[negative_indices]
        )
        return kernel, derivative


def decode_ring_activity(
    firing_rate: Array, preferred_phase: Array,
) -> tuple[float, float, float]:
    firing = np.asarray(firing_rate, dtype=float).reshape(-1)
    phase = np.asarray(preferred_phase, dtype=float).reshape(-1)
    if firing.shape != phase.shape or firing.size < 1:
        raise ValueError("Firing rates and preferred phases must align.")
    if np.any(~np.isfinite(firing)) or np.any(firing < 0.0):
        raise ValueError("Firing rates must be finite and nonnegative.")
    total = float(firing.sum())
    if total <= 0.0:
        raise ValueError("Ring activity must contain positive firing.")
    normalized = firing / total
    cosine = float(np.dot(normalized, np.cos(phase)))
    sine = float(np.dot(normalized, np.sin(phase)))
    decoded = _wrapped_phase(np.arctan2(sine, cosine))
    concentration = float(np.clip(np.hypot(cosine, sine), 0.0, 1.0))
    width = float(np.sqrt(-2.0 * np.log(max(concentration, 1.0e-15))))
    return decoded, concentration, width


def periodic_spectral_derivative(samples: Array) -> Array:
    """Differentiate uniform samples on [0, 2*pi) with Fourier modes."""

    values = np.asarray(samples, dtype=float).reshape(-1)
    if values.size < 3 or np.any(~np.isfinite(values)):
        raise ValueError("Periodic derivative samples must be finite/nontrivial.")
    count = values.size
    normalized_hat = np.fft.fft(values) / count
    mode = np.fft.fftfreq(count, d=1.0 / count)
    derivative_hat = 1j * mode * normalized_hat
    if count % 2 == 0:
        derivative_hat[count // 2] = 0.0
    return np.fft.ifft(count * derivative_hat).real


def _wrapped_phase(value: float) -> float:
    if not np.isfinite(value):
        raise ValueError("Phase must be finite.")
    return float(value % TWO_PI)


def _wrapped_signed(value: Array | float) -> Array:
    return (np.asarray(value, dtype=float) + np.pi) % TWO_PI - np.pi
