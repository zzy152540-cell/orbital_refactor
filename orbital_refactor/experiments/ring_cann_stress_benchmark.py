from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brain_inspired.passive_phase_observer import (
    PassiveRingCANNObserver,
    PeriodicStateInput,
)


@dataclass(frozen=True)
class RingCANNStressResult:
    timestamps: np.ndarray
    truth_phase: np.ndarray
    measured_phase_rate: np.ndarray
    phase_hint: np.ndarray
    hint_available: np.ndarray
    hint_accepted: np.ndarray
    dead_reckoning_phase: np.ndarray
    gated_complementary_phase: np.ndarray
    cann_no_cue_phase: np.ndarray
    cann_sparse_cue_phase: np.ndarray
    cann_gated_cue_phase: np.ndarray
    concentration_by_mode: dict[str, np.ndarray]
    phase_rmse_deg_by_mode: dict[str, float]
    outage_rmse_deg_by_mode: dict[str, float]
    final_error_deg_by_mode: dict[str, float]


def run_ring_cann_stress_benchmark(
    *, duration: float = 600.0, sample_dt: float = 2.0, seed: int = 0,
    rate_bias_deg_s: float = 0.005, rate_noise_deg_s: float = 0.01,
    rate_random_walk_deg_s_sqrt_s: float = 0.0002,
    hint_interval: float = 20.0, hint_noise_deg: float = 0.05,
    outage_window: tuple[float, float] = (200.0, 400.0),
    outlier_deg: float = 5.0, cue_gain: float = 0.05,
    gate_threshold_deg: float = 1.0,
    complementary_gain: float = 1.0,
) -> RingCANNStressResult:
    """Stress ring integration with drift, sparse cues, outage, and outliers."""

    if duration <= 0.0 or sample_dt <= 0.0 or hint_interval <= 0.0:
        raise ValueError("Durations and sampling intervals must be positive.")
    timestamps = np.arange(0.0, duration + 0.5 * sample_dt, sample_dt)
    if not np.isclose(timestamps[-1], duration):
        raise ValueError("duration must be an integer multiple of sample_dt.")
    rng = np.random.default_rng(seed)
    truth_rate = np.deg2rad(
        0.06 + 0.015 * np.sin(2.0 * np.pi * timestamps / 300.0)
    )
    truth_unwrapped = np.deg2rad(40.0) + _left_integral(truth_rate, sample_dt)
    truth_phase = truth_unwrapped % (2.0 * np.pi)
    random_walk = np.cumsum(
        rng.normal(
            0.0, np.deg2rad(rate_random_walk_deg_s_sqrt_s) * np.sqrt(sample_dt),
            timestamps.size,
        )
    )
    measured_rate = (
        truth_rate + np.deg2rad(rate_bias_deg_s) + random_walk
        + rng.normal(0.0, np.deg2rad(rate_noise_deg_s), timestamps.size)
    )
    hint_stride = max(1, int(round(hint_interval / sample_dt)))
    hint_available = np.zeros(timestamps.size, dtype=bool)
    hint_available[hint_stride::hint_stride] = True
    hint_available &= ~(
        (timestamps >= outage_window[0]) & (timestamps <= outage_window[1])
    )
    phase_hint = np.full(timestamps.size, np.nan)
    phase_hint[hint_available] = (
        truth_phase[hint_available]
        + rng.normal(0.0, np.deg2rad(hint_noise_deg), hint_available.sum())
    ) % (2.0 * np.pi)
    available_indices = np.flatnonzero(hint_available)
    if available_indices.size >= 4:
        for index, sign in zip(
            available_indices[[available_indices.size // 3, -2]], (1.0, -1.0),
        ):
            phase_hint[index] = (
                phase_hint[index] + sign * np.deg2rad(outlier_deg)
            ) % (2.0 * np.pi)

    dead_reckoning = (
        truth_phase[0] + _left_integral(measured_rate, sample_dt)
    ) % (2.0 * np.pi)
    complementary = _run_gated_integrator(
        timestamps, truth_phase[0], measured_rate, phase_hint,
        hint_available, np.deg2rad(gate_threshold_deg), complementary_gain,
    )
    no_cue, no_quality, _ = _run_cann(
        timestamps, truth_phase[0], measured_rate, phase_hint,
        np.zeros_like(hint_available), cue_gain, None,
    )
    sparse, sparse_quality, _ = _run_cann(
        timestamps, truth_phase[0], measured_rate, phase_hint,
        hint_available, cue_gain, None,
    )
    gated, gated_quality, accepted = _run_cann(
        timestamps, truth_phase[0], measured_rate, phase_hint,
        hint_available, cue_gain, np.deg2rad(gate_threshold_deg),
    )
    phases = {
        "dead_reckoning": dead_reckoning,
        "gated_complementary": complementary,
        "cann_no_cue": no_cue,
        "cann_sparse_cue": sparse,
        "cann_gated_cue": gated,
    }
    errors = {
        mode: _circular_difference(phase, truth_phase)
        for mode, phase in phases.items()
    }
    outage = (
        (timestamps >= outage_window[0]) & (timestamps <= outage_window[1])
    )
    return RingCANNStressResult(
        timestamps=timestamps, truth_phase=truth_phase,
        measured_phase_rate=measured_rate, phase_hint=phase_hint,
        hint_available=hint_available, hint_accepted=accepted,
        dead_reckoning_phase=dead_reckoning, cann_no_cue_phase=no_cue,
        gated_complementary_phase=complementary,
        cann_sparse_cue_phase=sparse, cann_gated_cue_phase=gated,
        concentration_by_mode={
            "cann_no_cue": no_quality, "cann_sparse_cue": sparse_quality,
            "cann_gated_cue": gated_quality,
        },
        phase_rmse_deg_by_mode={
            mode: float(np.rad2deg(np.sqrt(np.mean(error**2))))
            for mode, error in errors.items()
        },
        outage_rmse_deg_by_mode={
            mode: float(np.rad2deg(np.sqrt(np.mean(error[outage]**2))))
            for mode, error in errors.items()
        },
        final_error_deg_by_mode={
            mode: float(np.rad2deg(error[-1])) for mode, error in errors.items()
        },
    )


def write_ring_cann_stress_csv(
    result: RingCANNStressResult, output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    phases = _phase_by_mode(result)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "time_s", "truth_phase_rad", "measured_rate_rad_s",
            "phase_hint_rad", "hint_available", "hint_accepted", "mode",
            "decoded_phase_rad", "truth_error_rad", "bump_concentration",
        ))
        for index, timestamp in enumerate(result.timestamps):
            for mode, phase in phases.items():
                quality = result.concentration_by_mode.get(mode)
                writer.writerow((
                    timestamp, result.truth_phase[index],
                    result.measured_phase_rate[index], result.phase_hint[index],
                    int(result.hint_available[index]),
                    int(result.hint_accepted[index]), mode, phase[index],
                    _circular_difference(phase[index], result.truth_phase[index]),
                    "" if quality is None else quality[index],
                ))
    return output


def generate_ring_cann_stress_figure(
    result: RingCANNStressResult, output_path: str | Path,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "dead_reckoning": "ordinary ring integration",
        "gated_complementary": "gated conventional correction",
        "cann_no_cue": "CANN rate only",
        "cann_sparse_cue": "CANN all sparse cues",
        "cann_gated_cue": "CANN gated sparse cues",
    }
    phases = _phase_by_mode(result)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    truth_unwrapped = np.unwrap(result.truth_phase)
    axes[0, 0].plot(result.timestamps, np.rad2deg(truth_unwrapped), "k--",
                    linewidth=2.0, label="truth")
    for mode, phase in phases.items():
        axes[0, 0].plot(result.timestamps, np.rad2deg(_unwrap_aligned(
            phase, truth_unwrapped[0],
        )), label=labels[mode])
    axes[0, 0].set(title="Long-horizon phase tracking",
                   ylabel="unwrapped phase (deg)")
    axes[0, 0].legend()

    for mode, phase in phases.items():
        error = np.rad2deg(_circular_difference(phase, result.truth_phase))
        axes[0, 1].plot(result.timestamps, error, label=(
            f"{labels[mode]} | RMSE {result.phase_rmse_deg_by_mode[mode]:.3f} deg"
        ))
    axes[0, 1].axvspan(200.0, 400.0, color="gray", alpha=0.15,
                       label="cue outage")
    axes[0, 1].set(title="Error under rate drift and cue outage",
                   ylabel="phase error (deg)")
    axes[0, 1].legend()

    available = np.flatnonzero(result.hint_available)
    accepted = np.flatnonzero(result.hint_accepted)
    axes[1, 0].scatter(
        result.timestamps[available], np.rad2deg(_circular_difference(
            result.phase_hint[available], result.truth_phase[available],
        )), marker="x", label="available cues",
    )
    axes[1, 0].scatter(
        result.timestamps[accepted], np.rad2deg(_circular_difference(
            result.phase_hint[accepted], result.truth_phase[accepted],
        )), facecolors="none", edgecolors="tab:green", s=70,
        label="accepted by gate",
    )
    axes[1, 0].set(title="Sparse cue noise, outliers, and gating",
                   ylabel="cue error (deg)")
    axes[1, 0].legend()

    for mode, quality in result.concentration_by_mode.items():
        axes[1, 1].plot(result.timestamps, quality, label=labels[mode])
    axes[1, 1].set(title="Attractor-state quality",
                   ylabel="bump concentration")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xlabel("time (s)")
        axis.grid(alpha=0.3)
    figure.suptitle(
        "Ring CANN 600 s stress benchmark | bias + random walk + cue outage",
        fontsize=14,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)
    return output


def _run_cann(
    timestamps, initial_phase, measured_rate, phase_hint, hint_available,
    cue_gain, gate_threshold,
):
    observer = PassiveRingCANNObserver()
    initial = observer.initialize(phase=initial_phase, timestamp=timestamps[0])
    decoded = [initial.decoded_phase]
    quality = [initial.bump_concentration]
    accepted = np.zeros(timestamps.size, dtype=bool)
    for index in range(1, timestamps.size):
        use_hint = bool(hint_available[index])
        if use_hint and gate_threshold is not None:
            innovation = _circular_difference(
                phase_hint[index], decoded[-1],
            )
            use_hint = bool(abs(innovation) <= gate_threshold)
        accepted[index] = use_hint
        output = observer.update(PeriodicStateInput(
            timestamp=timestamps[index], phase_rate=measured_rate[index - 1],
            phase_hint=(phase_hint[index] if use_hint else None),
            phase_hint_valid=use_hint, cue_gain=cue_gain,
        ))
        decoded.append(output.decoded_phase)
        quality.append(output.bump_concentration)
    return np.asarray(decoded), np.asarray(quality), accepted


def _run_gated_integrator(
    timestamps, initial_phase, measured_rate, phase_hint, hint_available,
    gate_threshold, correction_gain,
):
    if not np.isfinite(correction_gain) or not 0.0 <= correction_gain <= 1.0:
        raise ValueError("complementary_gain must lie in [0, 1].")
    phase = np.empty(timestamps.size, dtype=float)
    phase[0] = initial_phase
    for index in range(1, timestamps.size):
        dt = timestamps[index] - timestamps[index - 1]
        propagated = (phase[index - 1] + measured_rate[index - 1] * dt) % (
            2.0 * np.pi
        )
        innovation = _circular_difference(phase_hint[index], propagated)
        if hint_available[index] and abs(innovation) <= gate_threshold:
            propagated = (propagated + correction_gain * innovation) % (
                2.0 * np.pi
            )
        phase[index] = propagated
    return phase


def _left_integral(rate, dt):
    return np.concatenate(([0.0], np.cumsum(np.asarray(rate)[:-1] * dt)))


def _phase_by_mode(result):
    return {
        "dead_reckoning": result.dead_reckoning_phase,
        "gated_complementary": result.gated_complementary_phase,
        "cann_no_cue": result.cann_no_cue_phase,
        "cann_sparse_cue": result.cann_sparse_cue_phase,
        "cann_gated_cue": result.cann_gated_cue_phase,
    }


def _circular_difference(actual, expected):
    return (np.asarray(actual) - np.asarray(expected) + np.pi) % (2.0 * np.pi) - np.pi


def _unwrap_aligned(phase, reference_initial):
    unwrapped = np.unwrap(np.asarray(phase, dtype=float))
    turns = np.round((unwrapped[0] - reference_initial) / (2.0 * np.pi))
    return unwrapped - turns * 2.0 * np.pi
