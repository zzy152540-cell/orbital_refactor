from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brain_inspired.ring_cann import RingCANN, RingCANNConfig


@dataclass(frozen=True)
class RingCANNTrace:
    scenario: str
    time: np.ndarray
    truth_phase: np.ndarray
    decoded_phase: np.ndarray
    phase_error: np.ndarray
    bump_concentration: np.ndarray
    bump_width: np.ndarray


def run_ring_cann_benchmark(
    *, sample_dt: float = 0.02, duration_scale: float = 1.0,
    config: RingCANNConfig | None = None,
) -> dict[str, RingCANNTrace]:
    """Run the four standalone acceptance scenarios for the ring CANN."""

    if not np.isfinite(sample_dt) or sample_dt <= 0.0:
        raise ValueError("sample_dt must be finite and positive.")
    if not np.isfinite(duration_scale) or duration_scale <= 0.0:
        raise ValueError("duration_scale must be finite and positive.")
    selected = config or RingCANNConfig()
    scenarios = (
        ("static", np.deg2rad(37.0), 0.0, 10.0, None),
        ("positive_rate", np.deg2rad(120.0), np.deg2rad(10.0), 20.0, None),
        ("negative_rate", np.deg2rad(120.0), -np.deg2rad(10.0), 20.0, None),
        ("wrap", np.deg2rad(359.0), np.deg2rad(5.0), 2.0, None),
        ("external_cue", np.deg2rad(110.0), 0.0, 2.0, np.deg2rad(90.0)),
    )
    return {
        name: _simulate(
            name=name, initial_phase=initial, phase_rate=rate,
            duration=duration * duration_scale, sample_dt=sample_dt,
            external_phase_hint=hint, config=selected,
        )
        for name, initial, rate, duration, hint in scenarios
    }


def write_ring_cann_benchmark_csv(
    traces: dict[str, RingCANNTrace], output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "scenario", "time_s", "truth_phase_rad", "decoded_phase_rad",
        "phase_error_rad", "bump_concentration", "bump_width_rad",
    )
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for trace in traces.values():
            for index, timestamp in enumerate(trace.time):
                writer.writerow({
                    "scenario": trace.scenario,
                    "time_s": timestamp,
                    "truth_phase_rad": trace.truth_phase[index],
                    "decoded_phase_rad": trace.decoded_phase[index],
                    "phase_error_rad": trace.phase_error[index],
                    "bump_concentration": trace.bump_concentration[index],
                    "bump_width_rad": trace.bump_width[index],
                })
    return output


def generate_ring_cann_benchmark_figure(
    traces: dict[str, RingCANNTrace], output_path: str | Path,
) -> Path:
    """Create a compact acceptance overview without requiring an interactive GUI."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    required = {
        "static", "positive_rate", "negative_rate", "wrap", "external_cue",
    }
    if set(traces) != required:
        raise ValueError(f"Expected benchmark scenarios: {sorted(required)}")
    output = Path(output_path)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    static = traces["static"]
    static_error_deg = np.rad2deg(static.phase_error)
    axes[0, 0].plot(static.time, static_error_deg)
    axes[0, 0].set_ylim(-0.01, 0.01)
    axes[0, 0].text(
        0.03, 0.92, f"max |error| = {np.max(np.abs(static_error_deg)):.2e} deg",
        transform=axes[0, 0].transAxes, va="top",
    )
    axes[0, 0].set(title="Static bump retention", ylabel="phase error (deg)")

    for name, label in (("positive_rate", "+10 deg/s"),
                        ("negative_rate", "-10 deg/s")):
        trace = traces[name]
        axes[0, 1].plot(
            trace.time, np.rad2deg(np.unwrap(trace.decoded_phase)), label=label,
        )
        axes[0, 1].plot(
            trace.time, np.rad2deg(np.unwrap(trace.truth_phase)),
            linestyle="--", alpha=0.65,
        )
    axes[0, 1].set(title="Bidirectional velocity integration",
                   ylabel="unwrapped phase (deg)")
    axes[0, 1].legend()

    wrap = traces["wrap"]
    axes[1, 0].plot(wrap.time, np.rad2deg(wrap.truth_phase), "--", label="truth")
    axes[1, 0].plot(wrap.time, np.rad2deg(wrap.decoded_phase), label="CANN")
    axes[1, 0].set(title="Circular boundary crossing", xlabel="time (s)",
                   ylabel="wrapped phase (deg)")
    axes[1, 0].legend()

    cue = traces["external_cue"]
    axes[1, 1].plot(cue.time, np.rad2deg(cue.phase_error), label="phase error")
    quality_axis = axes[1, 1].twinx()
    quality_axis.plot(cue.time, cue.bump_concentration, color="tab:orange",
                      label="concentration")
    axes[1, 1].set(title="External-cue correction", xlabel="time (s)",
                   ylabel="phase error (deg)")
    quality_axis.set_ylabel("bump concentration")
    axes[1, 1].legend(loc="upper right")
    quality_axis.legend(loc="lower right")

    for axis in axes.flat:
        axis.grid(alpha=0.3)
        if axis is not axes[0, 0] and axis is not axes[0, 1]:
            continue
        axis.set_xlabel("time (s)")
    figure.suptitle("Zhang-1996 Ring CANN | standalone engineering benchmark",
                    fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)
    return output


def _simulate(
    *, name: str, initial_phase: float, phase_rate: float, duration: float,
    sample_dt: float, external_phase_hint: float | None,
    config: RingCANNConfig,
) -> RingCANNTrace:
    cann = RingCANN(config)
    initial = cann.reset(initial_phase)
    time = [0.0]
    decoded = [initial.decoded_phase]
    concentration = [initial.bump_concentration]
    width = [initial.bump_width]
    elapsed = 0.0
    while elapsed < duration:
        step_dt = min(sample_dt, duration - elapsed)
        output = cann.step(
            phase_rate, step_dt, external_phase_hint=external_phase_hint,
        )
        elapsed += step_dt
        time.append(elapsed)
        decoded.append(output.decoded_phase)
        concentration.append(output.bump_concentration)
        width.append(output.bump_width)
    timestamps = np.asarray(time)
    truth = (initial_phase + phase_rate * timestamps) % (2.0 * np.pi)
    if external_phase_hint is not None:
        truth = np.full_like(timestamps, external_phase_hint % (2.0 * np.pi))
    decoded_array = np.asarray(decoded)
    error = (decoded_array - truth + np.pi) % (2.0 * np.pi) - np.pi
    return RingCANNTrace(
        scenario=name, time=timestamps, truth_phase=truth,
        decoded_phase=decoded_array, phase_error=error,
        bump_concentration=np.asarray(concentration),
        bump_width=np.asarray(width),
    )
