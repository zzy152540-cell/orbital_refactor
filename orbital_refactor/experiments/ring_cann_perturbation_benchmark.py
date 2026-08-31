from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brain_inspired.ring_cann import RingCANN


@dataclass(frozen=True)
class PerturbationTrace:
    condition: str
    timestamps: np.ndarray
    phase_error_deg: np.ndarray
    concentration: np.ndarray
    width: np.ndarray
    baseline_concentration: float
    final_phase_error_deg: float
    concentration_recovery_time_s: float | None


def run_ring_cann_perturbation_benchmark(
    *, recovery_duration: float = 3.0, sample_dt: float = 0.02,
    seed: int = 0,
) -> dict[str, PerturbationTrace]:
    """Measure autonomous bump recovery after one-time neural-state damage."""

    if recovery_duration <= 0.0 or sample_dt <= 0.0:
        raise ValueError("Recovery duration and sample_dt must be positive.")
    rng = np.random.default_rng(seed)
    conditions = {
        "noise_std_0.25": (rng.normal(0.0, 0.25, 180), None),
        "noise_std_0.50": (rng.normal(0.0, 0.50, 180), None),
        "noise_std_1.00": (rng.normal(0.0, 1.00, 180), None),
        "silence_center_10pct": (None, _centered_mask(180, 0.10, 60)),
        "silence_center_25pct": (None, _centered_mask(180, 0.25, 60)),
        "silence_leading_25pct": (None, _leading_mask(180, 0.25, 60)),
    }
    return {
        name: _run_condition(
            name=name, additive_input=additive, silenced_mask=mask,
            recovery_duration=recovery_duration, sample_dt=sample_dt,
        )
        for name, (additive, mask) in conditions.items()
    }


def write_ring_cann_perturbation_summary(
    traces: dict[str, PerturbationTrace], output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        name: {
            "immediate_phase_error_deg": float(trace.phase_error_deg[0]),
            "maximum_phase_error_deg": float(np.max(np.abs(trace.phase_error_deg))),
            "final_phase_error_deg": trace.final_phase_error_deg,
            "immediate_concentration": float(trace.concentration[0]),
            "baseline_concentration": trace.baseline_concentration,
            "final_concentration": float(trace.concentration[-1]),
            "concentration_recovery_time_s": trace.concentration_recovery_time_s,
        }
        for name, trace in traces.items()
    }
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output


def generate_ring_cann_perturbation_figure(
    traces: dict[str, PerturbationTrace], output_path: str | Path,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for name, trace in traces.items():
        axes[0, 0].plot(trace.timestamps, trace.phase_error_deg, label=name)
        axes[0, 1].plot(trace.timestamps, trace.concentration, label=name)
        axes[1, 0].plot(trace.timestamps, trace.width, label=name)
    baseline = next(iter(traces.values())).baseline_concentration
    axes[0, 1].axhline(baseline, color="black", linestyle="--",
                       label="pre-perturbation baseline")
    axes[0, 0].set(title="Decoded phase after transient damage",
                   ylabel="phase error (deg)")
    axes[0, 1].set(title="Bump concentration recovery",
                   ylabel="concentration")
    axes[1, 0].set(title="Bump width recovery", ylabel="circular width (rad)")

    names = tuple(traces)
    final_error = [abs(traces[name].final_phase_error_deg) for name in names]
    recovery = [
        traces[name].concentration_recovery_time_s
        if traces[name].concentration_recovery_time_s is not None else np.nan
        for name in names
    ]
    positions = np.arange(len(names))
    axes[1, 1].bar(positions, final_error, color="tab:blue",
                   label="final |phase error|")
    axes[1, 1].set_xticks(positions, names, rotation=25, ha="right")
    axes[1, 1].set_ylabel("final error (deg)")
    recovery_axis = axes[1, 1].twinx()
    recovery_axis.plot(positions, recovery, "D--", color="tab:orange",
                       label="concentration recovery")
    recovery_axis.set_ylabel("recovery time (s)")
    axes[1, 1].set_title("Residual phase shift versus shape recovery")
    axes[1, 1].legend(loc="upper left")
    recovery_axis.legend(loc="upper right")
    for axis in axes.flat[:3]:
        axis.set_xlabel("time after perturbation (s)")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3, axis="y")
    figure.suptitle(
        "Ring CANN autonomous recovery from transient neural perturbations",
        fontsize=14,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)
    return output


def _run_condition(
    *, name, additive_input, silenced_mask, recovery_duration, sample_dt,
):
    cann = RingCANN()
    target_phase = np.deg2rad(120.0)
    baseline = cann.reset(target_phase)
    perturbed = cann.apply_transient_perturbation(
        additive_input=additive_input, silenced_neuron_mask=silenced_mask,
    )
    timestamps = [0.0]
    outputs = [perturbed]
    elapsed = 0.0
    while elapsed < recovery_duration:
        dt = min(sample_dt, recovery_duration - elapsed)
        outputs.append(cann.step(0.0, dt))
        elapsed += dt
        timestamps.append(elapsed)
    phase_error = np.rad2deg(np.asarray([
        _circular_difference(output.decoded_phase, target_phase)
        for output in outputs
    ]))
    concentration = np.asarray([output.bump_concentration for output in outputs])
    width = np.asarray([output.bump_width for output in outputs])
    tolerance = 0.02 * baseline.bump_concentration
    recovered = np.flatnonzero(
        np.abs(concentration - baseline.bump_concentration) <= tolerance
    )
    recovery_time = (
        float(timestamps[recovered[0]]) if recovered.size else None
    )
    return PerturbationTrace(
        condition=name, timestamps=np.asarray(timestamps),
        phase_error_deg=phase_error, concentration=concentration, width=width,
        baseline_concentration=baseline.bump_concentration,
        final_phase_error_deg=float(phase_error[-1]),
        concentration_recovery_time_s=recovery_time,
    )


def _centered_mask(count, fraction, center_index):
    width = int(round(count * fraction))
    if width % 2 == 0:
        width += 1
    indices = (center_index - width // 2 + np.arange(width)) % count
    mask = np.zeros(count, dtype=bool)
    mask[indices] = True
    return mask


def _leading_mask(count, fraction, center_index):
    width = int(round(count * fraction))
    indices = (center_index + np.arange(1, width + 1)) % count
    mask = np.zeros(count, dtype=bool)
    mask[indices] = True
    return mask


def _circular_difference(actual, expected):
    return (actual - expected + np.pi) % (2.0 * np.pi) - np.pi
