from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brain_inspired.ring_cann import RingCANN


@dataclass(frozen=True)
class PermanentFailureTrace:
    condition: str
    failed_fraction: float
    timestamps: np.ndarray
    phase_error_deg: np.ndarray
    concentration: np.ndarray
    width: np.ndarray
    valid: np.ndarray


def run_ring_cann_permanent_failure_benchmark(
    *, duration: float = 10.0, sample_dt: float = 0.1, seed: int = 0,
) -> dict[str, PermanentFailureTrace]:
    """Keep neuron lesions active and measure long-lived attractor degradation."""

    if duration <= 0.0 or sample_dt <= 0.0:
        raise ValueError("Duration and sample_dt must be positive.")
    rng = np.random.default_rng(seed)
    count, center = 180, 60
    conditions = {}
    for fraction in (0.10, 0.25, 0.40):
        random_mask = np.zeros(count, dtype=bool)
        random_mask[rng.choice(
            count, size=int(round(count * fraction)), replace=False,
        )] = True
        conditions[f"random_{int(100 * fraction)}pct"] = random_mask
    conditions["centered_25pct"] = _contiguous_mask(count, 0.25, center, True)
    conditions["leading_25pct"] = _contiguous_mask(count, 0.25, center, False)
    return {
        name: _run_condition(
            name, mask, duration=duration, sample_dt=sample_dt,
        )
        for name, mask in conditions.items()
    }


def write_permanent_failure_summary(
    traces: dict[str, PermanentFailureTrace], output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        name: {
            "failed_fraction": trace.failed_fraction,
            "immediate_phase_error_deg": float(trace.phase_error_deg[0]),
            "final_phase_error_deg": float(trace.phase_error_deg[-1]),
            "maximum_phase_error_deg": float(np.max(np.abs(trace.phase_error_deg))),
            "immediate_concentration": float(trace.concentration[0]),
            "final_concentration": float(trace.concentration[-1]),
            "final_width_rad": float(trace.width[-1]),
            "all_outputs_valid": bool(np.all(trace.valid)),
        }
        for name, trace in traces.items()
    }
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output


def generate_permanent_failure_figure(
    traces: dict[str, PermanentFailureTrace], output_path: str | Path,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for name, trace in traces.items():
        axes[0, 0].plot(trace.timestamps, trace.phase_error_deg, label=name)
        axes[0, 1].plot(trace.timestamps, trace.concentration, label=name)
        axes[1, 0].plot(trace.timestamps, trace.width, label=name)
    names = tuple(traces)
    positions = np.arange(len(names))
    axes[1, 1].bar(
        positions,
        [abs(traces[name].phase_error_deg[-1]) for name in names],
        color="tab:blue", label="final |phase error|",
    )
    axes[1, 1].set_xticks(positions, names, rotation=25, ha="right")
    axes[1, 1].set_ylabel("final phase error (deg)")
    quality_axis = axes[1, 1].twinx()
    quality_axis.plot(
        positions, [traces[name].concentration[-1] for name in names],
        "D--", color="tab:orange", label="final concentration",
    )
    quality_axis.set_ylabel("final concentration")
    axes[0, 0].set(title="Phase stability with persistent failures",
                   ylabel="phase error (deg)")
    axes[0, 1].set(title="Persistent bump concentration",
                   ylabel="concentration")
    axes[1, 0].set(title="Persistent bump width", ylabel="width (rad)")
    for axis in axes.flat[:3]:
        axis.set_xlabel("time after failure (s)")
        axis.grid(alpha=0.3)
        axis.legend()
    axes[1, 1].set_title("Final phase and shape quality")
    axes[1, 1].grid(alpha=0.3, axis="y")
    axes[1, 1].legend(loc="upper left")
    quality_axis.legend(loc="upper right")
    figure.suptitle("Ring CANN under permanent neuron failures", fontsize=14)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)
    return output


def _run_condition(name, mask, *, duration, sample_dt):
    cann = RingCANN()
    target = np.deg2rad(120.0)
    cann.reset(target)
    outputs = [cann.set_neuron_failure_mask(mask)]
    timestamps = [0.0]
    elapsed = 0.0
    while elapsed < duration:
        dt = min(sample_dt, duration - elapsed)
        outputs.append(cann.step(0.0, dt))
        elapsed += dt
        timestamps.append(elapsed)
    return PermanentFailureTrace(
        condition=name, failed_fraction=float(np.mean(mask)),
        timestamps=np.asarray(timestamps),
        phase_error_deg=np.rad2deg(np.asarray([
            _circular_difference(output.decoded_phase, target)
            for output in outputs
        ])),
        concentration=np.asarray([output.bump_concentration for output in outputs]),
        width=np.asarray([output.bump_width for output in outputs]),
        valid=np.asarray([output.valid for output in outputs], dtype=bool),
    )


def _contiguous_mask(count, fraction, center, symmetric):
    width = int(round(count * fraction))
    start = center - width // 2 if symmetric else center + 1
    indices = (start + np.arange(width)) % count
    mask = np.zeros(count, dtype=bool)
    mask[indices] = True
    return mask


def _circular_difference(actual, expected):
    return (actual - expected + np.pi) % (2.0 * np.pi) - np.pi
