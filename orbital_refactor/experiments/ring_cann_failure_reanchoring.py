from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brain_inspired.ring_cann import RingCANN


@dataclass(frozen=True)
class FailureReanchoringTrace:
    lesion: str
    mode: str
    timestamps: np.ndarray
    phase_error_deg: np.ndarray
    concentration: np.ndarray
    cue_applied: np.ndarray


def run_failure_reanchoring_benchmark(
    *, duration: float = 10.0, sample_dt: float = 0.1,
    cue_interval: float = 0.5, seed: int = 0,
) -> dict[str, FailureReanchoringTrace]:
    """Test whether sparse correct cues can anchor a persistently damaged ring."""

    if duration <= 0.0 or sample_dt <= 0.0 or cue_interval <= 0.0:
        raise ValueError("Durations and cue interval must be positive.")
    if not np.isclose(cue_interval / sample_dt, round(cue_interval / sample_dt)):
        raise ValueError("cue_interval must be an integer multiple of sample_dt.")
    masks = _failure_masks(seed)
    modes = {
        "no_cue": 0.0, "weak_cue": 0.05,
        "standard_cue": 0.25, "strong_cue": 1.0,
    }
    return {
        f"{lesion}:{mode}": _run_condition(
            lesion=lesion, mode=mode, mask=mask, cue_gain=gain,
            duration=duration, sample_dt=sample_dt,
            cue_interval=cue_interval,
        )
        for lesion, mask in masks.items()
        for mode, gain in modes.items()
    }


def write_failure_reanchoring_summary(
    traces: dict[str, FailureReanchoringTrace], output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        key: {
            "final_phase_error_deg": float(trace.phase_error_deg[-1]),
            "maximum_phase_error_deg": float(np.max(np.abs(trace.phase_error_deg))),
            "phase_rmse_deg": float(np.sqrt(np.mean(trace.phase_error_deg**2))),
            "final_concentration": float(trace.concentration[-1]),
            "cue_count": int(trace.cue_applied.sum()),
        }
        for key, trace in traces.items()
    }
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output


def generate_failure_reanchoring_figure(
    traces: dict[str, FailureReanchoringTrace], output_path: str | Path,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lesions = ("random_10pct", "random_25pct", "centered_25pct", "leading_25pct")
    colors = {"no_cue": "tab:gray", "weak_cue": "tab:blue",
              "standard_cue": "tab:orange", "strong_cue": "tab:green"}
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for axis, lesion in zip(axes.flat, lesions):
        for mode in colors:
            trace = traces[f"{lesion}:{mode}"]
            axis.plot(
                trace.timestamps, trace.phase_error_deg,
                color=colors[mode], label=(
                    f"{mode} | final {trace.phase_error_deg[-1]:.2f} deg | "
                    f"C={trace.concentration[-1]:.2f}"
                ),
            )
        axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        axis.set(title=lesion, xlabel="time after failure (s)",
                 ylabel="phase error (deg)")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle(
        "Permanent neuron failure with sparse correct phase anchors (0.5 s)",
        fontsize=14,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)
    return output


def _run_condition(
    *, lesion, mode, mask, cue_gain, duration, sample_dt, cue_interval,
):
    target = np.deg2rad(120.0)
    cann = RingCANN()
    cann.reset(target)
    outputs = [cann.set_neuron_failure_mask(mask)]
    timestamps = [0.0]
    cue_flags = [False]
    elapsed = 0.0
    cue_stride = int(round(cue_interval / sample_dt))
    index = 0
    while elapsed < duration:
        dt = min(sample_dt, duration - elapsed)
        index += 1
        use_cue = cue_gain > 0.0 and index % cue_stride == 0
        outputs.append(cann.step(
            0.0, dt, external_phase_hint=(target if use_cue else None),
            cue_gain=cue_gain,
        ))
        elapsed += dt
        timestamps.append(elapsed)
        cue_flags.append(use_cue)
    return FailureReanchoringTrace(
        lesion=lesion, mode=mode, timestamps=np.asarray(timestamps),
        phase_error_deg=np.rad2deg(np.asarray([
            _circular_difference(output.decoded_phase, target)
            for output in outputs
        ])),
        concentration=np.asarray([output.bump_concentration for output in outputs]),
        cue_applied=np.asarray(cue_flags, dtype=bool),
    )


def _failure_masks(seed):
    rng = np.random.default_rng(seed)
    count, center = 180, 60
    result = {}
    for fraction in (0.10, 0.25):
        mask = np.zeros(count, dtype=bool)
        mask[rng.choice(
            count, size=int(round(count * fraction)), replace=False,
        )] = True
        result[f"random_{int(100 * fraction)}pct"] = mask
    width = int(round(0.25 * count))
    centered = np.zeros(count, dtype=bool)
    centered[(center - width // 2 + np.arange(width)) % count] = True
    leading = np.zeros(count, dtype=bool)
    leading[(center + 1 + np.arange(width)) % count] = True
    result["centered_25pct"] = centered
    result["leading_25pct"] = leading
    return result


def _circular_difference(actual, expected):
    return (actual - expected + np.pi) % (2.0 * np.pi) - np.pi
