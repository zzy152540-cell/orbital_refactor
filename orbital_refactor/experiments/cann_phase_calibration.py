from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from brain_inspired.ring_cann import RingCANN, RingCANNConfig


@dataclass(frozen=True)
class CANNPhaseCalibrationCase:
    group: str
    value: str
    phase_rate_deg_s: float
    sample_dt_s: float
    internal_dt_s: float
    num_neurons: int
    cue_interval_s: float | None
    cue_gain: float
    mean_error_deg: float
    rmse_deg: float
    error_std_deg: float
    error_peak_to_peak_deg: float
    equivalent_delay_s: float | None
    mean_concentration: float
    mean_width_rad: float


def run_cann_phase_calibration(
    *, duration: float = 20.0, burn_in: float = 5.0,
    base_rate_deg_s: float = 0.0607,
) -> dict[str, object]:
    if duration <= 0.0 or not 0.0 <= burn_in < duration:
        raise ValueError("Require duration > burn_in >= 0.")
    base = RingCANNConfig()
    specifications = []
    for rate in (-1.0, -0.2, -base_rate_deg_s, base_rate_deg_s, 0.2, 1.0):
        specifications.append(("phase_rate", f"{rate:.6g}", rate, 2.0, base, None, base.cue_gain))
    for sample_dt in (0.5, 1.0, 2.0, 5.0):
        specifications.append(("sample_dt", f"{sample_dt:g}", base_rate_deg_s, sample_dt, base, None, base.cue_gain))
    for internal_dt in (0.002, 0.001, 0.0005):
        specifications.append((
            "internal_dt", f"{internal_dt:g}", base_rate_deg_s, 2.0,
            replace(base, internal_dt=internal_dt), None, base.cue_gain,
        ))
    for neurons in (90, 180, 360):
        specifications.append((
            "num_neurons", str(neurons), base_rate_deg_s, 2.0,
            replace(base, num_neurons=neurons), None, base.cue_gain,
        ))
    for cue_interval in (2.0, 10.0, 20.0):
        specifications.append((
            "cue_interval", f"{cue_interval:g}", base_rate_deg_s, 2.0,
            base, cue_interval, base.cue_gain,
        ))
    for cue_gain in (0.1, 0.25, 0.5):
        specifications.append((
            "cue_gain", f"{cue_gain:g}", base_rate_deg_s, 2.0,
            base, 10.0, cue_gain,
        ))
    cases = tuple(
        _run_case(
            group=group, value=value, phase_rate_deg_s=rate,
            sample_dt=sample_dt, config=config, cue_interval=cue_interval,
            cue_gain=cue_gain, duration=duration, burn_in=burn_in,
        )
        for group, value, rate, sample_dt, config, cue_interval, cue_gain
        in specifications
    )
    motion_cases = [case for case in cases if case.group == "phase_rate"]
    rates = np.deg2rad([case.phase_rate_deg_s for case in motion_cases])
    biases = np.deg2rad([case.mean_error_deg for case in motion_cases])
    design = np.column_stack((np.ones_like(rates), rates))
    intercept, slope = np.linalg.lstsq(design, biases, rcond=None)[0]
    fitted = design @ np.array([intercept, slope])
    ss_res = float(np.sum((biases - fitted) ** 2))
    ss_tot = float(np.sum((biases - np.mean(biases)) ** 2))
    delay_fit = {
        "intercept_deg": float(np.rad2deg(intercept)),
        "delay_s": float(-slope),
        "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 1.0,
    }
    return {"cases": cases, "delay_fit": delay_fit}


def _run_case(
    *, group, value, phase_rate_deg_s, sample_dt, config,
    cue_interval, cue_gain, duration, burn_in,
) -> CANNPhaseCalibrationCase:
    rate = float(np.deg2rad(phase_rate_deg_s))
    initial_phase = float(np.deg2rad(37.0))
    cann = RingCANN(config)
    initial = cann.reset(initial_phase)
    times = [0.0]; decoded = [initial.decoded_phase]
    concentration = [initial.bump_concentration]; widths = [initial.bump_width]
    elapsed = 0.0
    while elapsed < duration:
        step_dt = min(sample_dt, duration - elapsed)
        next_time = elapsed + step_dt
        apply_cue = (
            cue_interval is not None
            and abs(next_time / cue_interval - round(next_time / cue_interval)) < 1e-9
        )
        truth = initial_phase + rate * next_time
        output = cann.step(rate, step_dt)
        if apply_cue:
            output = cann.apply_phase_cue(truth, cue_gain=cue_gain)
        elapsed = next_time
        times.append(elapsed); decoded.append(output.decoded_phase)
        concentration.append(output.bump_concentration); widths.append(output.bump_width)
    times = np.asarray(times)
    truth = (initial_phase + rate * times) % (2.0 * np.pi)
    error = (np.asarray(decoded) - truth + np.pi) % (2.0 * np.pi) - np.pi
    selected = times >= burn_in
    selected_error = error[selected]
    mean_error = float(np.mean(selected_error))
    equivalent_delay = None if abs(rate) < 1e-15 else float(-mean_error / rate)
    return CANNPhaseCalibrationCase(
        group=group, value=value, phase_rate_deg_s=float(phase_rate_deg_s),
        sample_dt_s=float(sample_dt), internal_dt_s=float(config.internal_dt),
        num_neurons=int(config.num_neurons), cue_interval_s=cue_interval,
        cue_gain=float(cue_gain), mean_error_deg=float(np.rad2deg(mean_error)),
        rmse_deg=float(np.rad2deg(np.sqrt(np.mean(selected_error**2)))),
        error_std_deg=float(np.rad2deg(np.std(selected_error))),
        error_peak_to_peak_deg=float(np.rad2deg(np.ptp(selected_error))),
        equivalent_delay_s=equivalent_delay,
        mean_concentration=float(np.mean(np.asarray(concentration)[selected])),
        mean_width_rad=float(np.mean(np.asarray(widths)[selected])),
    )


def write_cann_phase_calibration(result, output_dir: str | Path) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    cases = result["cases"]
    csv_path = output / "calibration.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=asdict(cases[0]).keys())
        writer.writeheader(); writer.writerows(asdict(case) for case in cases)
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(result["delay_fit"], indent=2), encoding="utf-8",
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = (
        ("phase_rate", "phase_rate_deg_s", "Phase rate (deg/s)"),
        ("internal_dt", "internal_dt_s", "Internal dt (s)"),
        ("num_neurons", "num_neurons", "Neuron count"),
        ("cue_interval", "cue_interval_s", "Cue interval (s)"),
    )
    for axis, (group, x_field, label) in zip(axes.flat, panels):
        selected = [case for case in cases if case.group == group]
        x = [getattr(case, x_field) for case in selected]
        axis.plot(x, [case.rmse_deg for case in selected], marker="o", label="RMSE")
        axis.plot(x, [abs(case.mean_error_deg) for case in selected], marker="s",
                  label="|mean bias|")
        axis.set_xlabel(label); axis.set_ylabel("Phase error (deg)")
        axis.set_title(group.replace("_", " ").title()); axis.grid(True); axis.legend()
    fig.suptitle(
        "Ring CANN analytic phase calibration | fitted delay "
        f"{result['delay_fit']['delay_s']:.6f} s",
    )
    fig.tight_layout()
    figure_path = output / "overview.png"
    fig.savefig(figure_path, dpi=180); plt.close(fig)
    return {"csv": csv_path, "summary": summary_path, "figure": figure_path}
