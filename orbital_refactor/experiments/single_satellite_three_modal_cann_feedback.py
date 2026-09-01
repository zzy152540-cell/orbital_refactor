from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.single_satellite_cann_comparison import (
    run_single_satellite_cann_comparison,
)


def run_single_satellite_three_modal_cann_feedback(
    *, inject_faults=True, **kwargs,
):
    common = dict(kwargs)
    common["enable_cann"] = False
    fault = "impulsive" if inject_faults else None
    baseline = run_single_satellite_cann_comparison(
        **common, infrared_fault_mode=fault, radar_fault_mode=fault,
        optical_fault_mode=fault,
    )
    processed = run_single_satellite_cann_comparison(
        **common, hybrid_cann_preprocess_ir=True,
        radar_cann_preprocess=True, optical_cann_preprocess=True,
        infrared_fault_mode=fault, radar_fault_mode=fault,
        optical_fault_mode=fault,
    )
    baseline_summary = baseline["summary"]
    processed_summary = processed["summary"]
    def _optional_change(processed_value, baseline_value):
        if processed_value is None or baseline_value is None:
            return None
        return processed_value - baseline_value

    summary = {
        "inject_faults": bool(inject_faults),
        "baseline": baseline_summary,
        "three_modal_cann": processed_summary,
        "position_rmse_change_m": (
            processed_summary["position_rmse_m"]
            - baseline_summary["position_rmse_m"]
        ),
        "outage_position_rmse_change_m": _optional_change(
            processed_summary["position_rmse_outage_m"],
            baseline_summary["position_rmse_outage_m"],
        ),
        "recovery_position_rmse_change_m": _optional_change(
            processed_summary["position_rmse_recovery_m"],
            baseline_summary["position_rmse_recovery_m"],
        ),
    }
    return {"baseline": baseline, "processed": processed, "summary": summary}


def run_staggered_modality_outage_comparison(*, outage_windows, **kwargs):
    outage_result = run_single_satellite_three_modal_cann_feedback(
        inject_faults=False, outage_windows=outage_windows, **kwargs,
    )
    reference = run_single_satellite_three_modal_cann_feedback(
        inject_faults=False, outage_windows={}, **kwargs,
    )
    timestamps = np.asarray(outage_result["baseline"]["timestamps"], dtype=float)
    impact = {}
    for modality, (start, end) in outage_windows.items():
        mask = (timestamps >= start) & (timestamps <= end)
        impact[modality] = {}
        for path_name in ("baseline", "processed"):
            outage_error = outage_result[path_name]["position_error_m"][mask]
            reference_error = reference[path_name]["position_error_m"][mask]
            outage_rmse = float(np.sqrt(np.mean(outage_error ** 2)))
            reference_rmse = float(np.sqrt(np.mean(reference_error ** 2)))
            impact[modality][path_name] = {
                "outage_position_rmse_m": outage_rmse,
                "no_outage_position_rmse_m": reference_rmse,
                "net_position_rmse_change_m": outage_rmse - reference_rmse,
            }
    outage_result["no_outage_reference"] = reference
    outage_result["summary"]["outage_impact_vs_no_outage"] = impact
    return outage_result


def write_single_satellite_three_modal_cann_feedback(result, output_dir):
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(result["summary"], indent=2), encoding="utf-8",
    )

    baseline = result["baseline"]
    processed = result["processed"]
    timestamps = np.asarray(baseline["timestamps"], dtype=float)
    baseline_position = np.maximum(baseline["position_error_m"], 1e-9)
    processed_position = np.maximum(processed["position_error_m"], 1e-9)
    baseline_velocity = np.maximum(baseline["velocity_error_mps"], 1e-12)
    processed_velocity = np.maximum(processed["velocity_error_mps"], 1e-12)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=False)
    axes[0].plot(timestamps, baseline_position, label="Without CANN", linewidth=1.5)
    axes[0].plot(timestamps, processed_position, label="With CANN", linewidth=1.5)
    reference = result.get("no_outage_reference")
    if reference is not None:
        axes[0].plot(
            timestamps, np.maximum(reference["baseline"]["position_error_m"], 1e-9),
            label="Without CANN, no outage", linestyle="--", alpha=0.7,
        )
        axes[0].plot(
            timestamps, np.maximum(reference["processed"]["position_error_m"], 1e-9),
            label="With CANN, no outage", linestyle=":", alpha=0.8,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Position error (m)")
    axes[0].set_title("Full-duration position error")

    axes[1].plot(timestamps, baseline_velocity, label="Without CANN", linewidth=1.5)
    axes[1].plot(timestamps, processed_velocity, label="With CANN", linewidth=1.5)
    if reference is not None:
        axes[1].plot(
            timestamps, np.maximum(reference["baseline"]["velocity_error_mps"], 1e-12),
            label="Without CANN, no outage", linestyle="--", alpha=0.7,
        )
        axes[1].plot(
            timestamps, np.maximum(reference["processed"]["velocity_error_mps"], 1e-12),
            label="With CANN, no outage", linestyle=":", alpha=0.8,
        )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Velocity error (m/s)")
    axes[1].set_title("Full-duration velocity error")

    outage_windows = result["summary"]["baseline"].get("outage_windows", {})
    detail_start = min(
        (float(window[0]) for window in outage_windows.values()), default=1200.0,
    )
    recovery = timestamps >= detail_start
    if not np.any(recovery):
        recovery = np.ones_like(timestamps, dtype=bool)
    axes[2].plot(
        timestamps[recovery], baseline_position[recovery],
        label="Without CANN", linewidth=1.5,
    )
    axes[2].plot(
        timestamps[recovery], processed_position[recovery],
        label="With CANN", linewidth=1.5,
    )
    if reference is not None:
        axes[2].plot(
            timestamps[recovery],
            np.maximum(reference["baseline"]["position_error_m"][recovery], 1e-9),
            label="Without CANN, no outage", linestyle="--", alpha=0.7,
        )
        axes[2].plot(
            timestamps[recovery],
            np.maximum(reference["processed"]["position_error_m"][recovery], 1e-9),
            label="With CANN, no outage", linestyle=":", alpha=0.8,
        )
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Position error (m)")
    axes[2].set_title("Outage and recovery detail")

    distinct_windows = len({tuple(window) for window in outage_windows.values()}) > 1
    outage_styles = {
        "opt": ("tab:purple", "Optical outage"),
        "ir": ("tab:green", "Infrared outage"),
        "rad": ("tab:brown", "Radar outage"),
    }
    if distinct_windows:
        outage_masks = [
            (
                ~np.asarray(baseline["available_by_modality"][name], dtype=bool),
                *outage_styles[name],
            )
            for name in ("opt", "ir", "rad") if name in outage_windows
        ]
    else:
        outage_masks = [
            (~np.asarray(baseline["available"], dtype=bool),
             "gray", "Measurement outage")
        ]
    for mask, color, label in outage_masks:
        for axis in axes[:2]:
            axis.fill_between(
                timestamps, 0.0, 1.0, where=mask, color=color, alpha=0.14,
                transform=axis.get_xaxis_transform(), label=label,
            )
        if np.any(mask & recovery):
            axes[2].fill_between(
                timestamps[recovery], 0.0, 1.0, where=mask[recovery],
                color=color, alpha=0.14,
                transform=axes[2].get_xaxis_transform(), label=label,
            )

    if result["summary"]["inject_faults"]:
        fault_times = (300.0, 450.0, 1250.0, 1350.0, 1500.0, 1650.0)
        for axis in axes:
            fault_label_pending = True
            for fault_time in fault_times:
                if axis.get_xlim()[0] <= fault_time <= axis.get_xlim()[1]:
                    axis.axvline(
                        fault_time, color="tab:red", linestyle="--", alpha=0.35,
                        linewidth=1.0,
                        label="Injected fault" if fault_label_pending else None,
                    )
                    fault_label_pending = False

    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), loc="best")
    fig.suptitle("Three-modal Federated-CI: CANN measurement-sidecar comparison")
    fig.tight_layout()
    figure_path = output / "overview.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {"summary": summary_path, "figure": figure_path}
