from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from adapters.synthetic_measurement_adapter import (
    create_infrared_observations,
    create_optical_observations,
    create_radar_observations,
)
from brain_inspired.orbital_phase_adapter import OrbitalPlaneFrame
from brain_inspired.orbital_phase_sidecar import run_orbital_phase_sidecar
from cooperative.multi_sat_pipeline import build_module_inputs
from interfaces.state_awareness_module import StateAwarenessModule
from orbital_core.constants import R_EARTH
from orbital_core.coordinates import state_history_eci_to_spri
from orbital_core.orbit_elements import keplerian_to_eci
from scenarios.multi_satellite_scenario import generate_cooperative_scenario


def run_single_satellite_cann_comparison(
    *, duration: float = 1800.0, dt: float = 2.0, seed: int = 0,
    outage_start: float = 600.0, outage_end: float = 1200.0,
    cue_interval_samples: int = 5,
    outage_modalities: tuple[str, ...] = ("opt", "ir", "rad"),
    enable_cann: bool = True,
):
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt)
    target = keplerian_to_eci(
        R_EARTH + 700e3, 0.001, np.deg2rad(55.0),
        np.deg2rad(15.0), 0.0, np.deg2rad(8.0),
    )
    observer = keplerian_to_eci(
        R_EARTH + 702e3, 0.0012, np.deg2rad(54.5),
        np.deg2rad(14.5), 0.0, np.deg2rad(7.0),
    )
    scenario = generate_cooperative_scenario(
        timestamps=timestamps, target_id="target",
        target_initial_state_eci=target,
        observer_initial_states_eci={"sat_01": observer},
    )
    observer_track = scenario.observer_trajectories["sat_01"]
    relative = scenario.relative_state_eci_by_node["sat_01"]
    spri = state_history_eci_to_spri(
        relative, observer_track.q_eci2pri_history,
    )
    requested_outages = {str(name).lower() for name in outage_modalities}
    unknown = requested_outages - {"opt", "ir", "rad"}
    if unknown:
        raise ValueError(f"Unknown outage modalities: {sorted(unknown)}")
    outage_window = (timestamps >= outage_start) & (timestamps <= outage_end)
    available_by_modality = {
        name: (~outage_window if name in requested_outages else np.ones_like(outage_window))
        for name in ("opt", "ir", "rad")
    }
    rng = np.random.default_rng(seed)
    observations = []
    observations += create_optical_observations(
        timestamps=timestamps, relative_position_spri=spri[:, :3],
        covariance=np.diag([2e-4, 2e-4]) ** 2,
        observer_id="sat_01", target_id="target", rng=rng,
        valid_flags=available_by_modality["opt"],
    )
    observations += create_infrared_observations(
        timestamps=timestamps, relative_position_spri=spri[:, :3],
        covariance=np.diag(np.deg2rad([0.05, 0.05])) ** 2,
        observer_id="sat_01", target_id="target", rng=rng,
        valid_flags=available_by_modality["ir"],
    )
    observations += create_radar_observations(
        timestamps=timestamps, relative_position_spri=spri[:, :3],
        relative_velocity_spri=spri[:, 3:],
        covariance=np.diag([30.0, 0.05]) ** 2,
        observer_id="sat_01", target_id="target", rng=rng,
        valid_flags=available_by_modality["rad"],
    )
    module_input = build_module_inputs(
        scenario=scenario,
        observations_by_node={"sat_01": observations},
        initial_error_by_node={"sat_01": np.array([50., -40., 30., .05, -.04, .03])},
        ci_grid_points=31,
    )["sat_01"]
    if enable_cann:
        module_input.config["brain_inspired"] = {
            "cann": {
                "enabled": True,
                "cue_interval_samples": cue_interval_samples,
            },
        }
    history = StateAwarenessModule().run_history(module_input)
    estimate_eci = observer_track.state_history_eci + history.fused_state_history
    truth_eci = scenario.target_trajectory.state_history_eci
    # Use the same estimator-derived fixed frame as the integrated sidecar so
    # truth and decoded phase are compared in one coordinate convention.
    cann = history.cann_sidecar_history
    truth_phase = None
    phase_error = None
    if cann is not None:
        frame = OrbitalPlaneFrame.from_state_eci(estimate_eci[0])
        truth_phase = run_orbital_phase_sidecar(
            timestamps=timestamps, state_history_eci=truth_eci, frame=frame,
            source_id="target:truth",
        ).source_phase
        phase_error = (
            (cann.decoded_phase - truth_phase + np.pi) % (2 * np.pi) - np.pi
        )
    position_error = np.linalg.norm(estimate_eci[:, :3] - truth_eci[:, :3], axis=1)
    velocity_error = np.linalg.norm(estimate_eci[:, 3:] - truth_eci[:, 3:], axis=1)
    def _window_rmse(values, mask):
        return float(np.sqrt(np.mean(values[mask] ** 2))) if np.any(mask) else None

    recovery_window = timestamps > outage_end
    summary = {
        "outage_modalities": sorted(requested_outages),
        "position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
        "position_rmse_outage_m": _window_rmse(position_error, outage_window),
        "position_rmse_recovery_m": _window_rmse(position_error, recovery_window),
        "velocity_rmse_mps": float(np.sqrt(np.mean(velocity_error**2))),
        "velocity_rmse_outage_mps": _window_rmse(velocity_error, outage_window),
        "velocity_rmse_recovery_mps": _window_rmse(velocity_error, recovery_window),
        "cann_enabled": bool(enable_cann),
        "cann_phase_rmse_deg": (
            float(np.rad2deg(np.sqrt(np.mean(phase_error**2))))
            if phase_error is not None else None
        ),
        "cann_max_phase_error_deg": (
            float(np.rad2deg(np.max(np.abs(phase_error))))
            if phase_error is not None else None
        ),
        "cann_valid_fraction": float(np.mean(cann.valid)) if cann is not None else None,
        "optical_valid_count": int(np.count_nonzero(history.measurement_valid_history["opt"])),
        "infrared_valid_count": int(np.count_nonzero(history.measurement_valid_history["ir"])),
        "radar_valid_count": int(np.count_nonzero(history.measurement_valid_history["rad"])),
    }
    return {
        "timestamps": timestamps, "position_error_m": position_error,
        "velocity_error_mps": velocity_error, "truth_phase": truth_phase,
        "estimated_state_history_eci": estimate_eci,
        "truth_state_history_eci": truth_eci,
        "cann": cann,
        "available": np.logical_and.reduce(tuple(available_by_modality.values())),
        "available_by_modality": available_by_modality,
        "summary": summary,
    }


def write_single_satellite_cann_results(result, output_dir: str | Path) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(result["summary"], indent=2, ensure_ascii=False), encoding="utf-8",
    )
    t = result["timestamps"]
    cann = result["cann"]
    phase_error = None
    if cann is not None:
        phase_error = (
            (cann.decoded_phase - result["truth_phase"] + np.pi) % (2*np.pi) - np.pi
        )
    panel_count = 3 if cann is not None else 2
    fig, axes = plt.subplots(panel_count, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(t, result["position_error_m"], label="Federated-CI position error")
    axes[0].set_ylabel("Position error (m)"); axes[0].grid(True); axes[0].legend()
    if cann is not None:
        axes[1].plot(t, np.rad2deg(phase_error), label="CANN phase error", color="tab:orange")
        axes[1].set_ylabel("Phase error (deg)"); axes[1].grid(True); axes[1].legend()
        axes[2].plot(t, cann.bump_concentration, label="Bump concentration")
        axes[2].plot(t, cann.bump_width, label="Bump width (rad)")
        axes[2].set_ylabel("CANN diagnostic"); axes[2].grid(True); axes[2].legend()
    else:
        axes[1].plot(t, result["velocity_error_mps"], label="Federated-CI velocity error",
                     color="tab:green")
        axes[1].set_ylabel("Velocity error (m/s)"); axes[1].grid(True); axes[1].legend()
    axes[-1].set_xlabel("Time (s)")
    for axis in axes:
        axis.fill_between(t, 0, 1, where=~result["available"], color="gray", alpha=.12,
                          transform=axis.get_xaxis_transform(), label=None)
    suffix = "with passive CANN" if cann is not None else "original baseline (no CANN)"
    fig.suptitle(f"Single-satellite three-modal Federated-CI: {suffix}")
    fig.tight_layout()
    figure_path = output / "overview.png"
    fig.savefig(figure_path, dpi=180); plt.close(fig)
    return {"summary": summary_path, "figure": figure_path}
