from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brain_inspired.orbital_phase_adapter import OrbitalPlaneFrame
from brain_inspired.orbital_phase_sidecar import (
    OrbitalPhaseSidecarHistory,
    run_orbital_phase_sidecar,
)
from cooperative.network_schmidt_runner import run_network_schmidt_filter
from experiments.v14_walker_geometry_audit import run_v14_walker_geometry_audit
from experiments.walker_filter_setup import build_walker_filter_case


@dataclass(frozen=True)
class WalkerCANNSidecarComparison:
    node_id: str
    truth: OrbitalPhaseSidecarHistory
    estimated_no_cue: OrbitalPhaseSidecarHistory
    estimated_periodic_cue: OrbitalPhaseSidecarHistory
    estimated_every_epoch_cue: OrbitalPhaseSidecarHistory
    phase_rmse_deg_by_mode: dict[str, float]
    maximum_phase_error_deg_by_mode: dict[str, float]
    source_position_rmse_m: float


def run_walker_cann_sidecar_comparison(
    *, duration: float = 60.0, dt: float = 2.0, seed: int = 0,
    maximum_range: float = 6000e3, cue_interval_samples: int = 5,
    node_id: str | None = None,
) -> WalkerCANNSidecarComparison:
    """Compare passive CANN operation on Walker truth and filter estimates."""

    if duration <= 0.0 or dt <= 0.0:
        raise ValueError("duration and dt must be positive.")
    audit = run_v14_walker_geometry_audit(
        total_satellites=20, plane_count=10, phasing=1,
        duration=1800.0, dt=30.0, maximum_range=maximum_range,
    )
    case = build_walker_filter_case(
        seed=seed, duration=duration, dt=dt, maximum_range=maximum_range,
        topology=audit.persistent_topology,
        truth_history_by_node=audit.scenario.truth_state_history_by_node,
        topology_type="walker_persistent",
    )
    history = run_network_schmidt_filter(
        timestamps=case["timestamps"],
        initial_state_by_node=case["initial_states"],
        initial_covariance_by_node=case["initial_covariances"],
        topology=case["topology"],
        observation_messages=case["observations"],
        absolute_position_observations=case["absolute_observations"],
        observation_usage="observer_only",
        process_noise_acceleration=1e-8,
        consider_refresh_mode="exact_transport_event_replay",
        state_messages_by_receiver=case["state_messages"],
        replay_history_window=10.0,
        expected_lineage_by_link=case["lineages"],
    )
    selected_node = history.node_ids[0] if node_id is None else str(node_id)
    if selected_node not in history.node_ids:
        raise ValueError(f"Unknown Walker node: {selected_node}")
    elements = audit.scenario.elements_by_node[selected_node]
    frame = OrbitalPlaneFrame.from_raan_inclination(
        raan=elements.raan, inclination=audit.scenario.config.inclination,
    )
    times = np.asarray(history.timestamps)
    truth_states = np.asarray(case["truth"][selected_node])
    estimated_states = np.asarray(
        history.active_state_history_by_node[selected_node]
    )
    truth = run_orbital_phase_sidecar(
        timestamps=times, state_history_eci=truth_states, frame=frame,
        source_id=f"{selected_node}:truth",
    )
    modes = {
        "no_cue": run_orbital_phase_sidecar(
            timestamps=times, state_history_eci=estimated_states, frame=frame,
            source_id=f"{selected_node}:estimate",
        ),
        "periodic_cue": run_orbital_phase_sidecar(
            timestamps=times, state_history_eci=estimated_states, frame=frame,
            cue_interval_samples=cue_interval_samples,
            source_id=f"{selected_node}:estimate",
        ),
        "every_epoch_cue": run_orbital_phase_sidecar(
            timestamps=times, state_history_eci=estimated_states, frame=frame,
            cue_interval_samples=1, source_id=f"{selected_node}:estimate",
        ),
    }
    errors = {
        mode: _circular_difference(trace.decoded_phase, truth.source_phase)
        for mode, trace in modes.items()
    }
    return WalkerCANNSidecarComparison(
        node_id=selected_node, truth=truth,
        estimated_no_cue=modes["no_cue"],
        estimated_periodic_cue=modes["periodic_cue"],
        estimated_every_epoch_cue=modes["every_epoch_cue"],
        phase_rmse_deg_by_mode={
            mode: float(np.rad2deg(np.sqrt(np.mean(error**2))))
            for mode, error in errors.items()
        },
        maximum_phase_error_deg_by_mode={
            mode: float(np.rad2deg(np.max(np.abs(error))))
            for mode, error in errors.items()
        },
        source_position_rmse_m=float(np.sqrt(np.mean(np.sum(
            (estimated_states[:, :3] - truth_states[:, :3]) ** 2, axis=1,
        )))),
    )


def write_walker_cann_comparison_csv(
    result: WalkerCANNSidecarComparison, output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    modes = _mode_histories(result)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "time_s", "truth_phase_rad", "mode", "decoded_phase_rad",
            "truth_error_rad", "source_residual_rad", "bump_concentration",
            "bump_width_rad", "cue_applied", "cross_track_position_m",
        ))
        for index, timestamp in enumerate(result.truth.timestamps):
            for mode, trace in modes.items():
                writer.writerow((
                    timestamp, result.truth.source_phase[index], mode,
                    trace.decoded_phase[index],
                    _circular_difference(
                        trace.decoded_phase[index], result.truth.source_phase[index],
                    ),
                    trace.phase_residual[index], trace.bump_concentration[index],
                    trace.bump_width[index], int(trace.cue_applied[index]),
                    trace.cross_track_position[index],
                ))
    return output


def generate_walker_cann_comparison_figure(
    result: WalkerCANNSidecarComparison, output_path: str | Path,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = _mode_histories(result)
    labels = {
        "no_cue": "estimate rate only",
        "periodic_cue": "estimate + periodic cue",
        "every_epoch_cue": "estimate + every-epoch cue",
    }
    times = result.truth.timestamps
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    unwrapped_truth = np.unwrap(result.truth.source_phase)
    axes[0, 0].plot(
        times, np.rad2deg(unwrapped_truth),
        color="black", linestyle="--", linewidth=2.0, label="truth phase",
    )
    for mode, trace in modes.items():
        axes[0, 0].plot(
            times, np.rad2deg(_unwrap_aligned(
                trace.decoded_phase, unwrapped_truth[0],
            )), label=labels[mode],
        )
    axes[0, 0].set(title="Orbital phase tracking", ylabel="unwrapped phase (deg)")
    axes[0, 0].legend()

    for mode, trace in modes.items():
        error = _circular_difference(trace.decoded_phase, result.truth.source_phase)
        axes[0, 1].plot(times, np.rad2deg(error), label=(
            f"{labels[mode]} | RMSE {result.phase_rmse_deg_by_mode[mode]:.3g} deg"
        ))
    axes[0, 1].set(title="CANN phase error against truth", ylabel="error (deg)")
    axes[0, 1].legend()

    for mode, trace in modes.items():
        axes[1, 0].plot(times, trace.bump_concentration, label=labels[mode])
    axes[1, 0].set(title="Attractor-state quality", ylabel="bump concentration")
    axes[1, 0].legend()

    axes[1, 1].plot(
        times, result.estimated_no_cue.cross_track_position,
        label="estimated-state cross track",
    )
    axes[1, 1].plot(
        times, result.truth.cross_track_position, "--", label="truth cross track",
    )
    axes[1, 1].set(title="Fixed-plane assumption diagnostic",
                   ylabel="cross-track position (m)")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xlabel("time (s)")
        axis.grid(alpha=0.3)
    figure.suptitle(
        f"Walker 20/10/1 CANN sidecar | {result.node_id} | "
        f"position RMSE {result.source_position_rmse_m:.2f} m",
        fontsize=14,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)
    return output


def _mode_histories(result: WalkerCANNSidecarComparison):
    return {
        "no_cue": result.estimated_no_cue,
        "periodic_cue": result.estimated_periodic_cue,
        "every_epoch_cue": result.estimated_every_epoch_cue,
    }


def _circular_difference(actual, expected):
    return (np.asarray(actual) - np.asarray(expected) + np.pi) % (2.0 * np.pi) - np.pi


def _unwrap_aligned(phase, reference_initial):
    unwrapped = np.unwrap(np.asarray(phase, dtype=float))
    turns = np.round((unwrapped[0] - reference_initial) / (2.0 * np.pi))
    return unwrapped - turns * 2.0 * np.pi
