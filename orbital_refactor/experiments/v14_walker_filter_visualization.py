from __future__ import annotations

from pathlib import Path

import numpy as np

from cooperative.network_schmidt_runner import run_network_schmidt_filter
from experiments.v14_exact_transport_scale_scan import build_exact_transport_case
from experiments.v14_walker_geometry_audit import run_v14_walker_geometry_audit
from scenarios.measurement_visibility import VisibilityConfig


def generate_v14_walker_filter_visualization(
    output_path: str | Path,
    *, duration: float = 600.0, dt: float = 2.0, seed: int = 0,
    maximum_range: float = 6000e3,
) -> Path:
    """Run one Walker case and save a compact truth/consistency overview."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    audit = run_v14_walker_geometry_audit(
        total_satellites=20, plane_count=10, phasing=1,
        duration=1800.0, dt=30.0, maximum_range=maximum_range,
    )
    initial_truth = {
        node: history[0]
        for node, history in audit.scenario.truth_state_history_by_node.items()
    }
    modalities = ("RADAR", "INFRARED", "OPTICAL")
    case = build_exact_transport_case(
        seed=seed, duration=duration, dt=dt,
        range_sigma=2.0, range_rate_sigma=0.05,
        az_el_sigma=np.deg2rad(0.05), optical_sigma=1e-3,
        absolute_sigma=3.0, process_noise_acceleration=1e-8,
        packet_loss=0.0, delay=0.0, acknowledge_messages=True,
        node_count=20, topology_type="walker_persistent",
        topology_override=audit.persistent_topology,
        truth_initial_state_by_node=initial_truth,
        visibility_by_modality={
            modality: VisibilityConfig(maximum_range=maximum_range)
            for modality in modalities
        },
        relative_modalities=modalities,
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
    times = np.asarray(history.timestamps)
    position_error_rms, position_three_sigma, mean_nees = _fleet_timelines(
        history, case["truth"],
    )
    nis_by_modality = _nis_timelines(history, modalities)

    figure = plt.figure(figsize=(14, 9), constrained_layout=True)
    orbit_axis = figure.add_subplot(2, 2, 1, projection="3d")
    error_axis = figure.add_subplot(2, 2, 2)
    nees_axis = figure.add_subplot(2, 2, 3)
    nis_axis = figure.add_subplot(2, 2, 4)

    for node in history.node_ids:
        truth = case["truth"][node][:, :3] / 1000.0
        orbit_axis.plot(truth[:, 0], truth[:, 1], truth[:, 2], linewidth=0.8)
        orbit_axis.scatter(*truth[0], s=8)
    orbit_axis.set_title("Walker 20/10/1 truth trajectories")
    orbit_axis.set_xlabel("ECI x (km)")
    orbit_axis.set_ylabel("ECI y (km)")
    orbit_axis.set_zlabel("ECI z (km)")

    error_axis.plot(times, position_error_rms, label="fleet position error RMS")
    error_axis.plot(
        times, position_three_sigma, "--", label="equivalent 3-sigma bound",
    )
    error_axis.set_title("Fleet position estimation")
    error_axis.set_xlabel("time (s)")
    error_axis.set_ylabel("position (m)")
    error_axis.grid(alpha=0.3)
    error_axis.legend()

    nees_axis.plot(times, mean_nees, color="tab:purple", label="fleet mean NEES")
    nees_axis.axhline(6.0, color="black", linestyle="--", label="expected mean = 6")
    nees_axis.axhspan(1.2373442458, 14.4493753354, alpha=0.12, color="tab:green",
                     label="single-state 95% interval")
    nees_axis.set_title("Consistency over time")
    nees_axis.set_xlabel("time (s)")
    nees_axis.set_ylabel("NEES")
    nees_axis.grid(alpha=0.3)
    nees_axis.legend()

    for modality, values in nis_by_modality.items():
        nis_axis.plot(times, values, label=modality, linewidth=1.0)
    nis_axis.axhline(2.0, color="black", linestyle="--", label="expected mean = 2")
    nis_axis.set_title("Mean innovation consistency by modality")
    nis_axis.set_xlabel("time (s)")
    nis_axis.set_ylabel("mean NIS")
    nis_axis.grid(alpha=0.3)
    nis_axis.legend()

    figure.suptitle(
        "V14 Walker-Delta 20/10/1 | static persistent topology | seed 0",
        fontsize=14,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=160)
    plt.close(figure)
    return target


def _fleet_timelines(history, truth):
    error_rms = []
    three_sigma = []
    mean_nees = []
    for index in range(len(history.timestamps)):
        errors = []
        bounds = []
        nees = []
        for node in history.node_ids:
            error = history.active_state_history_by_node[node][index] - truth[node][index]
            covariance = history.active_covariance_history_by_node[node][index]
            errors.append(error[:3])
            bounds.append(3.0 * np.sqrt(np.trace(covariance[:3, :3])))
            nees.append(float(error @ np.linalg.solve(covariance, error)))
        error_rms.append(float(np.sqrt(np.mean(np.sum(np.square(errors), axis=1)))))
        three_sigma.append(float(np.mean(bounds)))
        mean_nees.append(float(np.mean(nees)))
    return np.asarray(error_rms), np.asarray(three_sigma), np.asarray(mean_nees)


def _nis_timelines(history, modalities):
    result = {modality: [] for modality in modalities}
    for index in range(len(history.timestamps)):
        values = {modality: [] for modality in modalities}
        for node in history.node_ids:
            for information_id, nis in history.nis_history_by_node[node][index].items():
                lowered = information_id.lower()
                for modality in modalities:
                    if f":{modality.lower()}:" in lowered:
                        values[modality].append(nis)
                        break
        for modality in modalities:
            result[modality].append(
                float(np.mean(values[modality])) if values[modality] else np.nan
            )
    return {key: np.asarray(value) for key, value in result.items()}
