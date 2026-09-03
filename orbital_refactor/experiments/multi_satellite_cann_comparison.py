from __future__ import annotations

import numpy as np

from adapters.synthetic_measurement_adapter import (
    create_infrared_observations,
    create_optical_observations,
    create_radar_observations,
)
from cooperative.multi_sat_pipeline import run_cooperative_pipeline
from experiments.multimodal_cann_preprocessor import MultimodalCANNPreprocessor
from orbital_core.constants import R_EARTH
from orbital_core.coordinates import state_history_eci_to_spri
from orbital_core.orbit_elements import keplerian_to_eci
from scenarios.multi_satellite_scenario import generate_cooperative_scenario


def run_multi_satellite_cann_comparison(
    *, duration=300.0, dt=2.0, seed=0, fault_by_node=None,
):
    """Run a paired three-observer, three-modal baseline/CANN comparison."""
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt)
    target = keplerian_to_eci(
        R_EARTH + 700e3, 0.001, np.deg2rad(55.0),
        np.deg2rad(15.0), 0.0, np.deg2rad(8.0),
    )
    observers = {
        f"sat_{i + 1:02d}": keplerian_to_eci(
            R_EARTH + (702 + i) * 1e3, 0.0012,
            np.deg2rad(54.5 + 0.2 * i), np.deg2rad(14.5),
            0.0, np.deg2rad(7.0 + 0.4 * i),
        )
        for i in range(3)
    }
    scenario = generate_cooperative_scenario(
        timestamps=timestamps, target_id="target",
        target_initial_state_eci=target,
        observer_initial_states_eci=observers,
    )
    streams = {
        node_id: _make_stream(scenario, node_id, seed * 101 + i)
        for i, node_id in enumerate(observers)
    }
    _inject_faults(streams, timestamps, fault_by_node or {})
    initial_errors = {
        node_id: np.array([50., -40., 30., .05, -.04, .03])
        for node_id in observers
    }
    baseline = run_cooperative_pipeline(
        scenario=scenario, observations_by_node=streams,
        initial_error_by_node=initial_errors,
    )
    processed = run_cooperative_pipeline(
        scenario=scenario, observations_by_node=streams,
        initial_error_by_node=initial_errors,
        measurement_preprocessor_by_node={
            node_id: MultimodalCANNPreprocessor() for node_id in observers
        },
    )
    diagnostics = _fault_diagnostics(
        scenario, baseline, processed, fault_by_node or {}, dt,
    )
    return {
        "seed": int(seed), "timestamps": timestamps,
        "fault_by_node": fault_by_node or {},
        "baseline": baseline, "cann": processed,
        "summary": {
            "baseline_position_rmse_m": baseline.metrics.cooperative_position_rmse,
            "cann_position_rmse_m": processed.metrics.cooperative_position_rmse,
            "position_change_m": (
                processed.metrics.cooperative_position_rmse
                - baseline.metrics.cooperative_position_rmse
            ),
            "baseline_velocity_rmse_mps": baseline.metrics.cooperative_velocity_rmse,
            "cann_velocity_rmse_mps": processed.metrics.cooperative_velocity_rmse,
            "velocity_change_mps": (
                processed.metrics.cooperative_velocity_rmse
                - baseline.metrics.cooperative_velocity_rmse
            ),
            "baseline_local_position_rmse_m": baseline.metrics.local_position_rmse,
            "cann_local_position_rmse_m": processed.metrics.local_position_rmse,
            "fault_diagnostics": diagnostics,
        },
    }


def _fault_diagnostics(scenario, baseline, processed, schedule, dt):
    timestamps = np.asarray(scenario.timestamps, dtype=float)
    truth = np.asarray(scenario.target_trajectory.state_history_eci, dtype=float)
    result = {}
    for node_id, by_modality in schedule.items():
        fault_times = [float(t) for times in by_modality.values() for t in times]
        if not fault_times:
            continue
        mask = np.zeros(timestamps.size, dtype=bool)
        for fault_time in fault_times:
            mask |= np.abs(timestamps - fault_time) <= max(10.0, 2.0 * dt)
        entry = {}
        for label, run in (("baseline", baseline), ("cann", processed)):
            local_error = np.linalg.norm(
                run.local_absolute_state_history_by_node[node_id][:, :3]
                - truth[:, :3], axis=1,
            )
            fused_error = np.linalg.norm(
                run.cooperative_history.state_history_eci[:, :3]
                - truth[:, :3], axis=1,
            )
            weights = np.asarray([
                epoch.get(node_id, 0.0)
                for epoch in run.cooperative_history.node_weight_history
            ])
            entry[label] = {
                "local_position_rmse_m": float(np.sqrt(np.mean(local_error[mask] ** 2))),
                "fused_position_rmse_m": float(np.sqrt(np.mean(fused_error[mask] ** 2))),
                "mean_ci_weight": float(np.mean(weights[mask])),
                "max_ci_weight": float(np.max(weights[mask])),
            }
        result[node_id] = entry
    return result


def _make_stream(scenario, node_id, seed):
    timestamps = scenario.timestamps
    observer = scenario.observer_trajectories[node_id]
    relative = state_history_eci_to_spri(
        scenario.relative_state_eci_by_node[node_id],
        observer.q_eci2pri_history,
    )
    rng = np.random.default_rng(seed)
    common = dict(
        timestamps=timestamps, observer_id=node_id,
        target_id=scenario.target_id, rng=rng,
    )
    return [
        *create_optical_observations(
            **common, relative_position_spri=relative[:, :3],
            covariance=np.diag([2e-4, 2e-4]) ** 2,
        ),
        *create_infrared_observations(
            **common, relative_position_spri=relative[:, :3],
            covariance=np.diag(np.deg2rad([0.05, 0.05])) ** 2,
        ),
        *create_radar_observations(
            **common, relative_position_spri=relative[:, :3],
            relative_velocity_spri=relative[:, 3:],
            covariance=np.diag([30.0, 0.05]) ** 2,
        ),
    ]


def _inject_faults(streams, timestamps, schedule):
    offsets = {
        "infrared": np.deg2rad([5.0, 5.0]),
        "radar": np.array([3000.0, 5.0]),
        "optical": np.array([0.2, 0.4]),
    }
    for node_id, by_modality in schedule.items():
        if node_id not in streams:
            raise ValueError(f"Unknown fault node: {node_id}")
        for modality, fault_times in by_modality.items():
            name = modality.lower()
            if name not in offsets:
                raise ValueError(f"Unknown fault modality: {modality}")
            items = [item for item in streams[node_id] if item.modality.lower() == name]
            for fault_index, fault_time in enumerate(fault_times):
                index = int(np.argmin(np.abs(timestamps - float(fault_time))))
                if not items[index].valid_flag:
                    continue
                sign = 1.0 if fault_index % 2 == 0 else -1.0
                items[index].measurement = items[index].measurement + sign * offsets[name]
                items[index].metadata = {
                    **items[index].metadata, "injected_multisat_fault": True,
                }
