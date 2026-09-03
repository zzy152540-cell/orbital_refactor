from __future__ import annotations

import numpy as np

from adapters.synthetic_measurement_adapter import (
    create_single_satellite_visibility_flags,
    create_infrared_observations,
    create_optical_observations,
    create_radar_observations,
)
from cooperative.multi_sat_pipeline import run_cooperative_pipeline
from experiments.multimodal_cann_preprocessor import MultimodalCANNPreprocessor
from experiments.recovery_confirmation_preprocessor import (
    RecoveryConfirmationPreprocessor,
)
from orbital_core.constants import R_EARTH
from orbital_core.coordinates import state_history_eci_to_spri
from orbital_core.attitude import quat_conjugate_wxyz
from orbital_core.orbit_elements import keplerian_to_eci
from scenarios.multi_satellite_scenario import generate_cooperative_scenario
from scenarios.measurement_visibility import (
    VisibilityConfig, VisibilityTemporalFilterConfig,
)


def run_multi_satellite_cann_comparison(
    *, duration=300.0, dt=2.0, seed=0, fault_by_node=None,
    visibility_by_modality=None, temporal_filter_by_modality=None,
    dropout_windows_by_node=None, recovery_fault_samples=0,
    include_confirmation_baseline=False,
):
    """Run a paired three-observer, three-modal baseline/CANN comparison."""
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt)
    scenario = _build_scenario(timestamps)
    observers = scenario.observer_trajectories
    visibility = visibility_by_modality or {
        "RADAR": VisibilityConfig(maximum_range=175e3),
        "INFRARED": VisibilityConfig(maximum_range=120e3),
        "OPTICAL": VisibilityConfig(
            maximum_range=75e3,
            field_of_view_half_angle=np.deg2rad(89.0),
            boresight_axis=(0.0, 0.0, 1.0),
        ),
    }
    temporal = temporal_filter_by_modality or {
        name: VisibilityTemporalFilterConfig(acquisition_epochs=2, loss_epochs=2)
        for name in visibility
    }
    streams = {}
    visibility_rates = {}
    recovery_fault_schedule = {}
    for i, node_id in enumerate(observers):
        flags = _visibility_flags(scenario, node_id, visibility, temporal)
        flags, node_recovery = _apply_dropout_windows(
            flags, timestamps,
            (dropout_windows_by_node or {}).get(node_id, {}),
            int(recovery_fault_samples),
        )
        if node_recovery:
            recovery_fault_schedule[node_id] = node_recovery
        streams[node_id] = _make_stream(
            scenario, node_id, seed * 101 + i, flags,
        )
        visibility_rates[node_id] = {
            name: float(np.mean(values)) for name, values in flags.items()
        }
    combined_faults = _merge_fault_schedules(
        fault_by_node or {}, recovery_fault_schedule,
    )
    injected_fault_counts = _inject_faults(
        streams, timestamps, combined_faults,
    )
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
    confirmation = None
    if include_confirmation_baseline:
        confirmation = run_cooperative_pipeline(
            scenario=scenario, observations_by_node=streams,
            initial_error_by_node=initial_errors,
            measurement_preprocessor_by_node={
                node_id: RecoveryConfirmationPreprocessor()
                for node_id in observers
            },
        )
    diagnostics = _fault_diagnostics(
        scenario, baseline, processed, combined_faults, dt, confirmation,
    )
    return {
        "seed": int(seed), "timestamps": timestamps,
        "fault_by_node": fault_by_node or {},
        "baseline": baseline, "confirmation": confirmation, "cann": processed,
        "summary": {
            "baseline_position_rmse_m": baseline.metrics.cooperative_position_rmse,
            "cann_position_rmse_m": processed.metrics.cooperative_position_rmse,
            "confirmation_position_rmse_m": (
                None if confirmation is None
                else confirmation.metrics.cooperative_position_rmse
            ),
            "cann_minus_confirmation_position_rmse_m": (
                None if confirmation is None else (
                    processed.metrics.cooperative_position_rmse
                    - confirmation.metrics.cooperative_position_rmse
                )
            ),
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
            "visibility_rate_by_node": visibility_rates,
            "injected_fault_count_by_node_modality": injected_fault_counts,
            "recovery_fault_times_by_node_modality": recovery_fault_schedule,
        },
    }


def _apply_dropout_windows(flags, timestamps, windows_by_modality, recovery_count):
    result = {name: np.asarray(values, dtype=bool).copy()
              for name, values in flags.items()}
    recovery = {}
    for modality, windows in windows_by_modality.items():
        name = str(modality).upper()
        if name not in result:
            raise ValueError(f"Unknown dropout modality: {modality}")
        for start, end in windows:
            if end < start:
                raise ValueError("Dropout window end cannot precede start.")
            result[name][(timestamps >= float(start)) & (timestamps <= float(end))] = False
            if recovery_count:
                candidates = np.flatnonzero(
                    (timestamps > float(end)) & result[name]
                )[:recovery_count]
                recovery.setdefault(name.lower(), []).extend(
                    float(timestamps[index]) for index in candidates
                )
    return result, {name: tuple(times) for name, times in recovery.items()}


def _merge_fault_schedules(primary, secondary):
    merged = {
        node: {modality: tuple(times) for modality, times in modalities.items()}
        for node, modalities in primary.items()
    }
    for node, modalities in secondary.items():
        target = merged.setdefault(node, {})
        for modality, times in modalities.items():
            target[modality] = (*target.get(modality, ()), *times)
    return merged


def audit_multi_satellite_geometry(*, duration=1800.0, dt=2.0):
    """Return range statistics without generating measurements or filtering."""
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt)
    scenario = _build_scenario(timestamps)
    result = {}
    for node_id, relative in scenario.relative_state_eci_by_node.items():
        distance = np.linalg.norm(relative[:, :3], axis=1) / 1e3
        result[node_id] = {
            "minimum_range_km": float(np.min(distance)),
            "median_range_km": float(np.median(distance)),
            "maximum_range_km": float(np.max(distance)),
        }
    return result


def _build_scenario(timestamps):
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
    return generate_cooperative_scenario(
        timestamps=timestamps, target_id="target",
        target_initial_state_eci=target,
        observer_initial_states_eci=observers,
    )


def _fault_diagnostics(
    scenario, baseline, processed, schedule, dt, confirmation=None,
):
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
        runs = [("baseline", baseline), ("cann", processed)]
        if confirmation is not None:
            runs.insert(1, ("confirmation", confirmation))
        for label, run in runs:
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


def _visibility_flags(scenario, node_id, visibility, temporal):
    observer = scenario.observer_trajectories[node_id]
    attitude_i2s = np.vstack([
        quat_conjugate_wxyz(quaternion)
        for quaternion in observer.q_eci2pri_history
    ])
    result = create_single_satellite_visibility_flags(
        timestamps=scenario.timestamps,
        chief_state_history_eci=observer.state_history_eci,
        relative_target_state_history_eci=(
            scenario.relative_state_eci_by_node[node_id]
        ),
        visibility_by_modality=visibility,
        temporal_filter_by_modality=temporal,
        attitude_history_i2sensor_wxyz=attitude_i2s,
        observer_id=node_id, target_id=scenario.target_id,
    )
    return result.valid_flags_by_modality


def _make_stream(scenario, node_id, seed, valid_flags):
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
            valid_flags=valid_flags["OPTICAL"],
        ),
        *create_infrared_observations(
            **common, relative_position_spri=relative[:, :3],
            covariance=np.diag(np.deg2rad([0.05, 0.05])) ** 2,
            valid_flags=valid_flags["INFRARED"],
        ),
        *create_radar_observations(
            **common, relative_position_spri=relative[:, :3],
            relative_velocity_spri=relative[:, 3:],
            covariance=np.diag([30.0, 0.05]) ** 2,
            valid_flags=valid_flags["RADAR"],
        ),
    ]


def _inject_faults(streams, timestamps, schedule):
    offsets = {
        "infrared": np.deg2rad([5.0, 5.0]),
        "radar": np.array([3000.0, 5.0]),
        "optical": np.array([0.2, 0.4]),
    }
    counts = {}
    for node_id, by_modality in schedule.items():
        if node_id not in streams:
            raise ValueError(f"Unknown fault node: {node_id}")
        for modality, fault_times in by_modality.items():
            name = modality.lower()
            if name not in offsets:
                raise ValueError(f"Unknown fault modality: {modality}")
            items = [item for item in streams[node_id] if item.modality.lower() == name]
            count = 0
            for fault_index, fault_time in enumerate(fault_times):
                index = int(np.argmin(np.abs(timestamps - float(fault_time))))
                if not items[index].valid_flag:
                    continue
                sign = 1.0 if fault_index % 2 == 0 else -1.0
                items[index].measurement = items[index].measurement + sign * offsets[name]
                items[index].metadata = {
                    **items[index].metadata, "injected_multisat_fault": True,
                }
                count += 1
            counts.setdefault(node_id, {})[name] = count
    return counts
