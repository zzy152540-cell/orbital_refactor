from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np

from cooperative.network_schmidt_runner import run_network_schmidt_filter
from experiments.multimodal_cann_preprocessor import (
    MultimodalCANNPreprocessorConfig,
)
from experiments.network_cann_message_adapter import (
    preprocess_network_observation_messages,
)
from experiments.single_satellite_cann_comparison import (
    default_infrared_coupled_cann_config,
)
from experiments.v14_walker_geometry_audit import (
    run_v14_walker_geometry_audit,
)
from experiments.walker_filter_setup import build_walker_filter_case


IMPULSE_BY_MODALITY = {
    "RADAR": np.array([3000.0, 5.0]),
    "INFRARED": np.deg2rad(np.array([5.0, 5.0])),
    "OPTICAL": np.array([0.2, 0.4]),
}


def run_network_cann_impulse_comparison(
    *, duration=20.0, dt=2.0, seed=0, fault_time=8.0,
    fault_modalities=("RADAR", "INFRARED", "OPTICAL"),
    affected_edge_count=None, ring_internal_dt=0.002,
):
    """Compare Network Schmidt with and without CANN on paired faults."""
    case = _build_walker_case(seed=seed, duration=duration, dt=dt)
    faulty, affected_edges = inject_network_impulse_faults(
        case["observations"], fault_time=fault_time,
        fault_modalities=fault_modalities,
        affected_edge_count=affected_edge_count,
    )
    return _run_paired_comparison(
        case, faulty, affected_edges, fault_modalities,
        ring_internal_dt=ring_internal_dt,
    )


def run_network_cann_degradation_comparison(
    *, mode, duration=50.0, dt=2.0, seed=0,
    fault_modalities=("INFRARED", "OPTICAL"), affected_edge_count=5,
    drift_start=8.0, drift_step_fraction=0.05,
    outage_start=10.0, outage_end=34.0, ring_internal_dt=0.002,
):
    """Run gradual-drift or faulty-recovery paired network comparisons."""
    case = _build_walker_case(seed=seed, duration=duration, dt=dt)
    if mode == "gradual_drift":
        faulty, affected_edges = inject_network_gradual_drift(
            case["observations"], start_time=drift_start, dt=dt,
            step_fraction=drift_step_fraction,
            fault_modalities=fault_modalities,
            affected_edge_count=affected_edge_count,
        )
    elif mode == "faulty_recovery":
        faulty, affected_edges = inject_network_faulty_recovery(
            case["observations"], outage_start=outage_start,
            outage_end=outage_end, dt=dt,
            fault_modalities=fault_modalities,
            affected_edge_count=affected_edge_count,
        )
    else:
        raise ValueError(f"Unknown network degradation mode: {mode}")
    result = _run_paired_comparison(
        case, faulty, affected_edges, fault_modalities,
        ring_internal_dt=ring_internal_dt,
    )
    return {**result, "mode": mode}


def _build_walker_case(*, seed, duration, dt):
    audit = run_v14_walker_geometry_audit(
        total_satellites=20, plane_count=10, phasing=1,
        duration=1800.0, dt=30.0, maximum_range=6000e3,
    )
    case = build_walker_filter_case(
        seed=seed, duration=duration, dt=dt, maximum_range=6000e3,
        topology=audit.persistent_topology,
        truth_history_by_node=audit.scenario.truth_state_history_by_node,
        topology_type="walker_persistent",
    )
    case["comparison_seed"] = int(seed)
    return case


def _run_paired_comparison(
    case, faulty, affected_edges, fault_modalities, *, ring_internal_dt,
):
    run_arguments = dict(
        timestamps=case["timestamps"],
        initial_state_by_node=case["initial_states"],
        initial_covariance_by_node=case["initial_covariances"],
        topology=case["topology"],
        absolute_position_observations=case["absolute_observations"],
        observation_usage="observer_only",
        process_noise_acceleration=1e-8,
        consider_refresh_mode="exact_transport_event_replay",
        state_messages_by_receiver=case["state_messages"],
        replay_history_window=10.0,
        expected_lineage_by_link=case["lineages"],
    )
    started = perf_counter()
    baseline = run_network_schmidt_filter(
        observation_messages=faulty, **run_arguments,
    )
    baseline_seconds = perf_counter() - started
    coupled = default_infrared_coupled_cann_config()
    config = MultimodalCANNPreprocessorConfig(
        infrared_coupled_config=replace(
            coupled,
            ring=replace(coupled.ring, internal_dt=ring_internal_dt),
        ),
    )
    started = perf_counter()
    processed = preprocess_network_observation_messages(
        faulty, timestamps=case["timestamps"], config=config,
    )
    preprocessing_seconds = perf_counter() - started
    started = perf_counter()
    cann = run_network_schmidt_filter(
        observation_messages=processed, **run_arguments,
    )
    cann_filter_seconds = perf_counter() - started
    source = {item.information_id: item for item in faulty}
    affected_observers = {observer for observer, _ in affected_edges}
    return {
        "seed": case["comparison_seed"],
        "fault_modalities": tuple(str(name).upper() for name in fault_modalities),
        "affected_edges": affected_edges,
        "fault_count": sum(
            bool(item.metadata.get("injected_network_fault", False))
            for item in faulty
        ),
        "dropped_message_count": len(case["observations"]) - len(faulty),
        "cann_modified_count": sum(
            not np.array_equal(
                item.measurement, source[item.information_id].measurement,
            )
            for item in processed
        ),
        "cann_invalidated_count": sum(
            source[item.information_id].valid_flag and not item.valid_flag
            for item in processed
        ),
        "baseline_position_rmse_m": _position_rmse(
            baseline, case["truth"], baseline.node_ids,
        ),
        "cann_position_rmse_m": _position_rmse(
            cann, case["truth"], cann.node_ids,
        ),
        "baseline_affected_position_rmse_m": _position_rmse(
            baseline, case["truth"], affected_observers,
        ),
        "cann_affected_position_rmse_m": _position_rmse(
            cann, case["truth"], affected_observers,
        ),
        "baseline_rejection_count": _rejection_count(baseline),
        "cann_rejection_count": _rejection_count(cann),
        "baseline_filter_seconds": float(baseline_seconds),
        "cann_preprocessing_seconds": float(preprocessing_seconds),
        "cann_filter_seconds": float(cann_filter_seconds),
    }


def inject_network_impulse_faults(
    messages, *, fault_time, fault_modalities, affected_edge_count=None,
):
    selected_modalities = {str(name).upper() for name in fault_modalities}
    unknown = selected_modalities - set(IMPULSE_BY_MODALITY)
    if unknown:
        raise ValueError(f"Unsupported fault modalities: {sorted(unknown)}")
    edges = sorted({(item.observer_id, item.target_id) for item in messages})
    selected_edges = tuple(
        edges if affected_edge_count is None else edges[:affected_edge_count]
    )
    selected_edge_set = set(selected_edges)
    result = []
    for item in messages:
        inject = bool(
            (item.observer_id, item.target_id) in selected_edge_set
            and item.modality.upper() in selected_modalities
            and np.isclose(item.timestamp, fault_time, rtol=0.0, atol=1e-12)
        )
        result.append(
            replace(
                item,
                measurement=(
                    np.asarray(item.measurement, dtype=float)
                    + IMPULSE_BY_MODALITY[item.modality.upper()]
                ),
                metadata={**item.metadata, "injected_network_fault": True},
            )
            if inject else item
        )
    return result, selected_edges


def inject_network_gradual_drift(
    messages, *, start_time, dt, step_fraction, fault_modalities,
    affected_edge_count=None,
):
    selected_modalities, selected_edges = _fault_selection(
        messages, fault_modalities, affected_edge_count,
    )
    selected_edge_set = set(selected_edges)
    result = []
    for item in messages:
        step = max(0.0, (float(item.timestamp) - float(start_time)) / float(dt))
        inject = bool(
            step > 0.0
            and (item.observer_id, item.target_id) in selected_edge_set
            and item.modality.upper() in selected_modalities
        )
        result.append(
            replace(
                item,
                measurement=(
                    np.asarray(item.measurement, dtype=float)
                    + step * float(step_fraction)
                    * IMPULSE_BY_MODALITY[item.modality.upper()]
                ),
                metadata={**item.metadata, "injected_network_fault": True},
            )
            if inject else item
        )
    return result, selected_edges


def inject_network_faulty_recovery(
    messages, *, outage_start, outage_end, dt, fault_modalities,
    affected_edge_count=None,
):
    del dt
    selected_modalities, selected_edges = _fault_selection(
        messages, fault_modalities, affected_edge_count,
    )
    selected_edge_set = set(selected_edges)
    recovery_times = {}
    for edge in selected_edges:
        for modality in selected_modalities:
            times = sorted({
                float(item.timestamp) for item in messages
                if (item.observer_id, item.target_id) == edge
                and item.modality.upper() == modality
                and float(item.timestamp) > float(outage_end)
            })
            recovery_times[(*edge, modality)] = tuple(times[:2])
    result = []
    for item in messages:
        edge = (item.observer_id, item.target_id)
        selected = bool(
            edge in selected_edge_set
            and item.modality.upper() in selected_modalities
        )
        if selected and outage_start <= item.timestamp <= outage_end:
            continue
        recovery = recovery_times.get((*edge, item.modality.upper()), ())
        if item.timestamp in recovery:
            sign = 1.0 if item.timestamp == recovery[0] else -1.0
            item = replace(
                item,
                measurement=(
                    np.asarray(item.measurement, dtype=float)
                    + sign * IMPULSE_BY_MODALITY[item.modality.upper()]
                ),
                metadata={**item.metadata, "injected_network_fault": True},
            )
        result.append(item)
    return result, selected_edges


def _fault_selection(messages, fault_modalities, affected_edge_count):
    selected_modalities = {str(name).upper() for name in fault_modalities}
    unknown = selected_modalities - set(IMPULSE_BY_MODALITY)
    if unknown:
        raise ValueError(f"Unsupported fault modalities: {sorted(unknown)}")
    edges = sorted({(item.observer_id, item.target_id) for item in messages})
    selected_edges = tuple(
        edges if affected_edge_count is None else edges[:affected_edge_count]
    )
    return selected_modalities, selected_edges


def _position_rmse(history, truth, node_ids):
    errors = [
        np.asarray(history.active_state_history_by_node[node])[:, :3]
        - np.asarray(truth[node])[:, :3]
        for node in node_ids
    ]
    return float(np.sqrt(np.mean([
        np.sum(error * error, axis=1) for error in errors
    ])))


def _rejection_count(history):
    return sum(
        record.status != "ACCEPTED"
        for node in history.node_ids
        for epoch in history.integrity_history_by_node[node]
        for record in epoch.values()
    )
