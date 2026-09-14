"""P06 algorithm-only update-frequency development pre-scan."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from cooperative.network_schmidt_runner import run_network_schmidt_filter
from experiments.v14_walker_geometry_audit import run_v14_walker_geometry_audit
from experiments.walker_filter_setup import build_walker_filter_case
from visualization.cann_history_adapter import build_cann_visualization_history


@dataclass(frozen=True)
class UpdateFrequencyRecord:
    seed: int
    profile: str
    cann_node_count: int
    epoch_count: int
    filter_seconds: float
    cann_seconds: float
    total_seconds: float
    amortized_epoch_ms: float
    mean_update_frequency_hz: float
    realtime_factor: float
    latency_granularity: str
    epoch_latency_ms: tuple[float, ...]


@dataclass(frozen=True)
class UpdateFrequencyGroup:
    profile: str
    run_count: int
    mean_update_frequency_hz: float
    mean_epoch_latency_ms: float
    p95_epoch_latency_ms: float
    p99_epoch_latency_ms: float
    maximum_epoch_latency_ms: float
    over_200ms_epoch_fraction: float
    mean_realtime_factor: float
    threshold_met: bool
    latency_granularity: str


@dataclass(frozen=True)
class UpdateFrequencyReport:
    walker_definition: tuple[int, int, int]
    duration_seconds: float
    dt_seconds: float
    threshold_hz: float
    timing_scope: str
    timing_granularity: str
    formal_profile: str
    threshold_met: bool
    formal_latency_distribution_available: bool
    passed: bool
    records: tuple[UpdateFrequencyRecord, ...]
    groups: tuple[UpdateFrequencyGroup, ...]


def run_external_update_frequency_acceptance(
    *, seeds=(0, 1, 2, 3, 4), duration=20.0, dt=0.2,
    maximum_range=6000e3, threshold_hz=5.0,
):
    seeds = tuple(map(int, seeds))
    if not seeds or duration <= 0 or dt <= 0:
        raise ValueError("seeds, duration and dt must be valid and positive.")
    audit = run_v14_walker_geometry_audit(
        total_satellites=20, plane_count=10, phasing=1,
        duration=1800.0, dt=30.0, maximum_range=maximum_range,
    )
    records = []
    for seed in seeds:
        case = build_walker_filter_case(
            seed=seed, duration=duration, dt=dt, maximum_range=maximum_range,
            topology=audit.persistent_topology,
            truth_history_by_node=audit.scenario.truth_state_history_by_node,
            topology_type="walker_persistent",
        )
        started = perf_counter()
        filter_epoch_seconds = []
        history = run_network_schmidt_filter(
            timestamps=case["timestamps"], initial_state_by_node=case["initial_states"],
            initial_covariance_by_node=case["initial_covariances"],
            topology=case["topology"], observation_messages=case["observations"],
            absolute_position_observations=case["absolute_observations"],
            observation_usage="observer_only", process_noise_acceleration=1e-8,
            consider_refresh_mode="exact_transport_event_replay",
            state_messages_by_receiver=case["state_messages"],
            replay_history_window=max(10.0, 4.0 * dt),
            expected_lineage_by_link=case["lineages"],
            epoch_timing_callback=lambda _, __, elapsed: (
                filter_epoch_seconds.append(elapsed)
            ),
        )
        filter_seconds = perf_counter() - started
        nodes = tuple(history.node_ids)
        for profile, selected in (
            ("no_cann", ()), ("representative_cann", nodes[:1]),
            ("all_node_cann", nodes),
        ):
            cann_seconds = 0.0
            if selected:
                started = perf_counter()
                build_cann_visualization_history(
                    history=history, initial_state_by_node=case["initial_states"],
                    absolute_position_observations=case["absolute_observations"],
                    cann_node_ids=selected,
                )
                cann_seconds = perf_counter() - started
            total = filter_seconds + cann_seconds
            epochs = len(case["timestamps"])
            cann_amortized_seconds = cann_seconds / epochs
            epoch_latency_ms = tuple(
                1000.0 * (elapsed + cann_amortized_seconds)
                for elapsed in filter_epoch_seconds
            )
            granularity = (
                "measured_per_epoch"
                if not selected
                else "filter_per_epoch_plus_run_amortized_cann"
            )
            records.append(UpdateFrequencyRecord(
                seed=seed, profile=profile, cann_node_count=len(selected),
                epoch_count=epochs, filter_seconds=filter_seconds,
                cann_seconds=cann_seconds, total_seconds=total,
                amortized_epoch_ms=1000.0 * total / epochs,
                mean_update_frequency_hz=epochs / total,
                realtime_factor=duration / total,
                latency_granularity=granularity,
                epoch_latency_ms=epoch_latency_ms,
            ))
    groups = tuple(_summarize(name, records, threshold_hz) for name in (
        "no_cann", "representative_cann", "all_node_cann",
    ))
    return UpdateFrequencyReport(
        walker_definition=(20, 10, 1), duration_seconds=float(duration),
        dt_seconds=float(dt), threshold_hz=float(threshold_hz),
        timing_scope="algorithm_plus_optional_cann_without_gui_or_recording",
        timing_granularity=(
            "core_filter_per_epoch; optional_cann_run_amortized"
        ),
        formal_profile="no_cann",
        threshold_met=all(group.threshold_met for group in groups),
        formal_latency_distribution_available=True,
        passed=groups[0].threshold_met,
        records=tuple(records), groups=groups,
    )


def _summarize(profile, records, threshold):
    selected = tuple(item for item in records if item.profile == profile)
    elapsed = np.array([
        value
        for item in selected
        for value in item.epoch_latency_ms
    ])
    frequencies = np.array([item.mean_update_frequency_hz for item in selected])
    return UpdateFrequencyGroup(
        profile=profile, run_count=len(selected),
        mean_update_frequency_hz=float(np.mean(frequencies)),
        mean_epoch_latency_ms=float(np.mean(elapsed)),
        p95_epoch_latency_ms=float(np.percentile(elapsed, 95)),
        p99_epoch_latency_ms=float(np.percentile(elapsed, 99)),
        maximum_epoch_latency_ms=float(np.max(elapsed)),
        over_200ms_epoch_fraction=float(np.mean(elapsed > 200.0)),
        mean_realtime_factor=float(np.mean([item.realtime_factor for item in selected])),
        threshold_met=bool(np.mean(frequencies) >= threshold),
        latency_granularity=selected[0].latency_granularity,
    )


def save_external_update_frequency_report(report, output):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
