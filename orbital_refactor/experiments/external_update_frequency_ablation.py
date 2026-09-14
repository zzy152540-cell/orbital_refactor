"""Controlled P06 timing ablation for the Walker-20 production path."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from cooperative.network_schmidt_runner import run_network_schmidt_filter
from experiments.v14_walker_geometry_audit import run_v14_walker_geometry_audit
from experiments.walker_filter_setup import build_walker_filter_case


@dataclass(frozen=True)
class TimingAblationRecord:
    seed: int
    profile: str
    epoch_count: int
    total_seconds: float
    mean_frequency_hz: float
    mean_epoch_latency_ms: float
    p95_epoch_latency_ms: float
    p99_epoch_latency_ms: float
    maximum_epoch_latency_ms: float


@dataclass(frozen=True)
class TimingAblationReport:
    walker_definition: tuple[int, int, int]
    duration_seconds: float
    dt_seconds: float
    timing_scope: str
    records: tuple[TimingAblationRecord, ...]
    mean_latency_by_profile_ms: dict[str, float]
    exact_replay_increment_ms: float
    relative_measurement_increment_ms: float


def run_external_update_frequency_ablation(
    *, seeds=(0,), duration=4.0, dt=0.2, maximum_range=6000e3,
) -> TimingAblationReport:
    """Measure nested timing profiles without changing production defaults."""

    seeds = tuple(map(int, seeds))
    if not seeds or duration <= 0.0 or dt <= 0.0:
        raise ValueError("seeds, duration and dt must be valid and positive.")
    audit = run_v14_walker_geometry_audit(
        total_satellites=20, plane_count=10, phasing=1,
        duration=1800.0, dt=30.0, maximum_range=maximum_range,
    )
    records = []
    for seed in seeds:
        case = build_walker_filter_case(
            seed=seed, duration=duration, dt=dt,
            maximum_range=maximum_range,
            topology=audit.persistent_topology,
            truth_history_by_node=audit.scenario.truth_state_history_by_node,
            topology_type="walker_persistent",
        )
        profiles = (
            (
                "production_exact_replay_three_modal",
                case["observations"], "exact_transport_event_replay",
                case["state_messages"],
            ),
            (
                "propagate_only_three_modal",
                case["observations"], "propagate_only", None,
            ),
            (
                "propagate_only_absolute_only",
                (), "propagate_only", None,
            ),
        )
        for profile, observations, refresh_mode, state_messages in profiles:
            epoch_seconds = []
            started = perf_counter()
            run_network_schmidt_filter(
                timestamps=case["timestamps"],
                initial_state_by_node=case["initial_states"],
                initial_covariance_by_node=case["initial_covariances"],
                topology=case["topology"],
                observation_messages=observations,
                absolute_position_observations=case["absolute_observations"],
                observation_usage="observer_only",
                process_noise_acceleration=1e-8,
                consider_refresh_mode=refresh_mode,
                state_messages_by_receiver=state_messages,
                replay_history_window=max(10.0, 4.0 * dt),
                expected_lineage_by_link=(
                    case["lineages"] if state_messages is not None else None
                ),
                epoch_timing_callback=lambda _, __, elapsed: (
                    epoch_seconds.append(elapsed)
                ),
            )
            total = perf_counter() - started
            latency = 1000.0 * np.asarray(epoch_seconds)
            records.append(TimingAblationRecord(
                seed=seed, profile=profile, epoch_count=latency.size,
                total_seconds=total,
                mean_frequency_hz=float(latency.size / total),
                mean_epoch_latency_ms=float(np.mean(latency)),
                p95_epoch_latency_ms=float(np.percentile(latency, 95)),
                p99_epoch_latency_ms=float(np.percentile(latency, 99)),
                maximum_epoch_latency_ms=float(np.max(latency)),
            ))
    means = {
        profile: float(np.mean([
            item.mean_epoch_latency_ms for item in records
            if item.profile == profile
        ]))
        for profile in (
            "production_exact_replay_three_modal",
            "propagate_only_three_modal",
            "propagate_only_absolute_only",
        )
    }
    return TimingAblationReport(
        walker_definition=(20, 10, 1), duration_seconds=float(duration),
        dt_seconds=float(dt), timing_scope="algorithm_only_without_cann_gui_or_io",
        records=tuple(records), mean_latency_by_profile_ms=means,
        exact_replay_increment_ms=(
            means["production_exact_replay_three_modal"]
            - means["propagate_only_three_modal"]
        ),
        relative_measurement_increment_ms=(
            means["propagate_only_three_modal"]
            - means["propagate_only_absolute_only"]
        ),
    )


def save_external_update_frequency_ablation(report, output):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
