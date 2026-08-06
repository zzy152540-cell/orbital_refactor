from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from cooperative.network_schmidt_runner import run_network_schmidt_filter
from experiments.network_filter_metrics import network_history_metrics
from experiments.summary_statistics import interval_coverage, mean_metric_dict
from experiments.walker_filter_setup import (
    WALKER_FILTER_MODALITIES,
    build_walker_filter_case,
)
from experiments.v14_walker_geometry_audit import run_v14_walker_geometry_audit
from orbital_core.metrics import compute_nees_history, compute_rmse

NEES_95_DOF6 = (1.2373442458, 14.4493753354)


@dataclass(frozen=True)
class WalkerTemporalDiagnostic:
    start_timestamp: float
    end_timestamp: float
    sample_count: int
    position_rmse: float
    velocity_rmse: float
    mean_nees: float
    nees_95_coverage: float
    mean_position_standard_deviation: float
    mean_velocity_standard_deviation: float


@dataclass(frozen=True)
class WalkerFilterSmokeResult:
    walker_definition: tuple[int, int, int]
    node_count: int
    run_count: int
    persistent_undirected_edge_count: int
    minimum_node_degree: int
    maximum_node_degree: int
    mean_position_rmse: float
    mean_velocity_rmse: float
    mean_nees: float
    mean_nees_95_coverage: float
    mean_nis_by_modality: dict[str, float]
    mean_nis_95_coverage_by_modality: dict[str, float]
    observation_count_by_modality_per_run: dict[str, int]
    message_acceptance_rate: float
    message_rejection_count: int
    psd_failure_count: int
    minimum_joint_eigenvalue: float
    mean_run_seconds: float
    replay_count: int
    maximum_replay_seconds: float
    maximum_remote_event_count: int
    maximum_observation_count: int
    maximum_checkpoint_count: int
    temporal_diagnostics: tuple[WalkerTemporalDiagnostic, ...]


def run_v14_walker_filter_smoke(
    *, seeds: int = 1, duration: float = 60.0, dt: float = 2.0,
    maximum_range: float = 6000e3,
    diagnostic_boundaries: tuple[float, ...] | None = None,
) -> WalkerFilterSmokeResult:
    """Run current exact-replay filtering on the Walker 20/10/1 static graph."""

    if seeds < 1:
        raise ValueError("seeds must be at least one.")
    boundaries = _diagnostic_boundaries(duration, diagnostic_boundaries)
    audit = run_v14_walker_geometry_audit(
        total_satellites=20, plane_count=10, phasing=1,
        duration=1800.0, dt=30.0, maximum_range=maximum_range,
    )
    if audit.persistent_component_sizes != (20,):
        raise ValueError(
            "Walker 20/10/1 has no connected persistent topology at this range."
        )
    values = []
    temporal_values = [[] for _ in range(len(boundaries) - 1)]
    counts_per_run = None
    for seed in range(seeds):
        case = build_walker_filter_case(
            seed=seed, duration=duration, dt=dt, maximum_range=maximum_range,
            topology=audit.persistent_topology,
            truth_history_by_node=audit.scenario.truth_state_history_by_node,
            topology_type="walker_persistent",
        )
        counts = {
            modality: sum(
                observation.modality == modality
                for observation in case["observations"]
            )
            for modality in WALKER_FILTER_MODALITIES
        }
        counts_per_run = counts if counts_per_run is None else counts_per_run
        if counts != counts_per_run:
            raise RuntimeError("Walker observation counts changed between seeds.")
        started = perf_counter()
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
        values.append(network_history_metrics(
            history, case["truth"], len(case["transmitted_messages"]),
            perf_counter() - started,
        ))
        for segment_index, (start, end) in enumerate(
            zip(boundaries, boundaries[1:])
        ):
            temporal_values[segment_index].append(_temporal_metrics(
                history, case["truth"], start=start, end=end,
                include_end=segment_index == len(boundaries) - 2,
            ))
    accepted = sum(value[7] for value in values)
    rejected = sum(value[9] for value in values)
    return WalkerFilterSmokeResult(
        walker_definition=(20, 10, 1), node_count=20, run_count=seeds,
        persistent_undirected_edge_count=audit.persistent_undirected_edge_count,
        minimum_node_degree=audit.minimum_persistent_node_degree,
        maximum_node_degree=audit.maximum_persistent_node_degree,
        mean_position_rmse=float(np.mean([value[0] for value in values])),
        mean_velocity_rmse=float(np.mean([value[1] for value in values])),
        mean_nees=float(np.mean([value[2] for value in values])),
        mean_nees_95_coverage=float(np.mean([value[3] for value in values])),
        mean_nis_by_modality=mean_metric_dict([value[14] for value in values]),
        mean_nis_95_coverage_by_modality=mean_metric_dict(
            [value[15] for value in values]
        ),
        observation_count_by_modality_per_run=counts_per_run or {},
        message_acceptance_rate=(
            accepted / (accepted + rejected) if accepted + rejected else 0.0
        ),
        message_rejection_count=rejected,
        psd_failure_count=sum(value[10] for value in values),
        minimum_joint_eigenvalue=min(value[6] for value in values),
        mean_run_seconds=float(np.mean([value[12] for value in values])),
        replay_count=sum(value[13]["replay_count"] for value in values),
        maximum_replay_seconds=max(
            value[13]["maximum_replay_seconds"] for value in values
        ),
        maximum_remote_event_count=max(
            value[13]["maximum_remote_event_count"] for value in values
        ),
        maximum_observation_count=max(
            value[13]["maximum_observation_count"] for value in values
        ),
        maximum_checkpoint_count=max(
            value[13]["maximum_checkpoint_count"] for value in values
        ),
        temporal_diagnostics=tuple(
            WalkerTemporalDiagnostic(
                start_timestamp=boundaries[index],
                end_timestamp=boundaries[index + 1],
                sample_count=sum(value[0] for value in segment_values),
                position_rmse=_pooled_rmse(segment_values, 1),
                velocity_rmse=_pooled_rmse(segment_values, 2),
                mean_nees=float(np.mean([
                    item for value in segment_values for item in value[3]
                ])),
                nees_95_coverage=interval_coverage([
                    item for value in segment_values for item in value[3]
                ], NEES_95_DOF6),
                mean_position_standard_deviation=float(np.mean([
                    item for value in segment_values for item in value[4]
                ])),
                mean_velocity_standard_deviation=float(np.mean([
                    item for value in segment_values for item in value[5]
                ])),
            )
            for index, segment_values in enumerate(temporal_values)
        ),
    )


def _diagnostic_boundaries(duration, requested):
    if requested is None:
        requested = (
            (0.0, 120.0, 600.0, 1200.0, 1800.0)
            if np.isclose(duration, 1800.0) else (0.0, float(duration))
        )
    boundaries = tuple(float(value) for value in requested)
    if (
        len(boundaries) < 2 or boundaries[0] != 0.0
        or not np.isclose(boundaries[-1], duration)
        or not np.all(np.diff(boundaries) > 0.0)
    ):
        raise ValueError(
            "diagnostic_boundaries must increase from zero to duration."
        )
    return boundaries


def _temporal_metrics(history, truth, *, start, end, include_end):
    mask = (history.timestamps >= start) & (
        history.timestamps <= end if include_end else history.timestamps < end
    )
    position_errors = []
    velocity_errors = []
    nees = []
    position_sigmas = []
    velocity_sigmas = []
    for node in history.node_ids:
        error = history.active_state_history_by_node[node][mask] - truth[node][mask]
        covariance = history.active_covariance_history_by_node[node][mask]
        position_errors.append(error[:, :3])
        velocity_errors.append(error[:, 3:])
        node_nees = compute_nees_history(
            history.active_state_history_by_node[node], truth[node],
            history.active_covariance_history_by_node[node],
        )[mask]
        nees.extend(node_nees)
        position_sigmas.extend(np.sqrt(
            np.trace(covariance[:, :3, :3], axis1=1, axis2=2) / 3.0
        ))
        velocity_sigmas.extend(np.sqrt(
            np.trace(covariance[:, 3:, 3:], axis1=1, axis2=2) / 3.0
        ))
    position = np.vstack(position_errors)
    velocity = np.vstack(velocity_errors)
    return (
        position.shape[0], position, velocity, tuple(nees),
        tuple(position_sigmas), tuple(velocity_sigmas),
    )


def _pooled_rmse(segment_values, index):
    return compute_rmse(np.vstack([value[index] for value in segment_values]))
