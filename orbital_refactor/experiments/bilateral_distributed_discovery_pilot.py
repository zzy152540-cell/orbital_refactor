"""Paired pilot for logically distributed discovery, IOD and target consensus.

The two arms consume the same immutable sensor frontend but own independent
trackers and identity registries. The control arm stops after local filtering;
the cooperative arm additionally exchanges target reports, applies Consensus-CI
and feeds the fused posterior back into each node's local filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from brain_inspired.incremental_measurement_preprocessor import (
    IncrementalMultimodalCANNConfig,
    IncrementalMultimodalCANNPreprocessor,
)
from cooperative.topology import NetworkTopology
from orbital_core.coordinates import build_rtn_quaternion
from tracking import (
    AutonomousIODManager,
    DistributedTargetConsensus,
    DistributedTargetNodeNetwork,
    IODBufferConfig,
    MultiTargetAssociationPipeline,
    MultiTargetIODManager,
    NeighborTrackIdentityConsensus,
    OnlineMultiTargetTracker,
    unlabeled_iod_observations_from_messages,
)

from experiments.bilateral_multitarget_frontend import (
    BilateralFrontendConfig,
    build_bilateral_raw_frontend,
)
from experiments.multitarget_identity_evaluation import IdentityAwareTrackEvaluator


@dataclass(frozen=True)
class BilateralDistributedDiscoveryPilot:
    duration_seconds: float
    seed: int
    discovery_exchange_hops: int
    identity_exchange_hops: int
    state_exchange_hops: int
    observer_count: int
    target_count: int
    cann_enabled: bool
    shared_frontend: bool
    independent_arm_state: bool
    control_local_birth_count: int
    control_nodes_with_births: int
    control_estimate_count: int
    control_cooperative_identity_count: int
    control_identity_switch_count: int
    cooperative_local_birth_count: int
    cooperative_nodes_with_births: int
    cooperative_estimate_count: int
    cooperative_multi_observer_estimate_count: int
    cooperative_identity_count: int
    cooperative_identity_switch_count: int
    matched_sample_count: int
    control_position_mean_error_m: float
    cooperative_position_mean_error_m: float
    matched_position_improvement_percent: float
    control_position_rmse_m: float
    cooperative_position_rmse_m: float
    control_velocity_mean_error_mps: float
    cooperative_velocity_mean_error_mps: float
    matched_velocity_improvement_percent: float
    control_velocity_rmse_mps: float
    cooperative_velocity_rmse_mps: float
    matched_acceleration_sample_count: int
    control_acceleration_mean_error_mps2: float
    cooperative_acceleration_mean_error_mps2: float
    matched_acceleration_improvement_percent: float


def run_bilateral_distributed_discovery_pilot(
    *,
    duration_seconds: float = 20.0,
    seed: int = 0,
    sample_interval_seconds: float = 1.0,
    identity_exchange_hops: int = 4,
    discovery_exchange_hops: int = 4,
    state_exchange_hops: int = 1,
    enable_cann: bool = True,
) -> BilateralDistributedDiscoveryPilot:
    """Compare independent distributed control and cooperative CI arms."""

    config = BilateralFrontendConfig(
        duration_seconds=float(duration_seconds),
        sample_interval_seconds=float(sample_interval_seconds),
        random_seed=int(seed),
    )
    frontend = build_bilateral_raw_frontend(config)
    if int(discovery_exchange_hops) < 0:
        raise ValueError("discovery_exchange_hops cannot be negative.")

    # Simulator-only acceleration: deterministic byte-identical IOD inputs may
    # reuse a solved short arc. The two filtering/identity states remain private.
    iod_initialization_cache = {}
    control = _build_arm(
        frontend,
        identity_exchange_hops=identity_exchange_hops,
        state_exchange_hops=state_exchange_hops,
        enable_state_consensus=False,
        enable_cann=enable_cann,
        iod_initialization_cache=iod_initialization_cache,
    )
    cooperative = _build_arm(
        frontend,
        identity_exchange_hops=identity_exchange_hops,
        state_exchange_hops=state_exchange_hops,
        enable_state_consensus=True,
        enable_cann=enable_cann,
        iod_initialization_cache=iod_initialization_cache,
    )
    observer_ids = frontend.truth.manifest.friendly_observer_ids
    truth_ids = frontend.truth.manifest.hostile_target_ids
    topology = frontend.truth.manifest.friendly_topology
    control_evaluators = {
        node_id: IdentityAwareTrackEvaluator(truth_ids) for node_id in observer_ids
    }
    cooperative_evaluators = {
        node_id: IdentityAwareTrackEvaluator(truth_ids) for node_id in observer_ids
    }
    control_samples = {}
    cooperative_samples = {}
    control_states = {}
    cooperative_states = {}
    control_births = cooperative_births = 0
    control_nodes_with_births = set()
    cooperative_nodes_with_births = set()
    control_estimate_count = cooperative_estimate_count = 0
    cooperative_multi_observer_count = 0

    for epoch_index, timestamp_value in enumerate(frontend.truth.timestamps):
        timestamp = float(timestamp_value)
        observer_states = {
            node_id: frontend.observer_state_by_epoch[(node_id, timestamp)]
            for node_id in observer_ids
        }
        attitudes = {
            node_id: build_rtn_quaternion(state)
            for node_id, state in observer_states.items()
        }
        messages_by_node = {node_id: [] for node_id in observer_ids}
        for message in frontend.messages_by_timestamp[timestamp]:
            messages_by_node[message.observer_id].append(message)
        observations_by_node = {
            node_id: unlabeled_iod_observations_from_messages(
                tuple(messages),
                observer_state_by_epoch=frontend.observer_state_by_epoch,
            )
            for node_id, messages in messages_by_node.items()
        }
        discovery_by_node = {
            node_id: tuple(
                observation
                for source_id in _nodes_within_hops(
                    topology, node_id, int(discovery_exchange_hops)
                )
                for observation in observations_by_node[source_id]
            )
            for node_id in observer_ids
        }
        truth_states = {
            truth_id: history[epoch_index]
            for truth_id, history in (
                frontend.truth.hostile_truth_state_history_by_target.items()
            )
        }

        control_update = control["network"].step(
            timestamp=timestamp,
            observations_by_node=observations_by_node,
            discovery_observations_by_node=discovery_by_node,
            observer_states_eci=observer_states,
            q_eci2pri_by_observer=attitudes,
            apply_consensus_feedback=False,
        )
        cooperative_update = cooperative["network"].step(
            timestamp=timestamp,
            observations_by_node=observations_by_node,
            discovery_observations_by_node=discovery_by_node,
            observer_states_eci=observer_states,
            q_eci2pri_by_observer=attitudes,
            apply_consensus_feedback=True,
        )

        for node_id in observer_ids:
            control_tracking = control_update.tracking_by_node[node_id]
            cooperative_tracking = cooperative_update.tracking_by_node[node_id]
            control_node_births = len(control_tracking.initialized_states_by_target)
            cooperative_node_births = len(
                cooperative_tracking.initialized_states_by_target
            )
            control_births += control_node_births
            cooperative_births += cooperative_node_births
            if control_node_births:
                control_nodes_with_births.add(node_id)
            if cooperative_node_births:
                cooperative_nodes_with_births.add(node_id)

            control_estimates = control_update.consensus.estimates_by_node[node_id]
            cooperative_estimates = (
                cooperative_update.consensus.estimates_by_node[node_id]
            )
            control_estimate_count += len(control_estimates)
            cooperative_estimate_count += len(cooperative_estimates)
            cooperative_multi_observer_count += sum(
                len(estimate.contributing_observer_ids) > 1
                for estimate in cooperative_estimates.values()
            )
            _score_epoch(
                control_evaluators[node_id], control_samples, node_id,
                timestamp, control_estimates, truth_states, control_states,
            )
            _score_epoch(
                cooperative_evaluators[node_id], cooperative_samples, node_id,
                timestamp, cooperative_estimates, truth_states,
                cooperative_states,
            )

    common = tuple(sorted(set(control_samples) & set(cooperative_samples)))
    control_position = np.asarray([
        control_samples[key].position_error_m for key in common
    ])
    cooperative_position = np.asarray([
        cooperative_samples[key].position_error_m for key in common
    ])
    control_velocity = np.asarray([
        control_samples[key].velocity_error_mps for key in common
    ])
    cooperative_velocity = np.asarray([
        cooperative_samples[key].velocity_error_mps for key in common
    ])
    control_acceleration, cooperative_acceleration = _matched_acceleration_errors(
        control_states, cooperative_states, common
    )
    control_ids = _cooperative_ids(control["identity"])
    cooperative_ids = _cooperative_ids(cooperative["identity"])
    return BilateralDistributedDiscoveryPilot(
        duration_seconds=float(duration_seconds),
        seed=int(seed),
        discovery_exchange_hops=int(discovery_exchange_hops),
        identity_exchange_hops=int(identity_exchange_hops),
        state_exchange_hops=int(state_exchange_hops),
        observer_count=len(observer_ids),
        target_count=len(truth_ids),
        cann_enabled=bool(enable_cann),
        shared_frontend=True,
        independent_arm_state=True,
        control_local_birth_count=control_births,
        control_nodes_with_births=len(control_nodes_with_births),
        control_estimate_count=control_estimate_count,
        control_cooperative_identity_count=len(control_ids),
        control_identity_switch_count=sum(
            item.identity_switch_count for item in control_evaluators.values()
        ),
        cooperative_local_birth_count=cooperative_births,
        cooperative_nodes_with_births=len(cooperative_nodes_with_births),
        cooperative_estimate_count=cooperative_estimate_count,
        cooperative_multi_observer_estimate_count=cooperative_multi_observer_count,
        cooperative_identity_count=len(cooperative_ids),
        cooperative_identity_switch_count=sum(
            item.identity_switch_count for item in cooperative_evaluators.values()
        ),
        matched_sample_count=len(common),
        control_position_mean_error_m=_mean_or_nan(control_position),
        cooperative_position_mean_error_m=_mean_or_nan(cooperative_position),
        matched_position_improvement_percent=_improvement_percent(
            control_position, cooperative_position
        ),
        control_position_rmse_m=_rmse(control_position),
        cooperative_position_rmse_m=_rmse(cooperative_position),
        control_velocity_mean_error_mps=_mean_or_nan(control_velocity),
        cooperative_velocity_mean_error_mps=_mean_or_nan(cooperative_velocity),
        matched_velocity_improvement_percent=_improvement_percent(
            control_velocity, cooperative_velocity
        ),
        control_velocity_rmse_mps=_rmse(control_velocity),
        cooperative_velocity_rmse_mps=_rmse(cooperative_velocity),
        matched_acceleration_sample_count=len(control_acceleration),
        control_acceleration_mean_error_mps2=_mean_or_nan(control_acceleration),
        cooperative_acceleration_mean_error_mps2=(
            _mean_or_nan(cooperative_acceleration)
        ),
        matched_acceleration_improvement_percent=_improvement_percent(
            control_acceleration, cooperative_acceleration
        ),
    )


def _build_arm(
    frontend,
    *,
    identity_exchange_hops,
    state_exchange_hops,
    enable_state_consensus,
    enable_cann,
    iod_initialization_cache,
):
    observer_ids = frontend.truth.manifest.friendly_observer_ids
    topology = frontend.truth.manifest.friendly_topology
    identity = NeighborTrackIdentityConsensus(
        topology=topology,
        position_gate_m=100_000.0,
        velocity_gate_mps=500.0,
        process_noise_acceleration=frontend.config.process_noise_acceleration,
        minimum_match_margin=0.05,
        exchange_hops=int(identity_exchange_hops),
    )
    consensus_topology = (
        topology
        if enable_state_consensus
        else NetworkTopology({node_id: () for node_id in observer_ids})
    )
    network = DistributedTargetNodeNetwork(
        trackers_by_node={
            node_id: _local_tracker(
                frontend,
                enable_cann=enable_cann,
                iod_initialization_cache=iod_initialization_cache,
            )
            for node_id in observer_ids
        },
        consensus=DistributedTargetConsensus(
            scene_id=frontend.truth.manifest.scene_id,
            topology=consensus_topology,
            process_noise_acceleration=frontend.config.process_noise_acceleration,
            exchange_hops=int(state_exchange_hops),
        ),
        identity_resolver=identity,
    )
    return {"network": network, "identity": identity}


def _local_tracker(frontend, *, enable_cann, iod_initialization_cache):
    config = frontend.config
    return OnlineMultiTargetTracker(
        scene_id=frontend.truth.manifest.scene_id,
        process_noise_acceleration=config.process_noise_acceleration,
        association_pipeline=MultiTargetAssociationPipeline(
            gate_squared_mahalanobis=config.association_gate_squared_mahalanobis,
            minimum_cost_margin=config.association_minimum_cost_margin,
            prediction_position_sigma_floor_m=(
                config.association_position_sigma_floor_m
            ),
            prediction_velocity_sigma_floor_mps=(
                config.association_velocity_sigma_floor_mps
            ),
            autonomous_iod_manager=AutonomousIODManager(
                iod_manager=MultiTargetIODManager(
                    buffer_config=IODBufferConfig(
                        minimum_epochs=3,
                        minimum_observers=2,
                        minimum_time_span_seconds=(
                            2.0 * config.sample_interval_seconds
                        ),
                        maximum_time_span_seconds=60.0,
                    ),
                    initialization_cache=iod_initialization_cache,
                )
            ),
        ),
        max_coast_epochs=config.tracking_max_coast_epochs,
        max_lost_epochs=config.tracking_max_lost_epochs,
        maximum_track_count=config.maximum_track_count,
        preserve_angular_radial_nullspace=config.preserve_angular_radial_nullspace,
        enable_local_unified_multimodal_update=True,
        measurement_preprocessor_factory=(
            (
                lambda _observer, _target: IncrementalMultimodalCANNPreprocessor(
                    IncrementalMultimodalCANNConfig(profile="realtime")
                )
            )
            if enable_cann else None
        ),
    )


def _score_epoch(
    evaluator, output, node_id, timestamp, estimates, truths, state_output,
):
    state_by_track_id = {
        estimate.track_id: estimate.state_eci for estimate in estimates.values()
    }
    epoch = evaluator.update(
        timestamp=timestamp,
        state_by_track_id=state_by_track_id,
        truth_state_by_id=truths,
    )
    for truth_id, sample in epoch.samples_by_truth.items():
        key = (node_id, truth_id, timestamp)
        output[key] = sample
        state_output[key] = (
            state_by_track_id[sample.track_id].copy(),
            np.asarray(truths[truth_id], dtype=float).copy(),
        )


def _matched_acceleration_errors(control, cooperative, common):
    control_errors = []
    cooperative_errors = []
    by_series = {}
    for node_id, truth_id, timestamp in common:
        by_series.setdefault((node_id, truth_id), []).append(timestamp)
    for (node_id, truth_id), timestamps in by_series.items():
        times = sorted(timestamps)
        for previous, current in zip(times, times[1:]):
            dt = current - previous
            if dt <= 0.0:
                continue
            previous_key = (node_id, truth_id, previous)
            current_key = (node_id, truth_id, current)
            control_estimate_now, truth_now = control[current_key]
            control_estimate_previous, truth_previous = control[previous_key]
            cooperative_estimate_now, _ = cooperative[current_key]
            cooperative_estimate_previous, _ = cooperative[previous_key]
            truth_acceleration = (truth_now[3:] - truth_previous[3:]) / dt
            control_acceleration = (
                control_estimate_now[3:] - control_estimate_previous[3:]
            ) / dt
            cooperative_acceleration = (
                cooperative_estimate_now[3:] - cooperative_estimate_previous[3:]
            ) / dt
            control_errors.append(float(np.linalg.norm(
                control_acceleration - truth_acceleration
            )))
            cooperative_errors.append(float(np.linalg.norm(
                cooperative_acceleration - truth_acceleration
            )))
    return np.asarray(control_errors), np.asarray(cooperative_errors)


def _cooperative_ids(identity):
    return {
        target_id
        for values in identity.cooperative_target_ids_by_node().values()
        for target_id in values
    }


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else float("nan")


def _mean_or_nan(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def _improvement_percent(control: np.ndarray, candidate: np.ndarray) -> float:
    control_mean = _mean_or_nan(control)
    candidate_mean = _mean_or_nan(candidate)
    return (
        100.0 * (control_mean - candidate_mean) / control_mean
        if np.isfinite(control_mean) and control_mean > 0.0
        else float("nan")
    )


def _nodes_within_hops(topology, source_id: str, hops: int) -> tuple[str, ...]:
    visited = {str(source_id)}
    frontier = set(visited)
    for _ in range(int(hops)):
        frontier = {
            neighbor
            for node_id in frontier
            for neighbor in topology.neighbors(node_id)
            if neighbor not in visited
        }
        if not frontier:
            break
        visited.update(frontier)
    return tuple(sorted(visited))
