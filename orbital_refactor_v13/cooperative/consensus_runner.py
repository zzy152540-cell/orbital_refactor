from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from cooperative.age_weight import inflate_covariance_by_age
from cooperative.communication_channel import CommunicationChannel
from cooperative.delay_channel import DelayChannel
from cooperative.inter_satellite_observation_adapter import (
    InterSatelliteObservationAdapterResult,
    adapt_inter_satellite_observations,
)
from cooperative.inter_satellite_range import update_with_inter_satellite_observation_block
from cooperative.message_buffer import MessageBuffer
from cooperative.satellite_node import SatelliteNode
from cooperative.time_alignment import align_report_time
from cooperative.topology import NetworkTopology
from interfaces.data_objects import InterSatelliteObservation, NodeReport
from orbital_core.ci_fusion import ci_fuse_posteriors

Array = np.ndarray


@dataclass(frozen=True)
class CommunicationStats:
    attempted_report_count: int
    received_report_count: int
    dropped_report_count: int
    pending_report_count: int
    average_delay: float
    packet_loss_rate: float


@dataclass(frozen=True)
class DistributedConsensusHistory:
    timestamps: Array
    state_history_by_node: dict[str, Array]
    covariance_history_by_node: dict[str, Array]
    node_weight_history_by_node: dict[str, list[dict[str, float]]]
    iteration_weight_history_by_node: dict[str, list[list[dict[str, float]]]]
    received_reports_by_node: dict[str, list[list[str]]]
    inter_satellite_nis_history_by_node: dict[str, list[dict[str, float]]]
    inter_satellite_gate_history_by_node: dict[str, list[dict[str, bool]]]
    communication_stats: CommunicationStats

    @property
    def node_ids(self) -> list[str]:
        return list(self.state_history_by_node)

    @property
    def range_nis_history_by_node(self) -> dict[str, list[dict[str, float]]]:
        return {
            node_id: [
                {
                    key.split(":", 1)[0]: value
                    for key, value in per_epoch.items()
                    if key.endswith(":RANGE")
                }
                for per_epoch in history
            ]
            for node_id, history in self.inter_satellite_nis_history_by_node.items()
        }


def run_distributed_consensus_history(
    *,
    timestamps: Array,
    state_history_by_node: Mapping[str, Array],
    covariance_history_by_node: Mapping[str, Array],
    topology: NetworkTopology,
    communication_channel: CommunicationChannel | None = None,
    delay_channel: DelayChannel | None = None,
    objective: str = "trace",
    grid_points: int = 31,
    age_aware: bool = False,
    age_penalty: float = 1e-2,
    align_delayed_reports: bool = True,
    range_measurements_by_node: Mapping[str, Mapping[str, Array]] | None = None,
    range_variance: float | None = None,
    inter_satellite_observations: list[InterSatelliteObservation] | None = None,
    inter_satellite_gate_enable: bool = False,
    inter_satellite_gate_threshold: float = np.inf,
    inter_satellite_gate_mode: str = "soft",
    inter_satellite_soft_scale: float = 20.0,
    consensus_iterations: int = 1,
    inter_satellite_frame_by_modality: dict[str, str] | None = None,
) -> DistributedConsensusHistory:
    """Run multi-epoch distributed Consensus-CI over per-satellite histories.

    Inputs and outputs are keyed by satellite node ID. The result deliberately
    has no global state; every satellite keeps its own consensus-updated track.
    """

    times = np.asarray(timestamps, dtype=float).reshape(-1)
    if times.size == 0:
        raise ValueError("timestamps cannot be empty.")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("timestamps must be strictly increasing.")
    if consensus_iterations < 1:
        raise ValueError("consensus_iterations must be at least 1.")
    node_ids = topology.node_ids
    if set(state_history_by_node) != set(node_ids):
        raise ValueError("state_history_by_node keys must match topology node IDs.")
    if set(covariance_history_by_node) != set(node_ids):
        raise ValueError("covariance_history_by_node keys must match topology node IDs.")

    local_states = _validate_state_histories(state_history_by_node, times.size)
    local_covariances = _validate_covariance_histories(covariance_history_by_node, times.size)
    inter_satellite_data: InterSatelliteObservationAdapterResult | None = None
    if inter_satellite_observations is not None:
        if range_measurements_by_node is not None or range_variance is not None:
            raise ValueError(
                "Use either inter_satellite_observations or raw range_measurements_by_node, not both."
            )
        inter_satellite_data = adapt_inter_satellite_observations(
            inter_satellite_observations,
            timestamps=times,
        )
    elif range_measurements_by_node is not None:
        if range_variance is None:
            raise ValueError("range_variance is required when range measurements are provided.")
        inter_satellite_data = _adapt_raw_range_inputs(
            range_measurements_by_node,
            range_variance=float(range_variance),
        )

    output_states = {node_id: np.zeros((times.size, 6), dtype=float) for node_id in node_ids}
    output_covariances = {
        node_id: np.zeros((times.size, 6, 6), dtype=float)
        for node_id in node_ids
    }
    weight_history = {node_id: [] for node_id in node_ids}
    iteration_weight_history = {node_id: [] for node_id in node_ids}
    received_history = {node_id: [] for node_id in node_ids}
    inter_satellite_nis_history = {node_id: [] for node_id in node_ids}
    inter_satellite_gate_history = {node_id: [] for node_id in node_ids}
    buffers = {node_id: MessageBuffer() for node_id in node_ids}
    attempted_count = 0
    received_count = 0
    dropped_count = 0
    delay_sum = 0.0

    for index, timestamp in enumerate(times):
        local_reports = {
            node_id: _make_report(
                node_id=node_id,
                timestamp=float(timestamp),
                state=local_states[node_id][index],
                covariance=local_covariances[node_id][index],
            )
            for node_id in node_ids
        }

        for source_id, report in local_reports.items():
            for target_id in topology.neighbors(source_id):
                attempted_count += 1
                delivered = [report]
                if communication_channel is not None:
                    delivered = communication_channel.transmit(delivered)
                if not delivered:
                    dropped_count += 1
                    continue
                if delay_channel is not None:
                    delivered = delay_channel.transmit(delivered)
                for message in delivered:
                    buffers[target_id].push(message)

        epoch_local_reports: dict[str, NodeReport] = {}
        epoch_neighbor_reports: dict[str, list[NodeReport]] = {}
        epoch_nis: dict[str, dict[str, float]] = {}
        epoch_gates: dict[str, dict[str, bool]] = {}

        for node_id in node_ids:
            available = buffers[node_id].pop_available(float(timestamp))
            received_count += len(available)
            delay_sum += sum(float(report.communication_delay) for report in available)
            reports = [
                align_report_time(report, float(timestamp))
                if align_delayed_reports
                else report
                for report in available
            ]
            local_report = local_reports[node_id]
            inter_satellite_nis: dict[str, float] = {}
            inter_satellite_gates: dict[str, bool] = {}
            if inter_satellite_data is not None:
                (
                    local_report,
                    inter_satellite_nis,
                    inter_satellite_gates,
                ) = _apply_inter_satellite_updates(
                    local_report=local_report,
                    neighbor_reports=reports,
                    node_id=node_id,
                    sample_index=index,
                    observations=inter_satellite_data,
                    gate_enable=inter_satellite_gate_enable,
                    gate_threshold=inter_satellite_gate_threshold,
                    gate_mode=inter_satellite_gate_mode,
                    soft_scale=inter_satellite_soft_scale,
                    frame_by_modality=inter_satellite_frame_by_modality,
                )
            epoch_local_reports[node_id] = local_report
            epoch_neighbor_reports[node_id] = reports
            epoch_nis[node_id] = inter_satellite_nis
            epoch_gates[node_id] = inter_satellite_gates

        final_reports, iteration_weights = _run_consensus_iterations(
            local_reports=epoch_local_reports,
            neighbor_reports_by_node=epoch_neighbor_reports,
            timestamp=float(timestamp),
            iterations=consensus_iterations,
            objective=objective,
            grid_points=grid_points,
            age_aware=age_aware,
            age_penalty=age_penalty,
        )

        for node_id in node_ids:
            final_report = final_reports[node_id]
            output_states[node_id][index] = final_report.state_estimate
            output_covariances[node_id][index] = final_report.covariance
            final_weights = iteration_weights[node_id][-1]
            weight_history[node_id].append(final_weights)
            iteration_weight_history[node_id].append(iteration_weights[node_id])
            inter_satellite_nis_history[node_id].append(epoch_nis[node_id])
            inter_satellite_gate_history[node_id].append(epoch_gates[node_id])
            received_history[node_id].append(
                [report.node_id for report in epoch_neighbor_reports[node_id] if report.valid_flag]
            )

    return DistributedConsensusHistory(
        timestamps=times.copy(),
        state_history_by_node=output_states,
        covariance_history_by_node=output_covariances,
        node_weight_history_by_node=weight_history,
        iteration_weight_history_by_node=iteration_weight_history,
        received_reports_by_node=received_history,
        inter_satellite_nis_history_by_node=inter_satellite_nis_history,
        inter_satellite_gate_history_by_node=inter_satellite_gate_history,
        communication_stats=CommunicationStats(
            attempted_report_count=attempted_count,
            received_report_count=received_count,
            dropped_report_count=dropped_count,
            pending_report_count=sum(len(buffer.reports) for buffer in buffers.values()),
            average_delay=delay_sum / received_count if received_count else 0.0,
            packet_loss_rate=dropped_count / attempted_count if attempted_count else 0.0,
        ),
    )


def _run_consensus_iterations(
    *,
    local_reports: dict[str, NodeReport],
    neighbor_reports_by_node: dict[str, list[NodeReport]],
    timestamp: float,
    iterations: int,
    objective: str,
    grid_points: int,
    age_aware: bool,
    age_penalty: float,
) -> tuple[dict[str, NodeReport], dict[str, list[dict[str, float]]]]:
    current_reports = dict(local_reports)
    allowed_neighbors = {
        node_id: [report.node_id for report in reports if report.valid_flag]
        for node_id, reports in neighbor_reports_by_node.items()
    }
    weights_by_node = {node_id: [] for node_id in local_reports}

    for iteration in range(iterations):
        next_reports: dict[str, NodeReport] = {}
        for node_id, local_report in current_reports.items():
            if iteration == 0:
                reports = [local_report, *neighbor_reports_by_node[node_id]]
            else:
                reports = [
                    local_report,
                    *[
                        current_reports[neighbor_id]
                        for neighbor_id in allowed_neighbors[node_id]
                        if neighbor_id in current_reports
                    ],
                ]
            state, covariance, weights = _fuse_reports(
                reports,
                current_time=float(timestamp),
                objective=objective,
                grid_points=grid_points,
                age_aware=age_aware,
                age_penalty=age_penalty,
            )
            weights_by_node[node_id].append(weights)
            next_reports[node_id] = _copy_report_with_posterior(
                local_report,
                timestamp=timestamp,
                state=state,
                covariance=covariance,
            )
        current_reports = next_reports
    return current_reports, weights_by_node


def _copy_report_with_posterior(
    report: NodeReport,
    *,
    timestamp: float,
    state: Array,
    covariance: Array,
) -> NodeReport:
    return NodeReport(
        node_id=report.node_id,
        target_id=report.target_id,
        timestamp=float(timestamp),
        state_estimate=np.asarray(state, dtype=float).reshape(6).copy(),
        covariance=np.asarray(covariance, dtype=float).reshape(6, 6).copy(),
        quality_score=report.quality_score,
        health_status=report.health_status,
        communication_delay=report.communication_delay,
        valid_flag=report.valid_flag,
        source_timestamp=report.source_timestamp,
        arrival_timestamp=report.arrival_timestamp,
    )


def _adapt_raw_range_inputs(
    range_measurements_by_node: Mapping[str, Mapping[str, Array]],
    *,
    range_variance: float,
) -> InterSatelliteObservationAdapterResult:
    return InterSatelliteObservationAdapterResult(
        measurements_by_node={
            source: {
                target: {"RANGE": np.asarray(values, dtype=float)}
                for target, values in targets.items()
            }
            for source, targets in range_measurements_by_node.items()
        },
        covariance_by_modality={"RANGE": np.array([[float(range_variance)]], dtype=float)},
    )


def _apply_inter_satellite_updates(
    *,
    local_report: NodeReport,
    neighbor_reports: list[NodeReport],
    node_id: str,
    sample_index: int,
    observations: InterSatelliteObservationAdapterResult,
    gate_enable: bool,
    gate_threshold: float,
    gate_mode: str,
    soft_scale: float,
    frame_by_modality: dict[str, str] | None,
) -> tuple[NodeReport, dict[str, float], dict[str, bool]]:
    state = local_report.state_estimate.copy()
    covariance = local_report.covariance.copy()
    nis_by_observation: dict[str, float] = {}
    gate_by_observation: dict[str, bool] = {}
    measurements_for_node = observations.measurements_by_node.get(node_id, {})
    for report in neighbor_reports:
        if not report.valid_flag or report.node_id not in measurements_for_node:
            continue
        modality_histories = measurements_for_node[report.node_id]
        measurement_block: dict[str, float] = {}
        for modality in sorted(modality_histories):
            values = np.asarray(modality_histories[modality], dtype=float)
            if sample_index >= len(values):
                raise ValueError(
                    f"{modality} history for {node_id}->{report.node_id} is too short."
                )
            measurement = np.asarray(values[sample_index], dtype=float)
            if not np.all(np.isfinite(measurement)):
                continue
            measurement_block[modality] = (
                float(measurement)
                if measurement.shape == ()
                else measurement.copy()
            )
        if not measurement_block:
            continue
        update = update_with_inter_satellite_observation_block(
            state=state,
            covariance=covariance,
            neighbor_report=report,
            measurements_by_modality=measurement_block,
            covariance_by_modality=observations.covariance_by_modality,
            gate_enable=gate_enable,
            gate_threshold=gate_threshold,
            gate_mode=gate_mode,
            soft_scale=soft_scale,
            frame_by_modality=frame_by_modality,
        )
        if not update.skipped:
            state = update.state
            covariance = update.covariance
        nis_by_observation[f"{report.node_id}:BLOCK"] = update.nis
        gate_by_observation[f"{report.node_id}:BLOCK"] = update.gated
        for modality in update.modalities:
            nis_by_observation[f"{report.node_id}:{modality}"] = update.nis
            gate_by_observation[f"{report.node_id}:{modality}"] = update.gated
    return (
        NodeReport(
            node_id=local_report.node_id,
            target_id=local_report.target_id,
            timestamp=local_report.timestamp,
            state_estimate=state,
            covariance=covariance,
            quality_score=local_report.quality_score,
            health_status=local_report.health_status,
            communication_delay=local_report.communication_delay,
            valid_flag=local_report.valid_flag,
            source_timestamp=local_report.source_timestamp,
            arrival_timestamp=local_report.arrival_timestamp,
        ),
        nis_by_observation,
        gate_by_observation,
    )


def _make_report(
    *,
    node_id: str,
    timestamp: float,
    state: Array,
    covariance: Array,
) -> NodeReport:
    covariance = np.asarray(covariance, dtype=float).reshape(6, 6)
    return SatelliteNode(
        node_id=node_id,
        state=np.asarray(state, dtype=float).reshape(6),
        covariance=covariance,
    ).estimate(timestamp).to_report()


def _fuse_reports(
    reports: list[NodeReport],
    *,
    current_time: float,
    objective: str,
    grid_points: int,
    age_aware: bool,
    age_penalty: float,
) -> tuple[Array, Array, dict[str, float]]:
    valid = [report for report in reports if report.valid_flag]
    if not valid:
        raise ValueError("At least the local node report must be valid.")
    inputs = []
    for report in valid:
        covariance = report.covariance
        if age_aware:
            source_time = (
                report.source_timestamp
                if report.source_timestamp is not None
                else report.timestamp
            )
            covariance = inflate_covariance_by_age(
                covariance,
                max(float(current_time - source_time), 0.0),
                penalty=age_penalty,
            )
        inputs.append((report.node_id, report.state_estimate, covariance))
    fusion = ci_fuse_posteriors(inputs, objective=objective, grid_points=grid_points)
    return fusion.state, fusion.covariance, dict(fusion.weights)


def _validate_state_histories(
    histories: Mapping[str, Array],
    sample_count: int,
) -> dict[str, Array]:
    result = {}
    for node_id, history in histories.items():
        values = np.asarray(history, dtype=float)
        if values.shape != (sample_count, 6):
            raise ValueError(f"State history for {node_id} must have shape (N, 6).")
        result[node_id] = values
    return result


def _validate_covariance_histories(
    histories: Mapping[str, Array],
    sample_count: int,
) -> dict[str, Array]:
    result = {}
    for node_id, history in histories.items():
        values = np.asarray(history, dtype=float)
        if values.shape != (sample_count, 6, 6):
            raise ValueError(f"Covariance history for {node_id} must have shape (N, 6, 6).")
        result[node_id] = values
    return result
