from __future__ import annotations

import numpy as np

from cooperative.communication_channel import CommunicationChannel
from cooperative.consensus_runner import (
    CommunicationStats,
    DistributedConsensusHistory,
    _apply_inter_satellite_updates,
    _run_consensus_iterations,
)
from cooperative.delay_channel import DelayChannel
from cooperative.inter_satellite_observation_adapter import (
    InterSatelliteObservationAdapterResult,
    adapt_inter_satellite_observations,
)
from cooperative.message_buffer import MessageBuffer
from cooperative.satellite_node import SatelliteNode
from cooperative.time_alignment import align_report_time
from cooperative.topology import NetworkTopology
from interfaces.data_objects import InterSatelliteObservation, NodeReport
from orbital_core.absolute_filter import AbsoluteOrbitEKF
from orbital_core.dynamics import make_process_noise

Array = np.ndarray


def run_fleet_filter(
    *,
    timestamps: Array,
    initial_state_by_node: dict[str, Array],
    initial_covariance_by_node: dict[str, Array],
    topology: NetworkTopology,
    inter_satellite_observations: list[InterSatelliteObservation] | None = None,
    process_noise_acceleration_std: float = 1e-4,
    communication_channel: CommunicationChannel | None = None,
    delay_channel: DelayChannel | None = None,
    align_delayed_reports: bool = True,
    consensus_iterations: int = 1,
    ci_objective: str = "trace",
    ci_grid_points: int = 31,
    age_aware: bool = False,
    age_penalty: float = 1e-2,
    inter_satellite_gate_enable: bool = False,
    inter_satellite_gate_threshold: float = np.inf,
    inter_satellite_gate_mode: str = "soft",
    inter_satellite_soft_scale: float = 20.0,
    enable_state_consensus: bool = False,
    inter_satellite_frame_by_modality: dict[str, str] | None = None,
) -> DistributedConsensusHistory:
    """Run the v13 per-satellite predict/update/consensus feedback loop."""

    times = np.asarray(timestamps, dtype=float).reshape(-1)
    if times.size == 0:
        raise ValueError("timestamps cannot be empty.")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("timestamps must be strictly increasing.")
    if consensus_iterations < 1:
        raise ValueError("consensus_iterations must be at least 1.")
    if enable_state_consensus:
        raise ValueError(
            "State consensus is disabled for per-satellite absolute states: "
            "CI cannot fuse estimates of different physical satellites."
        )
    node_ids = topology.node_ids
    if set(initial_state_by_node) != set(node_ids):
        raise ValueError("initial_state_by_node keys must match topology node IDs.")
    if set(initial_covariance_by_node) != set(node_ids):
        raise ValueError("initial_covariance_by_node keys must match topology node IDs.")

    inter_satellite_data: InterSatelliteObservationAdapterResult | None = (
        adapt_inter_satellite_observations(inter_satellite_observations, timestamps=times)
        if inter_satellite_observations is not None
        else None
    )

    states = {
        node_id: np.asarray(initial_state_by_node[node_id], dtype=float).reshape(6).copy()
        for node_id in node_ids
    }
    covariances = {
        node_id: np.asarray(initial_covariance_by_node[node_id], dtype=float).reshape(6, 6).copy()
        for node_id in node_ids
    }
    state_history = {node_id: np.zeros((times.size, 6), dtype=float) for node_id in node_ids}
    covariance_history = {
        node_id: np.zeros((times.size, 6, 6), dtype=float)
        for node_id in node_ids
    }
    weight_history = {node_id: [] for node_id in node_ids}
    iteration_weight_history = {node_id: [] for node_id in node_ids}
    received_history = {node_id: [] for node_id in node_ids}
    nis_history = {node_id: [] for node_id in node_ids}
    gate_history = {node_id: [] for node_id in node_ids}
    buffers = {node_id: MessageBuffer() for node_id in node_ids}
    attempted_count = 0
    received_count = 0
    dropped_count = 0
    delay_sum = 0.0

    for index, timestamp in enumerate(times):
        if index > 0:
            dt = float(times[index] - times[index - 1])
            process_noise = make_process_noise(dt, process_noise_acceleration_std)
            for node_id in node_ids:
                filter_obj = AbsoluteOrbitEKF(process_noise=process_noise)
                states[node_id], covariances[node_id] = filter_obj.predict(
                    states[node_id],
                    covariances[node_id],
                    dt,
                )

        local_reports = {
            node_id: _make_report(
                node_id=node_id,
                timestamp=float(timestamp),
                state=states[node_id],
                covariance=covariances[node_id],
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

        final_reports = epoch_local_reports
        iteration_weights = {
            node_id: [{node_id: 1.0} for _ in range(consensus_iterations)]
            for node_id in node_ids
        }

        for node_id in node_ids:
            final_report = final_reports[node_id]
            states[node_id] = final_report.state_estimate.copy()
            covariances[node_id] = final_report.covariance.copy()
            state_history[node_id][index] = states[node_id]
            covariance_history[node_id][index] = covariances[node_id]
            weight_history[node_id].append(iteration_weights[node_id][-1])
            iteration_weight_history[node_id].append(iteration_weights[node_id])
            received_history[node_id].append(
                [report.node_id for report in epoch_neighbor_reports[node_id] if report.valid_flag]
            )
            nis_history[node_id].append(epoch_nis[node_id])
            gate_history[node_id].append(epoch_gates[node_id])

    return DistributedConsensusHistory(
        timestamps=times.copy(),
        state_history_by_node=state_history,
        covariance_history_by_node=covariance_history,
        node_weight_history_by_node=weight_history,
        iteration_weight_history_by_node=iteration_weight_history,
        received_reports_by_node=received_history,
        inter_satellite_nis_history_by_node=nis_history,
        inter_satellite_gate_history_by_node=gate_history,
        communication_stats=CommunicationStats(
            attempted_report_count=attempted_count,
            received_report_count=received_count,
            dropped_report_count=dropped_count,
            pending_report_count=sum(len(buffer.reports) for buffer in buffers.values()),
            average_delay=delay_sum / received_count if received_count else 0.0,
            packet_loss_rate=dropped_count / attempted_count if attempted_count else 0.0,
        ),
    )


def _make_report(
    *,
    node_id: str,
    timestamp: float,
    state: Array,
    covariance: Array,
) -> NodeReport:
    return SatelliteNode(
        node_id=node_id,
        state=np.asarray(state, dtype=float).reshape(6),
        covariance=np.asarray(covariance, dtype=float).reshape(6, 6),
    ).estimate(timestamp).to_report()
