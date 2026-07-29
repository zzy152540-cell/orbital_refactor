from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interfaces.data_objects import NodeReport
from orbital_core.ci_fusion import ci_fuse_posteriors
from orbital_core.quality import quality_score_from_covariance
from cooperative.communication_channel import CommunicationChannel
from cooperative.delay_channel import DelayChannel
from cooperative.message_buffer import MessageBuffer
from cooperative.time_alignment import align_report_time
from cooperative.age_weight import inflate_covariance_by_age

Array = np.ndarray


@dataclass(frozen=True)
class CooperativeFusionHistory:
    timestamps: Array
    state_history_eci: Array
    covariance_history: Array
    node_weight_history: list[dict[str, float]]
    active_node_history: list[list[str]]
    received_node_history: list[list[str]]


def relative_history_to_absolute(
    relative_state_history_eci: Array,
    observer_state_history_eci: Array,
) -> Array:
    relative = np.asarray(relative_state_history_eci, dtype=float)
    observer = np.asarray(observer_state_history_eci, dtype=float)
    if relative.shape != observer.shape or relative.ndim != 2 or relative.shape[1] != 6:
        raise ValueError("Relative and observer histories must both have shape (N, 6).")
    return relative + observer


def fuse_node_reports(
    reports: list[NodeReport],
    *,
    objective: str = "trace",
    grid_points: int = 31,
    communication_channel: CommunicationChannel | None = None,
    delay_channel: DelayChannel | None = None,
    age_aware: bool = False,
    age_penalty: float = 1e-2,
) -> tuple[Array, Array, dict[str, float]]:
    valid = [report for report in reports if report.valid_flag]
    if communication_channel is not None:
        valid = communication_channel.transmit(valid)
    if delay_channel is not None:
        # In v11 the delay information is attached to reports. The default
        # fusion still uses the latest received report; time alignment can be
        # added without changing this interface.
        valid = delay_channel.transmit(valid)
    if not valid:
        raise ValueError("No valid node reports are available for fusion.")
    fusion_inputs = []
    for report in valid:
        covariance = report.covariance
        if age_aware:
            source_time = (
                report.source_timestamp
                if report.source_timestamp is not None
                else report.timestamp
            )
            age = max(float(report.timestamp - source_time), 0.0)
            covariance = inflate_covariance_by_age(
                covariance,
                age,
                penalty=age_penalty,
            )
        fusion_inputs.append(
            (report.node_id, report.state_estimate, covariance)
        )

    result = ci_fuse_posteriors(
        fusion_inputs,
        objective=objective,
        grid_points=grid_points,
    )
    return result.state, result.covariance, result.weights


def fuse_local_histories(
    *,
    timestamps: Array,
    relative_state_history_by_node: dict[str, Array],
    covariance_history_by_node: dict[str, Array],
    observer_state_history_by_node: dict[str, Array],
    target_id: str,
    validity_history_by_node: dict[str, Array] | None = None,
    objective: str = "trace",
    grid_points: int = 31,
    communication_channel: CommunicationChannel | None = None,
    delay_channel: DelayChannel | None = None,
    age_aware: bool = False,
    age_penalty: float = 1e-2,
) -> CooperativeFusionHistory:
    """Convert local relative histories to absolute ECI and fuse them at each epoch."""
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
    node_ids = list(relative_state_history_by_node)
    if not node_ids:
        raise ValueError("At least one node history is required.")
    if len(node_ids) > 3:
        raise ValueError("The current simultaneous CI core supports at most three nodes.")

    absolute = {
        node_id: relative_history_to_absolute(
            relative_state_history_by_node[node_id], observer_state_history_by_node[node_id]
        )
        for node_id in node_ids
    }
    fused_states = np.zeros((len(timestamps), 6))
    fused_covariances = np.zeros((len(timestamps), 6, 6))
    weight_history: list[dict[str, float]] = []
    active_history: list[list[str]] = []
    received_history: list[list[str]] = []

    # Each fusion run owns an independent communication buffer.
    # Avoid function-level persistent state leaking between experiments.
    message_buffer = MessageBuffer()

    for index, timestamp in enumerate(timestamps):
        reports: list[NodeReport] = []
        for node_id in node_ids:
            covariance = np.asarray(covariance_history_by_node[node_id][index], dtype=float)
            valid = True if validity_history_by_node is None else bool(validity_history_by_node[node_id][index])
            reports.append(NodeReport(
                node_id=node_id,
                target_id=target_id,
                timestamp=float(timestamp),
                state_estimate=absolute[node_id][index].copy(),
                covariance=covariance.copy(),
                quality_score=quality_score_from_covariance(covariance),
                health_status="NORMAL" if valid else "UNAVAILABLE",
                communication_delay=0.0,
                valid_flag=valid,
            ))
        # Apply communication delay as an asynchronous message arrival process.
        if delay_channel is not None:
            delayed_reports = delay_channel.transmit(reports)
        else:
            delayed_reports = reports
            for report in delayed_reports:
                report.source_timestamp = report.timestamp
                report.arrival_timestamp = report.timestamp

        for report in delayed_reports:
            message_buffer.push(report)

        received_reports = message_buffer.pop_available(float(timestamp))

        # Align delayed states to the current fusion epoch before CI.
        reports = [
            align_report_time(report, float(timestamp))
            for report in received_reports
        ]

        active_nodes = [report.node_id for report in reports if report.valid_flag]
        if active_nodes:
            state, covariance, weights = fuse_node_reports(
                reports, objective=objective, grid_points=grid_points,
                communication_channel=communication_channel,
                delay_channel=None,
                age_aware=age_aware,
                age_penalty=age_penalty,
            )
        elif index > 0:
            # During a complete communication outage, hold the last cooperative
            # posterior rather than terminating the full simulation.
            state = fused_states[index - 1].copy()
            covariance = fused_covariances[index - 1].copy()
            weights = {}
        else:
            raise ValueError("No valid node report is available at the initial epoch.")
        fused_states[index] = state
        fused_covariances[index] = covariance
        weight_history.append(weights)
        active_history.append(active_nodes)
        received_history.append(list(weights.keys()))

    return CooperativeFusionHistory(
        timestamps=timestamps.copy(),
        state_history_eci=fused_states,
        covariance_history=fused_covariances,
        node_weight_history=weight_history,
        active_node_history=active_history,
        received_node_history=received_history,
    )
