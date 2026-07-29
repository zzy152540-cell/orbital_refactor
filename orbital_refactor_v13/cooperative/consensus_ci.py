from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from cooperative.communication_channel import CommunicationChannel
from cooperative.delay_channel import DelayChannel
from cooperative.satellite_node import NodeEstimate, SatelliteNode
from cooperative.topology import NetworkTopology
from interfaces.data_objects import NodeReport
from orbital_core.ci_fusion import ci_fuse_posteriors

Array = np.ndarray


@dataclass(frozen=True)
class ConsensusStepResult:
    estimates: dict[str, NodeEstimate]
    node_weight_by_node: dict[str, dict[str, float]]
    received_reports_by_node: dict[str, list[str]]


def run_consensus_ci_step(
    *,
    nodes: Mapping[str, SatelliteNode],
    topology: NetworkTopology,
    timestamp: float,
    objective: str = "trace",
    grid_points: int = 31,
    communication_channel: CommunicationChannel | None = None,
    delay_channel: DelayChannel | None = None,
) -> ConsensusStepResult:
    """Run one distributed Consensus-CI exchange.

    Each satellite fuses only its own report with reports from its configured
    neighbors. The output remains per-node and intentionally does not create a
    fleet-level global state.
    """

    if set(nodes) != set(topology.node_ids):
        raise ValueError("nodes keys must match topology node IDs.")
    local_reports = {
        node_id: node.estimate(timestamp).to_report()
        for node_id, node in nodes.items()
    }
    inboxes = _deliver_neighbor_reports(
        local_reports=local_reports,
        topology=topology,
        communication_channel=communication_channel,
        delay_channel=delay_channel,
    )

    estimates: dict[str, NodeEstimate] = {}
    weights_by_node: dict[str, dict[str, float]] = {}
    received_by_node: dict[str, list[str]] = {}
    for node_id, node in nodes.items():
        reports = [local_reports[node_id], *inboxes[node_id]]
        valid = [report for report in reports if report.valid_flag]
        if not valid:
            estimate = node.estimate(timestamp)
            estimates[node_id] = estimate
            weights_by_node[node_id] = {}
            received_by_node[node_id] = []
            continue
        fusion = ci_fuse_posteriors(
            [
                (report.node_id, report.state_estimate, report.covariance)
                for report in valid
            ],
            objective=objective,
            grid_points=grid_points,
        )
        updated = node.with_posterior(fusion.state, fusion.covariance)
        estimates[node_id] = updated.estimate(timestamp)
        weights_by_node[node_id] = dict(fusion.weights)
        received_by_node[node_id] = [report.node_id for report in valid if report.node_id != node_id]

    return ConsensusStepResult(
        estimates=estimates,
        node_weight_by_node=weights_by_node,
        received_reports_by_node=received_by_node,
    )


def _deliver_neighbor_reports(
    *,
    local_reports: Mapping[str, NodeReport],
    topology: NetworkTopology,
    communication_channel: CommunicationChannel | None,
    delay_channel: DelayChannel | None,
) -> dict[str, list[NodeReport]]:
    inboxes = {node_id: [] for node_id in topology.node_ids}
    for source_id, report in local_reports.items():
        outgoing = [report]
        if communication_channel is not None:
            outgoing = communication_channel.transmit(outgoing)
        if delay_channel is not None:
            outgoing = delay_channel.transmit(outgoing)
        for delivered in outgoing:
            for target_id in topology.neighbors(source_id):
                inboxes[target_id].append(delivered)
    return inboxes
