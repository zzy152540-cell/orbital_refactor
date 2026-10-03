"""Target-centric distributed consensus for multi-observer tracking.

This module is the bridge between the multi-target tracker and the existing
network-topology/message-channel infrastructure.  Fusion is always partitioned
by physical target and track identity.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace

import numpy as np

from cooperative.message_transport import MessageChannel, TypedMessageBuffer
from cooperative.temporal_alignment import propagate_state_covariance
from cooperative.topology import NetworkTopology

from .data_contracts import (
    GlobalTargetEstimate,
    TargetEstimateMessage,
    TargetNodeReport,
)
from .target_fusion import fuse_target_reports


@dataclass(frozen=True)
class DistributedTargetConsensusStep:
    """Per-node target posteriors and communication diagnostics for one epoch."""

    timestamp: float
    estimates_by_node: Mapping[str, Mapping[str, GlobalTargetEstimate]]
    received_message_ids_by_node: Mapping[str, tuple[str, ...]]
    rejected_message_ids_by_node: Mapping[str, tuple[str, ...]]
    available_source_ids_by_node: Mapping[str, tuple[str, ...]]
    recovered_source_ids_by_node: Mapping[str, tuple[str, ...]]
    pending_message_count: int


class DistributedTargetConsensus:
    """Stateful one-hop Consensus-CI exchange partitioned by target identity."""

    def __init__(
        self,
        *,
        scene_id: str,
        topology: NetworkTopology,
        message_channel: MessageChannel | None = None,
        process_noise_acceleration: float = 1e-4,
        objective: str = "trace",
        grid_points: int = 31,
    ) -> None:
        self.scene_id = str(scene_id).strip()
        if not self.scene_id:
            raise ValueError("scene_id must be a non-empty identifier.")
        if process_noise_acceleration < 0.0:
            raise ValueError("process_noise_acceleration cannot be negative.")
        self.topology = topology
        self.message_channel = message_channel
        self.process_noise_acceleration = float(process_noise_acceleration)
        self.objective = str(objective)
        self.grid_points = int(grid_points)
        self._buffers = {
            node_id: TypedMessageBuffer[TargetEstimateMessage]()
            for node_id in topology.node_ids
        }
        self._last_timestamp: float | None = None
        self._links_awaiting_recovery: set[tuple[str, str]] = set()

    def step(
        self,
        *,
        timestamp: float,
        local_reports: Iterable[TargetNodeReport],
        active_source_ids: Iterable[str] | None = None,
    ) -> DistributedTargetConsensusStep:
        epoch = float(timestamp)
        if not np.isfinite(epoch):
            raise ValueError("timestamp must be finite.")
        if self._last_timestamp is not None and epoch <= self._last_timestamp:
            raise ValueError("Consensus timestamps must be strictly increasing.")

        reports = tuple(local_reports)
        self._validate_reports(reports, epoch)
        report_source_ids = {report.observer_id for report in reports}
        active_sources = (
            report_source_ids
            if active_source_ids is None
            else {str(value) for value in active_source_ids}
        )
        unknown_sources = active_sources - set(self.topology.node_ids)
        if unknown_sources:
            raise ValueError(
                f"Active sources are absent from the topology: {sorted(unknown_sources)}"
            )
        local_by_node: dict[str, list[TargetNodeReport]] = defaultdict(list)
        delivery_succeeded_by_source = {
            node_id: False for node_id in self.topology.node_ids
        }
        for report in reports:
            local_by_node[report.observer_id].append(report)
            message = self._message_from_report(report)
            delivered = (
                message
                if self.message_channel is None
                else self.message_channel.transmit(message)
            )
            if delivered is None:
                continue
            delivery_succeeded_by_source[report.observer_id] = True
            for neighbor_id in self.topology.neighbors(report.observer_id):
                self._buffers[neighbor_id].push(delivered)

        for source_id, succeeded in delivery_succeeded_by_source.items():
            link_failed = (
                source_id not in active_sources
                or (source_id in report_source_ids and not succeeded)
            )
            if not link_failed:
                continue
            for receiver_id in self.topology.neighbors(source_id):
                self._links_awaiting_recovery.add((receiver_id, source_id))

        estimates_by_node: dict[str, dict[str, GlobalTargetEstimate]] = {}
        received_by_node: dict[str, tuple[str, ...]] = {}
        rejected_by_node: dict[str, tuple[str, ...]] = {}
        available_by_node: dict[str, tuple[str, ...]] = {}
        recovered_by_node: dict[str, tuple[str, ...]] = {}
        for node_id in self.topology.node_ids:
            incoming = self._buffers[node_id].pop_available(epoch)
            received_by_node[node_id] = tuple(message.message_id for message in incoming)
            available_sources = tuple(sorted({
                message.source_node_id for message in incoming
            }))
            available_by_node[node_id] = available_sources
            recovered_sources = tuple(sorted(
                source_id for source_id in available_sources
                if (node_id, source_id) in self._links_awaiting_recovery
            ))
            recovered_by_node[node_id] = recovered_sources
            for source_id in recovered_sources:
                self._links_awaiting_recovery.discard((node_id, source_id))
            estimates, rejected = self._fuse_for_node(
                node_id=node_id,
                timestamp=epoch,
                local_reports=local_by_node.get(node_id, ()),
                incoming=incoming,
            )
            estimates_by_node[node_id] = estimates
            rejected_by_node[node_id] = tuple(rejected)

        self._last_timestamp = epoch
        return DistributedTargetConsensusStep(
            timestamp=epoch,
            estimates_by_node=estimates_by_node,
            received_message_ids_by_node=received_by_node,
            rejected_message_ids_by_node=rejected_by_node,
            available_source_ids_by_node=available_by_node,
            recovered_source_ids_by_node=recovered_by_node,
            pending_message_count=sum(len(buffer) for buffer in self._buffers.values()),
        )

    def _validate_reports(
        self,
        reports: tuple[TargetNodeReport, ...],
        timestamp: float,
    ) -> None:
        seen: set[tuple[str, str]] = set()
        for report in reports:
            if report.observer_id not in self.topology.node_ids:
                raise ValueError(
                    f"Report source {report.observer_id!r} is absent from the topology."
                )
            if not np.isclose(report.timestamp, timestamp):
                raise ValueError("Every local report must match the consensus timestamp.")
            key = (report.observer_id, report.target_id)
            if key in seen:
                raise ValueError("Only one local report per observer and target is allowed.")
            seen.add(key)

    def _message_from_report(self, report: TargetNodeReport) -> TargetEstimateMessage:
        information_ids = report.used_measurement_ids or (
            f"local:{report.observer_id}:{report.target_id}:{report.timestamp:.9f}",
        )
        return TargetEstimateMessage(
            message_id=(
                f"{self.scene_id}:{report.observer_id}:{report.target_id}:"
                f"{report.track_id}:{report.timestamp:.9f}"
            ),
            scene_id=self.scene_id,
            source_node_id=report.observer_id,
            target_id=report.target_id,
            track_id=report.track_id,
            timestamp=report.timestamp,
            state_eci=report.state_eci,
            covariance_eci=report.covariance_eci,
            quality_score=report.quality_score,
            valid_flag=report.valid_flag,
            information_ids=information_ids,
            lineage_id=f"{self.scene_id}:{report.target_id}:{report.track_id}",
        )

    def _fuse_for_node(
        self,
        *,
        node_id: str,
        timestamp: float,
        local_reports: Iterable[TargetNodeReport],
        incoming: Iterable[TargetEstimateMessage],
    ) -> tuple[dict[str, GlobalTargetEstimate], list[str]]:
        local = tuple(local_reports)
        messages = tuple(incoming)
        local_track_by_target = {item.target_id: item.track_id for item in local}
        grouped: dict[tuple[str, str], list[TargetNodeReport]] = defaultdict(list)
        used_information: dict[tuple[str, str], set[str]] = defaultdict(set)
        rejected: list[str] = []

        for report in local:
            if not report.valid_flag:
                continue
            key = (report.target_id, report.track_id)
            grouped[key].append(report)
            used_information[key].update(report.used_measurement_ids)

        for message in sorted(messages, key=lambda item: item.message_id):
            expected_lineage = f"{self.scene_id}:{message.target_id}:{message.track_id}"
            local_track = local_track_by_target.get(message.target_id)
            if (
                message.scene_id != self.scene_id
                or message.lineage_id != expected_lineage
                or (local_track is not None and message.track_id != local_track)
                or not message.valid_flag
            ):
                rejected.append(message.message_id)
                continue
            key = (message.target_id, message.track_id)
            information_ids = set(message.information_ids)
            if information_ids and information_ids <= used_information[key]:
                rejected.append(message.message_id)
                continue
            aligned = self._aligned_report(message, timestamp)
            grouped[key].append(aligned)
            used_information[key].update(information_ids)

        estimates: dict[str, GlobalTargetEstimate] = {}
        for (target_id, _track_id), target_reports in sorted(grouped.items()):
            if target_id in estimates:
                raise ValueError(
                    f"Node {node_id!r} received competing track IDs for target {target_id!r}."
                )
            estimates[target_id] = fuse_target_reports(
                target_reports,
                objective=self.objective,
                grid_points=self.grid_points,
            )
        return estimates, rejected

    def _aligned_report(
        self,
        message: TargetEstimateMessage,
        timestamp: float,
    ) -> TargetNodeReport:
        source_timestamp = float(
            message.timestamp
            if message.source_timestamp is None
            else message.source_timestamp
        )
        dt = float(timestamp) - source_timestamp
        if dt < -1e-12:
            raise ValueError("Target estimate messages cannot travel backward in time.")
        state = message.state_eci
        covariance = message.covariance_eci
        if dt > 1e-12:
            state, covariance = propagate_state_covariance(
                state,
                covariance,
                dt,
                process_noise_acceleration=self.process_noise_acceleration,
            )
        return TargetNodeReport(
            observer_id=message.source_node_id,
            target_id=message.target_id,
            track_id=message.track_id,
            timestamp=timestamp,
            state_eci=state,
            covariance_eci=covariance,
            quality_score=message.quality_score,
            valid_flag=message.valid_flag,
            frame=message.frame,
            used_measurement_ids=message.information_ids,
        )
