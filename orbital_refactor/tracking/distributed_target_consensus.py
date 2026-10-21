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

from cooperative.message_transport import (
    MessageChannel,
    TypedMessageBuffer,
    message_source_id,
)
from cooperative.temporal_alignment import propagate_state_covariance
from cooperative.topology import NetworkTopology

from .data_contracts import (
    GlobalTargetEstimate,
    RADIAL_RANGE_CAPABILITY,
    RADIAL_RANGE_RATE_CAPABILITY,
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
    """Stateful bounded-hop Consensus-CI exchange by target identity."""

    def __init__(
        self,
        *,
        scene_id: str,
        topology: NetworkTopology,
        message_channel: MessageChannel | None = None,
        process_noise_acceleration: float = 1e-4,
        objective: str = "trace",
        grid_points: int = 31,
        prefer_complete_radial_lineage: bool = False,
        exchange_hops: int = 1,
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
        self.prefer_complete_radial_lineage = bool(
            prefer_complete_radial_lineage
        )
        if int(exchange_hops) < 1:
            raise ValueError("exchange_hops must be positive.")
        if int(exchange_hops) > 1 and message_channel is not None:
            raise ValueError(
                "Multi-hop synchronous exchange does not support a delayed "
                "or lossy message channel."
            )
        self.exchange_hops = int(exchange_hops)
        self._buffers = {
            node_id: TypedMessageBuffer[TargetEstimateMessage]()
            for node_id in topology.node_ids
        }
        self._last_timestamp: float | None = None
        self._links_awaiting_recovery: set[tuple[str, str]] = set()

    def update_topology(self, topology: NetworkTopology) -> None:
        """Switch edges or membership while preserving valid in-flight data.

        Buffers belonging to retained nodes survive. Buffers for joining nodes
        start empty, and messages originating from departed nodes are removed
        so that stale membership cannot contribute to later CI updates.
        """

        old_node_ids = set(self.topology.node_ids)
        new_node_ids = set(topology.node_ids)
        new_links = {
            (receiver_id, source_id)
            for source_id in topology.node_ids
            for receiver_id in topology.neighbors(source_id)
        }
        old_links = {
            (receiver_id, source_id)
            for source_id in self.topology.node_ids
            for receiver_id in self.topology.neighbors(source_id)
        }
        self._links_awaiting_recovery = {
            link for link in self._links_awaiting_recovery if link in new_links
        }
        self._links_awaiting_recovery.update(new_links - old_links)
        retained_buffers = {
            node_id: self._buffers[node_id]
            for node_id in sorted(old_node_ids & new_node_ids)
        }
        for buffer in retained_buffers.values():
            buffer.discard_where(
                lambda message: message_source_id(message) not in new_node_ids
            )
        self._buffers = {
            node_id: retained_buffers.get(
                node_id, TypedMessageBuffer[TargetEstimateMessage](),
            )
            for node_id in topology.node_ids
        }
        self.topology = topology

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
            for receiver_id in self._exchange_receivers(report.observer_id):
                self._buffers[receiver_id].push(delivered)

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

    def _exchange_receivers(self, source_id: str) -> tuple[str, ...]:
        """Return nodes reached within the configured synchronous hop bound."""

        visited = {source_id}
        frontier = {source_id}
        for _ in range(self.exchange_hops):
            next_frontier = {
                neighbor_id
                for node_id in frontier
                for neighbor_id in self.topology.neighbors(node_id)
                if neighbor_id not in visited
            }
            if not next_frontier:
                break
            visited.update(next_frontier)
            frontier = next_frontier
        return tuple(sorted(visited - {source_id}))

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
            information_capabilities=report.information_capabilities,
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
            selected_reports = target_reports
            if self.prefer_complete_radial_lineage:
                complete_radial = [
                    report for report in target_reports
                    if {
                        RADIAL_RANGE_CAPABILITY,
                        RADIAL_RANGE_RATE_CAPABILITY,
                    } <= set(report.information_capabilities)
                ]
                if complete_radial:
                    selected_reports = complete_radial
            estimates[target_id] = fuse_target_reports(
                selected_reports,
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
            information_capabilities=message.information_capabilities,
        )
