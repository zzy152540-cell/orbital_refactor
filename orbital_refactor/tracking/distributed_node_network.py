"""Independent per-observer runtimes connected by target-level Consensus-CI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from .association.online_tracker import (
    ConsensusFeedbackApplication,
    OnlineMultiTargetTracker,
    OnlineTrackingUpdate,
)
from .distributed_target_consensus import (
    DistributedTargetConsensus,
    DistributedTargetConsensusStep,
)
from .distributed_identity import (
    CooperativeTrackIdentityResolver,
    TargetIdentityResolution,
)


@dataclass(frozen=True)
class DistributedNodeNetworkUpdate:
    timestamp: float
    tracking_by_node: Mapping[str, OnlineTrackingUpdate]
    consensus: DistributedTargetConsensusStep
    feedback_by_node: Mapping[str, ConsensusFeedbackApplication]
    identity_resolution: TargetIdentityResolution | None = None


class DistributedTargetNodeNetwork:
    """Run one isolated tracker per observer and exchange only target reports."""

    def __init__(
        self,
        *,
        trackers_by_node: Mapping[str, OnlineMultiTargetTracker],
        consensus: DistributedTargetConsensus,
        identity_resolver: CooperativeTrackIdentityResolver | None = None,
    ) -> None:
        trackers = dict(trackers_by_node)
        if set(trackers) != set(consensus.topology.node_ids):
            raise ValueError("Tracker node IDs must match the consensus topology.")
        if any(tracker.scene_id != consensus.scene_id for tracker in trackers.values()):
            raise ValueError("Every node tracker must use the consensus scene ID.")
        self.trackers_by_node = trackers
        self.consensus = consensus
        self.identity_resolver = identity_resolver

    def step(
        self,
        *,
        timestamp: float,
        observations_by_node: Mapping[str, tuple],
        observer_states_eci: Mapping[str, np.ndarray],
        q_eci2pri_by_observer: Mapping[str, np.ndarray],
    ) -> DistributedNodeNetworkUpdate:
        active_nodes = set(observer_states_eci)
        if active_nodes != set(q_eci2pri_by_observer):
            raise ValueError("Active observer states and attitudes must use identical IDs.")
        if not active_nodes <= set(self.trackers_by_node):
            raise ValueError("Active observers must belong to the configured node network.")
        unknown_observation_nodes = set(observations_by_node) - set(self.trackers_by_node)
        if unknown_observation_nodes:
            raise ValueError(
                f"Observation batches reference unknown nodes: {sorted(unknown_observation_nodes)}"
            )

        tracking_by_node = {}
        reports = []
        for node_id in sorted(active_nodes):
            tracking = self.trackers_by_node[node_id].step(
                timestamp=timestamp,
                observations=observations_by_node.get(node_id, ()),
                observer_states_eci={node_id: observer_states_eci[node_id]},
                q_eci2pri_by_observer={node_id: q_eci2pri_by_observer[node_id]},
            )
            tracking_by_node[node_id] = tracking
            reports.extend(tracking.local_reports)
            if self.identity_resolver is not None:
                for target_id in tracking.retired_target_ids:
                    retired_track = tracking.tracks_by_target[target_id]
                    self.identity_resolver.retire_local_track(
                        observer_id=node_id,
                        local_target_id=target_id,
                        local_track_id=retired_track.estimate.track_id,
                    )

        identity_resolution = (
            None
            if self.identity_resolver is None
            else self.identity_resolver.resolve(reports)
        )
        consensus_reports = (
            reports if identity_resolution is None else identity_resolution.reports
        )
        consensus_step = self.consensus.step(
            timestamp=timestamp,
            local_reports=consensus_reports,
            active_source_ids=active_nodes,
        )
        feedback_by_node = {}
        for node_id in tracking_by_node:
            node_estimates = consensus_step.estimates_by_node[node_id]
            if self.identity_resolver is not None:
                node_estimates = self.identity_resolver.localize_estimates(
                    observer_id=node_id,
                    estimates=node_estimates,
                )
            node_consensus = replace(
                consensus_step,
                estimates_by_node={node_id: node_estimates},
            )
            feedback_by_node[node_id] = self.trackers_by_node[
                node_id
            ].apply_consensus_feedback(node_consensus)

        return DistributedNodeNetworkUpdate(
            timestamp=float(timestamp),
            tracking_by_node=tracking_by_node,
            consensus=consensus_step,
            feedback_by_node=feedback_by_node,
            identity_resolution=identity_resolution,
        )
