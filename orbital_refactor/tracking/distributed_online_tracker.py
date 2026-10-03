"""One-call orchestration for online multi-target distributed estimation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

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


@dataclass(frozen=True)
class DistributedOnlineTrackingUpdate:
    tracking: OnlineTrackingUpdate
    consensus: DistributedTargetConsensusStep
    feedback: ConsensusFeedbackApplication


class DistributedOnlineMultiTargetTracker:
    """Run local filtering, target-wise network CI, and prior feedback."""

    def __init__(
        self,
        *,
        tracker: OnlineMultiTargetTracker,
        consensus: DistributedTargetConsensus,
    ) -> None:
        if tracker.scene_id != consensus.scene_id:
            raise ValueError("Tracker and consensus scene IDs must match.")
        self.tracker = tracker
        self.consensus = consensus

    def step(
        self,
        *,
        timestamp: float,
        observations,
        observer_states_eci: Mapping[str, np.ndarray],
        q_eci2pri_by_observer: Mapping[str, np.ndarray],
    ) -> DistributedOnlineTrackingUpdate:
        tracking = self.tracker.step(
            timestamp=timestamp,
            observations=observations,
            observer_states_eci=observer_states_eci,
            q_eci2pri_by_observer=q_eci2pri_by_observer,
        )
        consensus = self.consensus.step(
            timestamp=timestamp,
            local_reports=tracking.local_reports,
            active_source_ids=observer_states_eci,
        )
        feedback = self.tracker.apply_consensus_feedback(consensus)
        return DistributedOnlineTrackingUpdate(
            tracking=tracking,
            consensus=consensus,
            feedback=feedback,
        )
