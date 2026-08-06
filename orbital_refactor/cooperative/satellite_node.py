from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interfaces.data_objects import NodeReport
from orbital_core.quality import quality_score_from_covariance

Array = np.ndarray


@dataclass(frozen=True)
class NodeEstimate:
    node_id: str
    timestamp: float
    state: Array
    covariance: Array
    acceleration: Array
    quality: float
    health_status: str = "NORMAL"
    valid_flag: bool = True

    def to_report(self) -> NodeReport:
        return NodeReport(
            node_id=self.node_id,
            target_id=self.node_id,
            timestamp=float(self.timestamp),
            state_estimate=np.asarray(self.state, dtype=float).reshape(6).copy(),
            covariance=np.asarray(self.covariance, dtype=float).reshape(6, 6).copy(),
            quality_score=float(self.quality),
            health_status=self.health_status,
            communication_delay=0.0,
            valid_flag=bool(self.valid_flag),
            source_timestamp=float(self.timestamp),
            arrival_timestamp=float(self.timestamp),
        )


@dataclass(frozen=True)
class SatelliteNode:
    """Minimal v13 satellite node state holder.

    This class represents the new model where each satellite owns its own
    estimate, rather than serving only as an observer for a shared target.
    """

    node_id: str
    state: Array
    covariance: Array
    acceleration: Array | None = None
    health_status: str = "NORMAL"
    valid_flag: bool = True

    def estimate(self, timestamp: float) -> NodeEstimate:
        covariance = np.asarray(self.covariance, dtype=float).reshape(6, 6)
        acceleration = (
            np.zeros(3, dtype=float)
            if self.acceleration is None
            else np.asarray(self.acceleration, dtype=float).reshape(3)
        )
        return NodeEstimate(
            node_id=self.node_id,
            timestamp=float(timestamp),
            state=np.asarray(self.state, dtype=float).reshape(6).copy(),
            covariance=covariance.copy(),
            acceleration=acceleration.copy(),
            quality=quality_score_from_covariance(covariance),
            health_status=self.health_status,
            valid_flag=bool(self.valid_flag),
        )

    def with_posterior(self, state: Array, covariance: Array) -> "SatelliteNode":
        return SatelliteNode(
            node_id=self.node_id,
            state=np.asarray(state, dtype=float).reshape(6).copy(),
            covariance=np.asarray(covariance, dtype=float).reshape(6, 6).copy(),
            acceleration=(
                None
                if self.acceleration is None
                else np.asarray(self.acceleration, dtype=float).reshape(3).copy()
            ),
            health_status=self.health_status,
            valid_flag=self.valid_flag,
        )
