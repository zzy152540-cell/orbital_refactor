from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from interfaces.data_objects import NodeReport


@dataclass
class DelayChannel:
    """Fixed communication delay model."""

    delay_by_node: dict[str, float] = field(default_factory=dict)

    def transmit(
        self,
        reports: list[NodeReport],
    ) -> list[NodeReport]:
        delayed = []
        for report in reports:
            delay = float(self.delay_by_node.get(report.node_id, 0.0))
            delayed.append(
                NodeReport(
                    node_id=report.node_id,
                    target_id=report.target_id,
                    timestamp=report.timestamp,
                    state_estimate=report.state_estimate.copy(),
                    covariance=report.covariance.copy(),
                    quality_score=report.quality_score,
                    health_status=report.health_status,
                    communication_delay=delay,
                    valid_flag=report.valid_flag,
                    source_timestamp=report.timestamp,
                    arrival_timestamp=report.timestamp + delay,
                )
            )
        return delayed
