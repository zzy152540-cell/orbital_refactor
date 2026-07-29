from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from interfaces.data_objects import NodeReport


@dataclass
class CommunicationChannel:
    """Simple stochastic communication channel.

    packet_loss_rate:
        probability that a generated NodeReport is lost.
    """

    packet_loss_rate: dict[str, float] = field(default_factory=dict)
    random_seed: int = 42

    def __post_init__(self):
        self.rng = np.random.default_rng(self.random_seed)

    def transmit(
        self,
        reports: list[NodeReport],
    ) -> list[NodeReport]:
        received = []

        for report in reports:
            loss = self.packet_loss_rate.get(
                report.node_id,
                0.0,
            )

            if self.rng.random() >= loss:
                received.append(report)

        return received
