from __future__ import annotations

from dataclasses import dataclass, field

from interfaces.data_objects import NodeReport


@dataclass
class MessageBuffer:
    """Store delayed communication messages and release arrived reports."""

    reports: list[NodeReport] = field(default_factory=list)

    def push(self, report: NodeReport):
        self.reports.append(report)

    def pop_available(self, current_time: float) -> list[NodeReport]:
        available = []
        remain = []
        for report in self.reports:
            arrival = report.arrival_timestamp
            if arrival is None or arrival <= current_time:
                available.append(report)
            else:
                remain.append(report)
        self.reports = remain
        return available
