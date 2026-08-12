from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class NeighborMeasurementQualityPolicy:
    """Deterministic two-level robust weighting for relative modalities."""

    base_inflation_by_modality: Mapping[str, float] = field(
        default_factory=dict
    )
    age_grace: float = 0.0
    age_inflation_per_second: float = 0.0
    loss_inflation_per_packet: float = 0.0
    resynchronization_inflation: float = 0.0
    maximum_inflation: float = 1e6

    def __post_init__(self) -> None:
        values = (
            *self.base_inflation_by_modality.values(),
            self.age_grace,
            self.age_inflation_per_second,
            self.loss_inflation_per_packet,
            self.resynchronization_inflation,
            self.maximum_inflation,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Quality-policy parameters must be nonnegative.")

    def inflation(
        self, *, modality: str, age: float,
        consecutive_losses: int, resynchronization_required: bool,
    ) -> float:
        if age < 0.0 or consecutive_losses < 0:
            raise ValueError("Age and consecutive losses must be nonnegative.")
        if modality not in self.base_inflation_by_modality:
            return 0.0
        value = float(self.base_inflation_by_modality[modality])
        value += self.age_inflation_per_second * max(
            0.0, float(age) - self.age_grace
        )
        value += self.loss_inflation_per_packet * int(consecutive_losses)
        if resynchronization_required:
            value += self.resynchronization_inflation
        return min(value, self.maximum_inflation)


@dataclass(frozen=True)
class NeighborLinkQuality:
    age: float = 0.0
    consecutive_losses: int = 0
    resynchronization_required: bool = False


def build_neighbor_link_quality_schedule(
    *, receiver_id, timestamps, neighbor_ids, message_records,
) -> dict[tuple[str, str, float], NeighborLinkQuality]:
    """Convert chronological transport records into epoch link quality."""

    times = tuple(float(value) for value in timestamps)
    records = sorted(
        message_records,
        key=lambda item: (
            float(item.get("current_timestamp", 0.0)),
            str(item.get("source_id", "")),
        ),
    )
    output = {}
    for neighbor in tuple(str(value) for value in neighbor_ids):
        last_source_timestamp = None
        losses = 0
        resync = False
        cursor = 0
        relevant = [
            record for record in records
            if str(record.get("receiver_id")) == str(receiver_id)
            and str(record.get("source_id")) == neighbor
        ]
        for timestamp in times:
            while cursor < len(relevant) and float(
                relevant[cursor].get("current_timestamp", 0.0)
            ) <= timestamp:
                record = relevant[cursor]
                if bool(record.get("accepted")):
                    last_source_timestamp = float(
                        record.get(
                            "message_timestamp",
                            record.get("current_timestamp"),
                        )
                    )
                    losses = int(
                        record.get("consecutive_losses_before_delivery", 0)
                    )
                resync = neighbor in set(
                    record.get("resync_required_neighbors", ())
                )
                cursor += 1
            output[(str(receiver_id), neighbor, timestamp)] = NeighborLinkQuality(
                age=(
                    max(0.0, timestamp - times[0])
                    if last_source_timestamp is None
                    else max(0.0, timestamp - last_source_timestamp)
                ),
                consecutive_losses=losses,
                resynchronization_required=resync,
            )
    return output
