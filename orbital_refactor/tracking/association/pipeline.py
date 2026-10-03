from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from tracking.data_contracts import GlobalTargetEstimate, TargetInitialState
from tracking.initial_orbit.data_contracts import IODObservation
from tracking.initial_orbit.multi_target_initializer import MultiTargetIODUpdate

from .candidate_manager import AutonomousIODManager
from .data_contracts import AssociationResult, UnlabeledIODObservation
from .global_nearest_neighbor import associate_to_tracks, label_matches


@dataclass(frozen=True)
class AssociationPipelineUpdate:
    association: AssociationResult
    observations_by_target: Mapping[str, tuple[IODObservation, ...]]
    autonomous_iod: MultiTargetIODUpdate


class MultiTargetAssociationPipeline:
    """Route unlabeled measurements to tracks or autonomous short-arc IOD."""

    def __init__(
        self,
        *,
        gate_squared_mahalanobis: float = 13.815510557964274,
        minimum_cost_margin: float = 2.0,
        autonomous_iod_manager: AutonomousIODManager | None = None,
    ):
        if gate_squared_mahalanobis <= 0.0:
            raise ValueError("gate_squared_mahalanobis must be positive.")
        self.gate_squared_mahalanobis = float(gate_squared_mahalanobis)
        if minimum_cost_margin < 0.0:
            raise ValueError("minimum_cost_margin must be non-negative.")
        self.minimum_cost_margin = float(minimum_cost_margin)
        self.autonomous_iod_manager = autonomous_iod_manager or AutonomousIODManager()

    def step(
        self,
        observations,
        tracks: Mapping[str, GlobalTargetEstimate | TargetInitialState],
    ) -> AssociationPipelineUpdate:
        values = tuple(observations)
        if any(not isinstance(item, UnlabeledIODObservation) for item in values):
            raise TypeError("Association pipeline requires unlabeled IOD observations.")
        association = associate_to_tracks(
            values,
            tracks,
            gate_squared_mahalanobis=self.gate_squared_mahalanobis,
            minimum_cost_margin=self.minimum_cost_margin,
        )
        labeled = label_matches(values, association)
        grouped: dict[str, list[IODObservation]] = {}
        for item in labeled:
            grouped.setdefault(item.target_id, []).append(item)
        ambiguous = set(association.ambiguous_observation_indices)
        unassigned = tuple(
            values[index] for index in association.unassigned_observation_indices
            if index not in ambiguous
        )
        autonomous = self.autonomous_iod_manager.ingest(unassigned)
        return AssociationPipelineUpdate(
            association=association,
            observations_by_target={
                target_id: tuple(items) for target_id, items in sorted(grouped.items())
            },
            autonomous_iod=autonomous,
        )

