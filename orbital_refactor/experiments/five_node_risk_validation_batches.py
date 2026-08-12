from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Iterable

from cooperative.topology_policy import UndirectedEdge
from experiments.short_horizon_counterfactual_study import (
    ShortHorizonCounterfactualStudy,
    action_kind_summaries,
    run_short_horizon_counterfactual_study,
    swap_abstention_summaries,
    swap_nis_retention_gate_summaries,
    swap_oracle_summary,
    swap_predictor_selection_summaries,
)


@dataclass(frozen=True)
class FiveNodeRiskValidationBatch:
    batch_id: str
    seeds: tuple[int, ...]
    decision_epochs: tuple[int, ...]
    horizon_epochs: tuple[int, ...]
    relative_modalities: tuple[str, ...]
    packet_loss_by_edge: tuple[tuple[UndirectedEdge, float], ...]
    communication_delay_by_edge: tuple[tuple[UndirectedEdge, float], ...]
    study: ShortHorizonCounterfactualStudy


def save_five_node_risk_validation_batch(
    batch: FiveNodeRiskValidationBatch, path: str | Path,
) -> Path:
    """Persist a generated batch for trusted local research reuse."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(serialize_five_node_risk_validation_batch(batch))
    return output


def load_five_node_risk_validation_batch(
    path: str | Path,
) -> FiveNodeRiskValidationBatch:
    """Load a batch created locally by the matching save function.

    Pickle files must only be loaded from trusted local experiment output.
    """

    return deserialize_five_node_risk_validation_batch(Path(path).read_bytes())


def serialize_five_node_risk_validation_batch(
    batch: FiveNodeRiskValidationBatch,
) -> bytes:
    return pickle.dumps(batch, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_five_node_risk_validation_batch(
    payload: bytes,
) -> FiveNodeRiskValidationBatch:
    value = pickle.loads(payload)
    if not isinstance(value, FiveNodeRiskValidationBatch):
        raise ValueError("File does not contain a five-node validation batch.")
    return value


def run_five_node_risk_validation_batch(
    *, batch_id: str, seeds: Iterable[int],
    decision_epochs: tuple[int, ...] = (2,),
    horizon_epochs: tuple[int, ...] = (2,),
    relative_modalities: tuple[str, ...] = (
        "RANGE", "RANGE_RATE", "AZ_EL",
    ),
    packet_loss_by_edge: dict[UndirectedEdge, float],
    communication_delay_by_edge: dict[UndirectedEdge, float],
) -> FiveNodeRiskValidationBatch:
    """Run one restartable five-node online validation partition."""

    seed_values = tuple(int(seed) for seed in seeds)
    if not batch_id or not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("batch_id and unique nonempty seeds are required.")
    study = run_short_horizon_counterfactual_study(
        node_counts=(5,), seeds=seed_values,
        decision_epochs=decision_epochs,
        horizon_epochs=horizon_epochs,
        relative_modalities=relative_modalities,
        backend="online_orchestrator",
        packet_loss_by_edge=packet_loss_by_edge,
        communication_delay_by_edge=communication_delay_by_edge,
        future_batch_relative_observations=True,
    )
    return FiveNodeRiskValidationBatch(
        batch_id=str(batch_id), seeds=seed_values,
        decision_epochs=tuple(decision_epochs),
        horizon_epochs=tuple(horizon_epochs),
        relative_modalities=tuple(relative_modalities),
        packet_loss_by_edge=tuple(sorted(packet_loss_by_edge.items())),
        communication_delay_by_edge=tuple(sorted(
            communication_delay_by_edge.items()
        )),
        study=study,
    )


def combine_five_node_risk_validation_batches(
    *batches: FiveNodeRiskValidationBatch,
) -> ShortHorizonCounterfactualStudy:
    """Merge compatible seed-disjoint batches into a recalibratable study."""

    if not batches:
        raise ValueError("At least one validation batch is required.")
    reference = batches[0]
    fields = (
        "decision_epochs", "horizon_epochs", "relative_modalities",
        "packet_loss_by_edge", "communication_delay_by_edge",
    )
    if any(
        getattr(batch, field) != getattr(reference, field)
        for batch in batches[1:] for field in fields
    ):
        raise ValueError("Validation batches have incompatible configurations.")
    batch_ids = tuple(batch.batch_id for batch in batches)
    seeds = tuple(seed for batch in batches for seed in batch.seeds)
    if len(set(batch_ids)) != len(batch_ids) or len(set(seeds)) != len(seeds):
        raise ValueError("Validation batch IDs and seeds must be disjoint.")
    records = tuple(
        record for batch in batches for record in batch.study.records
    )
    observations = tuple(
        value for batch in batches
        for value in batch.study.decision_observations
    )
    return ShortHorizonCounterfactualStudy(
        node_counts=(5,), seeds=tuple(sorted(seeds)),
        decision_epochs=reference.decision_epochs,
        horizon_epochs=reference.horizon_epochs,
        relative_modalities=reference.relative_modalities,
        decision_observations=observations,
        records=records,
        summaries_by_action_kind=action_kind_summaries(records),
        swap_oracle_summary=swap_oracle_summary(records),
        swap_predictor_summaries=swap_predictor_selection_summaries(records),
        swap_nis_retention_gate_summaries=(
            swap_nis_retention_gate_summaries(records)
        ),
        swap_abstention_summaries=swap_abstention_summaries(records),
    )
