from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from experiments.short_horizon_counterfactual_study import (
    ShortHorizonActionRecord,
    ShortHorizonCounterfactualStudy,
)


@dataclass(frozen=True)
class ActionLabelStabilityCell:
    action_kind: str
    decision_epoch: int
    horizon_epochs: int
    replica_count: int
    safe_positive_rate: float
    pairwise_label_agreement: float
    mean_position_rmse_reduction: float
    position_rmse_standard_deviation: float


@dataclass(frozen=True)
class ActionLabelStabilitySummary:
    action_kind: str
    cell_count: int
    minimum_replica_count: int
    mean_safe_positive_rate: float
    mean_pairwise_label_agreement: float
    unanimous_cell_rate: float
    ambiguous_cell_rate: float
    mean_position_rmse_reduction: float
    mean_position_rmse_standard_deviation: float


@dataclass(frozen=True)
class ActionLabelStabilityReport:
    seed_count: int
    cells: tuple[ActionLabelStabilityCell, ...]
    overall: ActionLabelStabilitySummary
    by_action_kind: tuple[ActionLabelStabilitySummary, ...]


def analyze_graph_action_label_stability(
    study: ShortHorizonCounterfactualStudy,
) -> ActionLabelStabilityReport:
    """Align identical actions across Monte Carlo seeds and audit label noise."""

    grouped = defaultdict(list)
    for record in study.records:
        if record.action_kind != "keep":
            grouped[_action_signature(record)].append(record)
    cells = tuple(
        _cell(records)
        for records in grouped.values()
        if len({record.seed for record in records}) >= 2
    )
    if not cells:
        raise ValueError("Label stability requires replicated actions across seeds.")
    action_kinds = sorted({cell.action_kind for cell in cells})
    return ActionLabelStabilityReport(
        seed_count=len(set(study.seeds)),
        cells=cells,
        overall=_summary("all", cells),
        by_action_kind=tuple(
            _summary(kind, tuple(
                cell for cell in cells if cell.action_kind == kind
            ))
            for kind in action_kinds
        ),
    )


def _action_signature(record: ShortHorizonActionRecord):
    return (
        record.node_count,
        record.decision_epoch,
        record.horizon_epochs,
        record.action_kind,
        tuple(sorted(record.active_edges)),
        tuple(sorted(record.added_edges)),
        tuple(sorted(record.removed_edges)),
    )


def _cell(records):
    first = records[0]
    labels = np.asarray([
        record.position_rmse_reduction > 0.0
        and record.nees_calibration_improvement >= 0.0
        and record.nees_coverage_calibration_improvement >= 0.0
        for record in records
    ], dtype=bool)
    gains = np.asarray([
        record.position_rmse_reduction for record in records
    ], dtype=float)
    positive = int(np.sum(labels))
    negative = len(labels) - positive
    pair_count = len(labels) * (len(labels) - 1) // 2
    agreeing_pairs = (
        positive * (positive - 1) // 2
        + negative * (negative - 1) // 2
    )
    return ActionLabelStabilityCell(
        action_kind=first.action_kind,
        decision_epoch=first.decision_epoch,
        horizon_epochs=first.horizon_epochs,
        replica_count=len(records),
        safe_positive_rate=float(np.mean(labels)),
        pairwise_label_agreement=agreeing_pairs / pair_count,
        mean_position_rmse_reduction=float(np.mean(gains)),
        position_rmse_standard_deviation=float(np.std(gains)),
    )


def _summary(action_kind, cells):
    positive_rates = np.asarray([
        cell.safe_positive_rate for cell in cells
    ])
    return ActionLabelStabilitySummary(
        action_kind=action_kind,
        cell_count=len(cells),
        minimum_replica_count=min(cell.replica_count for cell in cells),
        mean_safe_positive_rate=float(np.mean(positive_rates)),
        mean_pairwise_label_agreement=float(np.mean([
            cell.pairwise_label_agreement for cell in cells
        ])),
        unanimous_cell_rate=float(np.mean(
            (positive_rates == 0.0) | (positive_rates == 1.0)
        )),
        ambiguous_cell_rate=float(np.mean(
            (positive_rates >= 0.25) & (positive_rates <= 0.75)
        )),
        mean_position_rmse_reduction=float(np.mean([
            cell.mean_position_rmse_reduction for cell in cells
        ])),
        mean_position_rmse_standard_deviation=float(np.mean([
            cell.position_rmse_standard_deviation for cell in cells
        ])),
    )
