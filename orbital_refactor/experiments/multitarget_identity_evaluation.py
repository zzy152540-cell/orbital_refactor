"""Identity-preserving truth scoring for multi-target experiment runners."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class IdentityAwareErrorSample:
    timestamp: float
    track_id: str
    truth_id: str
    position_error_m: float
    velocity_error_mps: float


@dataclass(frozen=True)
class IdentityAwareEpochResult:
    timestamp: float
    samples_by_truth: Mapping[str, IdentityAwareErrorSample]
    missing_truth_ids: tuple[str, ...]
    unassigned_track_ids: tuple[str, ...]


class IdentityAwareTrackEvaluator:
    """Lock track-to-truth identity and expose reassignment diagnostics.

    Hungarian assignment is used when a track first appears.  Once assigned,
    the track continues to be scored against the same physical truth.  A later
    frame-wise nearest assignment is diagnostic only and cannot hide a swap by
    changing the RMSE reference.
    """

    def __init__(self, truth_ids) -> None:
        identifiers = tuple(str(value) for value in truth_ids)
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError("truth_ids must be nonempty and unique.")
        self.truth_ids = identifiers
        self.assignment_by_track_id: dict[str, str] = {}
        self.active_track_id_by_truth: dict[str, str] = {}
        self.previous_nearest_truth_by_track_id: dict[str, str] = {}
        self.state_history_by_truth: dict[
            str, list[tuple[float, np.ndarray]]
        ] = {truth_id: [] for truth_id in identifiers}
        self.error_history_by_truth: dict[
            str, list[IdentityAwareErrorSample]
        ] = {truth_id: [] for truth_id in identifiers}
        self.identity_switch_count = 0
        self.nearest_truth_disagreement_count = 0
        self.track_replacement_count = 0
        self.missing_truth_sample_count = 0
        self.unassigned_track_sample_count = 0
        self._started = False

    def update(
        self,
        *,
        timestamp: float,
        state_by_track_id: Mapping[str, np.ndarray],
        truth_state_by_id: Mapping[str, np.ndarray],
    ) -> IdentityAwareEpochResult:
        epoch = float(timestamp)
        if not np.isfinite(epoch):
            raise ValueError("timestamp must be finite.")
        if set(truth_state_by_id) != set(self.truth_ids):
            raise ValueError("truth_state_by_id must match configured truth_ids.")
        estimates = {
            str(track_id): _state(state, "estimated state")
            for track_id, state in state_by_track_id.items()
        }
        truths = {
            str(truth_id): _state(state, "truth state")
            for truth_id, state in truth_state_by_id.items()
        }
        self._assign_new_tracks(estimates, truths)
        optimal = _minimum_position_assignment(estimates, truths)
        for track_id, truth_id in optimal.items():
            previous = self.previous_nearest_truth_by_track_id.get(track_id)
            if previous is not None and previous != truth_id:
                self.identity_switch_count += 1
            self.previous_nearest_truth_by_track_id[track_id] = truth_id

        active_assignments = {
            track_id: truth_id
            for track_id, truth_id in self.assignment_by_track_id.items()
            if track_id in estimates
            and self.active_track_id_by_truth.get(truth_id) == track_id
        }
        samples = {}
        for track_id, truth_id in active_assignments.items():
            if optimal.get(track_id) not in {None, truth_id}:
                self.nearest_truth_disagreement_count += 1
            estimate = estimates[track_id]
            truth = truths[truth_id]
            sample = IdentityAwareErrorSample(
                timestamp=epoch,
                track_id=track_id,
                truth_id=truth_id,
                position_error_m=float(np.linalg.norm(
                    estimate[:3] - truth[:3]
                )),
                velocity_error_mps=float(np.linalg.norm(
                    estimate[3:] - truth[3:]
                )),
            )
            samples[truth_id] = sample
            self.state_history_by_truth[truth_id].append(
                (epoch, estimate.copy())
            )
            self.error_history_by_truth[truth_id].append(sample)

        missing = tuple(sorted(set(self.truth_ids) - set(samples)))
        unassigned = tuple(sorted(set(estimates) - set(active_assignments)))
        if estimates:
            self._started = True
        if self._started:
            self.missing_truth_sample_count += len(missing)
            self.unassigned_track_sample_count += len(unassigned)
        return IdentityAwareEpochResult(
            timestamp=epoch,
            samples_by_truth=MappingProxyType(samples),
            missing_truth_ids=missing,
            unassigned_track_ids=unassigned,
        )

    def position_rmse_by_truth(self) -> dict[str, float]:
        return {
            truth_id: _rmse(
                sample.position_error_m
                for sample in self.error_history_by_truth[truth_id]
            )
            for truth_id in self.truth_ids
        }

    def _assign_new_tracks(self, estimates, truths) -> None:
        active_tracks = set(estimates)
        for truth_id, track_id in tuple(self.active_track_id_by_truth.items()):
            if track_id not in active_tracks:
                del self.active_track_id_by_truth[truth_id]
        new_tracks = tuple(sorted(
            track_id for track_id in estimates
            if track_id not in self.assignment_by_track_id
        ))
        free_truths = tuple(
            truth_id for truth_id in self.truth_ids
            if truth_id not in self.active_track_id_by_truth
        )
        if not new_tracks or not free_truths:
            return
        assignment = _minimum_position_assignment(
            {track_id: estimates[track_id] for track_id in new_tracks},
            {truth_id: truths[truth_id] for truth_id in free_truths},
        )
        previously_assigned_truths = set(self.assignment_by_track_id.values())
        for track_id, truth_id in assignment.items():
            if truth_id in previously_assigned_truths:
                self.track_replacement_count += 1
            self.assignment_by_track_id[track_id] = truth_id
            self.active_track_id_by_truth[truth_id] = track_id


def _minimum_position_assignment(estimates, truths) -> dict[str, str]:
    estimate_items = tuple(sorted(estimates.items()))
    truth_items = tuple(sorted(truths.items()))
    if not estimate_items or not truth_items:
        return {}
    costs = np.asarray([
        [
            np.linalg.norm(estimate[:3] - truth[:3])
            for _, truth in truth_items
        ]
        for _, estimate in estimate_items
    ])
    rows, columns = linear_sum_assignment(costs)
    return {
        estimate_items[row][0]: truth_items[column][0]
        for row, column in zip(rows, columns)
    }


def _state(value, label):
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.shape != (6,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite six-state vector.")
    return result


def _rmse(values) -> float:
    array = np.asarray(tuple(values), dtype=float)
    return (
        float(np.sqrt(np.mean(np.square(array))))
        if array.size else float("nan")
    )
