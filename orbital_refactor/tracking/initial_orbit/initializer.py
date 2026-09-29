from __future__ import annotations

import numpy as np

from tracking.data_contracts import TargetInitialState

from .batch_least_squares import refine_initial_state
from .data_contracts import InitialOrbitEstimate, IODObservation, IODStatus
from .multimodal_initializer import multimodal_state_guess


def initialize_target(
    observations,
    *,
    target_id: str,
    track_id: str | None = None,
) -> InitialOrbitEstimate:
    selected = tuple(
        item for item in observations
        if item.valid_flag and item.target_id == target_id
    )
    if len(selected) < 4:
        return _failure(target_id, track_id, IODStatus.INSUFFICIENT_OBSERVATIONS)
    try:
        epoch, guess, guess_diagnostics = multimodal_state_guess(selected)
        result, covariance, diagnostics = refine_initial_state(
            selected, epoch=epoch, initial_state_eci=guess,
        )
    except ValueError as exc:
        return _failure(
            target_id, track_id, IODStatus.INSUFFICIENT_OBSERVATIONS,
            message=str(exc),
        )
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        return _failure(
            target_id, track_id, IODStatus.NUMERICAL_FAILURE,
            message=str(exc),
        )
    diagnostics = {**guess_diagnostics, **diagnostics, "solverMessage": result.message}
    if diagnostics["jacobianRank"] < 6 or not np.isfinite(
        diagnostics["informationConditionNumber"]
    ):
        return InitialOrbitEstimate(
            target_id=target_id,
            track_id=track_id or f"track-{target_id}",
            epoch=epoch,
            state_eci=result.x,
            covariance_eci=covariance,
            status=IODStatus.POOR_GEOMETRY,
            method="MULTIMODAL_J2_BATCH_LS",
            converged=False,
            quality_score=0.0,
            diagnostics=diagnostics,
        )
    quality = 1.0 / (1.0 + diagnostics["residualRms"])
    return InitialOrbitEstimate(
        target_id=target_id,
        track_id=track_id or f"track-{target_id}",
        epoch=epoch,
        state_eci=result.x,
        covariance_eci=covariance,
        status=IODStatus.SUCCESS if result.success else IODStatus.NUMERICAL_FAILURE,
        method="MULTIMODAL_J2_BATCH_LS",
        converged=bool(result.success),
        quality_score=quality if result.success else 0.0,
        diagnostics=diagnostics,
    )


def target_initial_state_from_iod(value: InitialOrbitEstimate) -> TargetInitialState:
    if value.status is not IODStatus.SUCCESS or value.state_eci is None:
        raise ValueError("Only a successful IOD estimate can initialize a target track.")
    return TargetInitialState(
        target_id=value.target_id,
        track_id=value.track_id,
        timestamp=value.epoch,
        state_eci=value.state_eci,
        covariance_eci=value.covariance_eci,
    )


def _failure(target_id, track_id, status, *, message=None):
    return InitialOrbitEstimate(
        target_id=target_id,
        track_id=track_id or f"track-{target_id}",
        epoch=0.0,
        state_eci=None,
        covariance_eci=None,
        status=status,
        method="MULTIMODAL_J2_BATCH_LS",
        converged=False,
        quality_score=0.0,
        diagnostics={} if message is None else {"message": message},
    )
