"""Audit per-target error growth in one bilateral control-scene seed."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from experiments.bilateral_multitarget_frontend import (
    BilateralFrontendConfig,
    build_bilateral_raw_frontend,
)
from experiments.multitarget_identity_evaluation import IdentityAwareTrackEvaluator
from orbital_core.coordinates import (
    build_rtn_quaternion,
    quat_to_dcm_wxyz,
)
from tracking.association.frontend_adapter import (
    unlabeled_iod_observations_from_messages,
)
from tracking.association.online_tracker import OnlineMultiTargetTracker
from tracking.association.pipeline import MultiTargetAssociationPipeline
from tracking.multi_observer_bearing_filter import _tangent_basis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dropout-modality")
    parser.add_argument("--dropout-start", type=float)
    parser.add_argument("--dropout-end", type=float)
    parser.add_argument("--analysis-start", type=float)
    parser.add_argument("--analysis-end", type=float)
    parser.add_argument("--unified-multimodal-update", action="store_true")
    parser.add_argument("--unified-window-epochs", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit_seed(
        seed=args.seed,
        duration_seconds=args.duration,
        dropout_modality=args.dropout_modality,
        dropout_start_seconds=args.dropout_start,
        dropout_end_seconds=args.dropout_end,
        analysis_start_seconds=args.analysis_start,
        analysis_end_seconds=args.analysis_end,
        unified_multimodal_update=args.unified_multimodal_update,
        unified_window_epochs=args.unified_window_epochs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.output}")


def audit_seed(
    *, seed: int, duration_seconds: float,
    dropout_modality: str | None = None,
    dropout_start_seconds: float | None = None,
    dropout_end_seconds: float | None = None,
    analysis_start_seconds: float | None = None,
    analysis_end_seconds: float | None = None,
    unified_multimodal_update: bool = False,
    unified_window_epochs: int = 1,
) -> dict:
    frontend = build_bilateral_raw_frontend(BilateralFrontendConfig(
        duration_seconds=duration_seconds,
        sample_interval_seconds=1.0,
        random_seed=seed,
        dropout_modality=dropout_modality,
        dropout_start_seconds=dropout_start_seconds,
        dropout_end_seconds=dropout_end_seconds,
    ))
    config = frontend.config
    tracker = OnlineMultiTargetTracker(
        scene_id=frontend.truth.manifest.scene_id,
        association_pipeline=MultiTargetAssociationPipeline(
            gate_squared_mahalanobis=config.association_gate_squared_mahalanobis,
            minimum_cost_margin=config.association_minimum_cost_margin,
            prediction_position_sigma_floor_m=(
                config.association_position_sigma_floor_m
            ),
            prediction_velocity_sigma_floor_mps=(
                config.association_velocity_sigma_floor_mps
            ),
        ),
        max_coast_epochs=config.tracking_max_coast_epochs,
        max_lost_epochs=config.tracking_max_lost_epochs,
        maximum_track_count=config.maximum_track_count,
        preserve_angular_radial_nullspace=(
            config.preserve_angular_radial_nullspace
        ),
        enable_multi_observer_bearing_update=(
            config.use_multi_observer_bearing_update
        ),
        enable_unified_multimodal_update=unified_multimodal_update,
        unified_multimodal_window_epochs=unified_window_epochs,
    )
    truth_ids = frontend.truth.manifest.hostile_target_ids
    evaluator = IdentityAwareTrackEvaluator(truth_ids)
    history = {target_id: [] for target_id in truth_ids}
    initial = {}
    matched_counts = defaultdict(Counter)
    unobserved_epoch_counts = Counter()
    target_histories = frontend.truth.hostile_truth_state_history_by_target

    for epoch_index, timestamp_value in enumerate(frontend.truth.timestamps):
        timestamp = float(timestamp_value)
        observer_states = {
            observer_id: frontend.observer_state_by_epoch[(observer_id, timestamp)]
            for observer_id in frontend.truth.manifest.friendly_observer_ids
        }
        observations = unlabeled_iod_observations_from_messages(
            frontend.messages_by_timestamp[timestamp],
            observer_state_by_epoch=frontend.observer_state_by_epoch,
        )
        bearing_geometry = _bearing_geometry_by_truth(
            observations,
            truth_target_by_detection_group=(
                frontend.truth_target_by_detection_group
            ),
            truth_state_by_id={
                target_id: states[epoch_index]
                for target_id, states in target_histories.items()
            },
        )
        update = tracker.step(
            timestamp=timestamp,
            observations=observations,
            observer_states_eci=observer_states,
            q_eci2pri_by_observer={
                observer_id: build_rtn_quaternion(state)
                for observer_id, state in observer_states.items()
            },
        )
        truth_by_id = {
            target_id: states[epoch_index]
            for target_id, states in target_histories.items()
        }
        tracks_by_track_id = {
            track.estimate.track_id: track
            for track in update.tracks_by_target.values()
            if track.lifecycle.value != "TERMINATED"
        }
        epoch_result = evaluator.update(
            timestamp=timestamp,
            state_by_track_id={
                track_id: track.estimate.state_eci
                for track_id, track in tracks_by_track_id.items()
            },
            truth_state_by_id=truth_by_id,
        )
        target_id_by_track_id = {
            track_id: track.target_id
            for track_id, track in tracks_by_track_id.items()
        }
        truth_by_internal_target = {
            target_id_by_track_id[track_id]: truth_id
            for track_id, truth_id in evaluator.assignment_by_track_id.items()
            if track_id in target_id_by_track_id
        }
        epoch_match_counts = defaultdict(Counter)
        for match in update.association.matches:
            truth_id = truth_by_internal_target.get(match.target_id)
            if truth_id is None:
                continue
            item = observations[match.observation_index]
            modality = str(
                item.metadata.get("sourceModality", item.modality)
            ).upper()
            epoch_match_counts[truth_id][modality] += 1
            matched_counts[truth_id][modality] += 1
        for internal_target_id in update.association.unobserved_target_ids:
            truth_id = truth_by_internal_target.get(internal_target_id)
            if truth_id is not None:
                unobserved_epoch_counts[truth_id] += 1

        for truth_id, sample in epoch_result.samples_by_truth.items():
            track = tracks_by_track_id[sample.track_id]
            covariance = np.asarray(track.estimate.covariance_eci, dtype=float)
            rtn_diagnostics = _rtn_error_diagnostics(
                estimate_state=track.estimate.state_eci,
                truth_state=truth_by_id[truth_id],
                covariance_eci=covariance,
            )
            internal_target_id = track.target_id
            local_reports = [
                report for report in update.local_reports
                if report.target_id == internal_target_id and report.valid_flag
            ]
            visible_count = len(
                frontend.visible_observers_by_target_and_timestamp[
                    (truth_id, timestamp)
                ]
            )
            history[truth_id].append({
                "timestamp": timestamp,
                "positionErrorM": sample.position_error_m,
                "velocityErrorMps": sample.velocity_error_mps,
                "positionCovarianceTraceM2": float(np.trace(covariance[:3, :3])),
                "velocityCovarianceTraceM2ps2": float(np.trace(covariance[3:, 3:])),
                **rtn_diagnostics,
                "visibleObserverCount": visible_count,
                "bearingGeometry": bearing_geometry.get(truth_id),
                "matchedMeasurementCountByModality": dict(
                    sorted(epoch_match_counts[truth_id].items())
                ),
                "globalNodeWeights": dict(sorted(
                    getattr(track.estimate, "node_weights", {}).items()
                )),
                "localReports": [
                    {
                        "observerId": report.observer_id,
                        "positionErrorM": float(np.linalg.norm(
                            report.state_eci[:3] - truth_by_id[truth_id][:3]
                        )),
                        "velocityErrorMps": float(np.linalg.norm(
                            report.state_eci[3:] - truth_by_id[truth_id][3:]
                        )),
                        "positionCovarianceTraceM2": float(np.trace(
                            report.covariance_eci[:3, :3]
                        )),
                        "modalityWeights": dict(sorted(
                            report.modality_weights.items()
                        )),
                    }
                    for report in local_reports
                ],
                "lifecycle": track.lifecycle.value,
            })

        for state in update.initialized_states_by_target.values():
            truth_id = evaluator.assignment_by_track_id.get(state.track_id)
            if truth_id is None or truth_id in initial:
                continue
            truth = truth_by_id[truth_id]
            initial[truth_id] = {
                "timestamp": timestamp,
                "positionErrorM": float(np.linalg.norm(
                    state.state_eci[:3] - truth[:3]
                )),
                "velocityErrorMps": float(np.linalg.norm(
                    state.state_eci[3:] - truth[3:]
                )),
                "positionCovarianceTraceM2": float(np.trace(
                    state.covariance_eci[:3, :3]
                )),
                "velocityCovarianceTraceM2ps2": float(np.trace(
                    state.covariance_eci[3:, 3:]
                )),
                "iodDiagnostics": dict(
                    update.iod_diagnostics_by_target.get(state.target_id, {})
                ),
            }

    summaries = {}
    for truth_id, samples in history.items():
        position = np.asarray(
            [sample["positionErrorM"] for sample in samples], dtype=float
        )
        velocity = np.asarray(
            [sample["velocityErrorMps"] for sample in samples], dtype=float
        )
        summaries[truth_id] = {
            "positionRmseM": _rmse(position),
            "maximumPositionErrorM": _maximum(position),
            "velocityRmseMps": _rmse(velocity),
            "firstEpochAbove1km": next((
                sample["timestamp"] for sample in samples
                if sample["positionErrorM"] > 1_000.0
            ), None),
            "initialOrbit": initial.get(truth_id),
            "matchedMeasurementCountByModality": dict(
                sorted(matched_counts[truth_id].items())
            ),
            "unobservedEpochCount": int(unobserved_epoch_counts[truth_id]),
        }
    phase_diagnostics = {
        "FULL": _phase_rtn_diagnostics(history),
    }
    phase_start = (
        analysis_start_seconds
        if analysis_start_seconds is not None else dropout_start_seconds
    )
    phase_end = (
        analysis_end_seconds
        if analysis_end_seconds is not None else dropout_end_seconds
    )
    if phase_start is not None and phase_end is not None:
        phase_diagnostics["ANALYSIS_WINDOW"] = _phase_rtn_diagnostics(
            history,
            start=float(phase_start),
            end=float(phase_end),
        )
        phase_diagnostics["ANALYSIS_WINDOW_GEOMETRY"] = (
            _phase_geometry_diagnostics(
                history,
                start=float(phase_start),
                end=float(phase_end),
            )
        )
    return {
        "schemaVersion": "bilateral-seed-outlier-audit.v1",
        "seed": int(seed),
        "durationSeconds": float(duration_seconds),
        "dropoutModality": dropout_modality,
        "dropoutStartSeconds": dropout_start_seconds,
        "dropoutEndSeconds": dropout_end_seconds,
        "analysisStartSeconds": phase_start,
        "analysisEndSeconds": phase_end,
        "unifiedMultimodalUpdate": bool(unified_multimodal_update),
        "unifiedWindowEpochs": int(unified_window_epochs),
        "identitySwitchCount": evaluator.identity_switch_count,
        "nearestTruthDisagreementCount": (
            evaluator.nearest_truth_disagreement_count
        ),
        "targetSummaries": summaries,
        "phaseDiagnostics": phase_diagnostics,
        "historyByTarget": history,
    }


def _rtn_error_diagnostics(*, estimate_state, truth_state, covariance_eci):
    estimate = np.asarray(estimate_state, dtype=float).reshape(6)
    truth = np.asarray(truth_state, dtype=float).reshape(6)
    covariance = np.asarray(covariance_eci, dtype=float).reshape(6, 6)
    eci_to_rtn = quat_to_dcm_wxyz(build_rtn_quaternion(truth)).T
    transform = np.zeros((6, 6), dtype=float)
    transform[:3, :3] = eci_to_rtn
    transform[3:, 3:] = eci_to_rtn
    error_rtn = transform @ (estimate - truth)
    covariance_rtn = transform @ covariance @ transform.T
    covariance_rtn = 0.5 * (covariance_rtn + covariance_rtn.T)
    position_covariance = covariance_rtn[:3, :3]
    position_variances = np.maximum(
        np.diag(position_covariance), np.finfo(float).eps,
    )
    position_error = error_rtn[:3]
    return {
        "positionErrorRtnM": position_error.tolist(),
        "velocityErrorRtnMps": error_rtn[3:].tolist(),
        "positionCovarianceDiagonalRtnM2": position_variances.tolist(),
        "positionNormalizedSquaredErrorRtn": (
            np.square(position_error) / position_variances
        ).tolist(),
        "positionNees": float(
            position_error @ np.linalg.pinv(position_covariance) @ position_error
        ),
    }


def _phase_rtn_diagnostics(history, *, start=None, end=None):
    samples = [
        sample
        for target_samples in history.values()
        for sample in target_samples
        if (start is None or sample["timestamp"] >= start)
        and (end is None or sample["timestamp"] <= end)
    ]
    if not samples:
        return {"sampleCount": 0}
    position_errors = np.asarray(
        [sample["positionErrorRtnM"] for sample in samples], dtype=float,
    )
    velocity_errors = np.asarray(
        [sample["velocityErrorRtnMps"] for sample in samples], dtype=float,
    )
    position_variances = np.asarray(
        [sample["positionCovarianceDiagonalRtnM2"] for sample in samples],
        dtype=float,
    )
    normalized_squared = np.asarray(
        [sample["positionNormalizedSquaredErrorRtn"] for sample in samples],
        dtype=float,
    )
    return {
        "sampleCount": len(samples),
        "positionRmseRtnM": np.sqrt(
            np.mean(np.square(position_errors), axis=0)
        ).tolist(),
        "velocityRmseRtnMps": np.sqrt(
            np.mean(np.square(velocity_errors), axis=0)
        ).tolist(),
        "meanPositionSigmaRtnM": np.mean(
            np.sqrt(position_variances), axis=0
        ).tolist(),
        "meanPositionNormalizedSquaredErrorRtn": np.mean(
            normalized_squared, axis=0
        ).tolist(),
        "meanPositionNees": float(np.mean([
            sample["positionNees"] for sample in samples
        ])),
    }


def _phase_geometry_diagnostics(history, *, start=None, end=None):
    samples = [
        sample
        for target_samples in history.values()
        for sample in target_samples
        if (start is None or sample["timestamp"] >= start)
        and (end is None or sample["timestamp"] <= end)
        and sample.get("bearingGeometry") is not None
    ]
    if not samples:
        return {"sampleCount": 0}
    geometries = [sample["bearingGeometry"] for sample in samples]
    condition = np.asarray([
        item["informationConditionNumber"] for item in geometries
        if item["informationConditionNumber"] is not None
    ], dtype=float)
    weakest = np.asarray([
        item["weakestInformationEigenvaluePerM2"] for item in geometries
    ], dtype=float)
    radial_alignment = np.asarray([
        item["weakestDirectionAbsoluteRtn"][0] for item in geometries
    ], dtype=float)
    position_error = np.asarray([
        sample["positionErrorM"] for sample in samples
    ], dtype=float)
    log_weakest = np.log10(np.maximum(weakest, np.finfo(float).tiny))
    return {
        "sampleCount": len(samples),
        "meanObserverCount": float(np.mean([
            item["observerCount"] for item in geometries
        ])),
        "minimumWeakestInformationEigenvaluePerM2": float(np.min(weakest)),
        "meanWeakestInformationEigenvaluePerM2": float(np.mean(weakest)),
        "meanInformationConditionNumber": (
            float(np.mean(condition)) if condition.size else None
        ),
        "maximumInformationConditionNumber": (
            float(np.max(condition)) if condition.size else None
        ),
        "meanWeakestDirectionAbsoluteRtn": np.mean([
            item["weakestDirectionAbsoluteRtn"] for item in geometries
        ], axis=0).tolist(),
        "meanWeakestDirectionRadialAlignment": float(
            np.mean(radial_alignment)
        ),
        "positionErrorVsLogWeakestInformationCorrelation": (
            float(np.corrcoef(position_error, log_weakest)[0, 1])
            if len(samples) > 1 else None
        ),
    }


def _bearing_geometry_by_truth(
    observations,
    *,
    truth_target_by_detection_group,
    truth_state_by_id,
):
    """Compute offline LOS information geometry without aiding the tracker."""

    grouped = defaultdict(list)
    for item in observations:
        if item.modality != "LOS" or not item.valid_flag:
            continue
        group_id = item.detection_group_id
        truth_id = truth_target_by_detection_group.get(group_id)
        if truth_id is not None:
            grouped[truth_id].append(item)
    result = {}
    for truth_id, items in grouped.items():
        truth_position = np.asarray(
            truth_state_by_id[truth_id], dtype=float,
        ).reshape(6)[:3]
        information = np.zeros((3, 3), dtype=float)
        observer_ids = set()
        modality_counts = Counter()
        for item in items:
            relative = truth_position - item.observer_state_eci[:3]
            distance = float(np.linalg.norm(relative))
            if distance <= 0.0:
                continue
            line = relative / distance
            tangent = np.vstack(_tangent_basis(line))
            jacobian = (
                tangent @ (np.eye(3) - np.outer(line, line)) / distance
            )
            line_covariance = item.metadata.get("lineOfSightCovarianceEci")
            if line_covariance is None:
                tangent_covariance = np.asarray(item.covariance, dtype=float)
            else:
                tangent_covariance = (
                    tangent
                    @ np.asarray(line_covariance, dtype=float).reshape(3, 3)
                    @ tangent.T
                )
            tangent_covariance = 0.5 * (
                tangent_covariance + tangent_covariance.T
            )
            information += (
                jacobian.T @ np.linalg.pinv(tangent_covariance) @ jacobian
            )
            observer_ids.add(item.observer_id)
            modality = str(item.metadata.get("sourceModality", "LOS")).upper()
            modality_counts[modality] += 1
        information = 0.5 * (information + information.T)
        raw_eigenvalues, eigenvectors = np.linalg.eigh(information)
        eigenvalues = np.maximum(raw_eigenvalues, 0.0)
        largest = float(eigenvalues[-1]) if eigenvalues.size else 0.0
        threshold = max(largest * 1e-10, np.finfo(float).eps)
        positive = eigenvalues[eigenvalues > threshold]
        rank = int(positive.size)
        condition = (
            float(positive[-1] / positive[0]) if rank == 3 else None
        )
        predicted_sigma = (
            float(np.sqrt(np.trace(np.linalg.pinv(information))))
            if rank == 3 else None
        )
        weakest_direction_eci = eigenvectors[:, 0]
        eci_to_rtn = quat_to_dcm_wxyz(
            build_rtn_quaternion(truth_state_by_id[truth_id])
        ).T
        weakest_direction_rtn = eci_to_rtn @ weakest_direction_eci
        result[truth_id] = {
            "observerCount": len(observer_ids),
            "measurementCount": len(items),
            "measurementCountByModality": dict(sorted(modality_counts.items())),
            "informationRank": rank,
            "informationEigenvaluesPerM2": eigenvalues.tolist(),
            "weakestInformationEigenvaluePerM2": (
                float(eigenvalues[0]) if eigenvalues.size else 0.0
            ),
            "informationConditionNumber": condition,
            "measurementOnlyPositionSigmaM": predicted_sigma,
            "weakestDirectionEci": weakest_direction_eci.tolist(),
            "weakestDirectionAbsoluteRtn": np.abs(
                weakest_direction_rtn
            ).tolist(),
        }
    return result


def _rmse(values: np.ndarray) -> float | None:
    return (
        float(np.sqrt(np.mean(np.square(values)))) if values.size else None
    )


def _maximum(values: np.ndarray) -> float | None:
    return float(np.max(values)) if values.size else None


if __name__ == "__main__":
    main()
