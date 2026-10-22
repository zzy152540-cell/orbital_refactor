import pytest

from experiments.summarize_bilateral_baseline_matrix import (
    _p05_evidence,
    _p07_evidence,
    summarize_payloads,
)


def _payload(seed, position):
    arm = {
        "final_position_rmse_m": position,
        "final_maximum_position_error_m": position * 2.0,
        "final_velocity_rmse_mps": position / 10.0,
        "final_acceleration_rmse_mps2": position / 100.0,
        "worst_target_position_rmse_m": position * 1.5,
        "processing_mean_ms": 10.0,
        "processing_p95_ms": 12.0,
        "identity_switch_count": 0,
        "nearest_truth_disagreement_count": 0,
        "track_replacement_count": 0,
        "missing_truth_sample_count": 0,
        "consensus_worst_target_position_rmse_m": None,
        "consensus_identity_switch_count": 0,
        "consensus_nearest_truth_disagreement_count": 0,
        "consensus_track_replacement_count": 0,
        "consensus_missing_truth_sample_count": 0,
        "position_rmse_m_by_target": {"hostile-1": position},
        "consensus_position_rmse_m_by_target": {},
    }
    return {
        "scenario": {"seed": seed, "durationSeconds": 20.0},
        "arms": {"example": arm},
    }


def test_summary_reports_cross_seed_statistics_and_target_values():
    summary = summarize_payloads((_payload(0, 2.0), _payload(1, 4.0)))

    metric = summary["arms"]["example"]["metrics"][
        "final_position_rmse_m"
    ]
    target = summary["arms"]["example"]["positionRmseByTarget"][
        "hostile-1"
    ]
    assert summary["seeds"] == [0, 1]
    assert metric == {
        "count": 2, "mean": 3.0, "std": 1.0, "min": 2.0, "max": 4.0,
    }
    assert target["mean"] == 3.0
    assert summary["arms"]["example"]["consensusFullRunMetrics"][
        "position_rmse_m"
    ]["count"] == 0
    assert summary["arms"]["example"]["identityAudit"][
        "switchTotal"
    ] == 0


def test_summary_rejects_duplicate_seeds():
    with pytest.raises(ValueError, match="unique"):
        summarize_payloads((_payload(0, 2.0), _payload(0, 4.0)))


def test_p05_uses_exact_single_observer_sample_fingerprint():
    payloads = []
    for seed in (0, 1):
        phase = {
            "FULL_RUN": {
                "position_rmse_m": 10.0,
                "position_sample_count": 20,
            }
        }
        baseline = {
            "node_ids": ["observer-0"],
            "scored_sample_count": 20,
            "scored_sample_key_sha256": f"digest-{seed}",
            "scored_truth_ids": ["target-0", "target-1"],
            "phase_metrics": phase,
        }
        candidate = {
            **baseline,
            "phase_metrics": {"FULL_RUN": {
                "position_rmse_m": 9.0,
                "position_sample_count": 20,
            }},
        }
        payloads.append({
            "scenario": {"seed": seed},
            "localNodeBaselines": {
                "single_node_multimodal_no_cann": baseline,
                "single_node_multimodal_cann": candidate,
            },
            "matchedSingleObserverCannVsModality": {
                modality: {
                    "local_sample_count": 20,
                    "consensus_sample_count": 20,
                    "common_sample_count": 20,
                    "local_only_sample_count": 0,
                    "consensus_only_sample_count": 0,
                    "local_position_mean_error_m": 10.0,
                    "consensus_position_mean_error_m": 9.0,
                    "local_position_rmse_m": 10.0,
                    "consensus_position_rmse_m": 9.0,
                        "local_velocity_mean_error_mps": 1.0,
                        "consensus_velocity_mean_error_mps": 0.9,
                        "local_velocity_rmse_mps": 1.0,
                        "consensus_velocity_rmse_mps": 0.9,
                }
                for modality in ("radar", "infrared", "optical")
            },
            "matchedSingleObserverAblation": {
                name: {
                    "local_sample_count": 20,
                    "consensus_sample_count": 20,
                    "common_sample_count": 20,
                    "local_only_sample_count": 0,
                    "consensus_only_sample_count": 0,
                    "local_position_mean_error_m": 10.0,
                    "consensus_position_mean_error_m": 9.0,
                    "local_position_rmse_m": 10.0,
                    "consensus_position_rmse_m": 9.0,
                        "local_velocity_mean_error_mps": 1.0,
                        "consensus_velocity_mean_error_mps": 0.9,
                        "local_velocity_rmse_mps": 1.0,
                        "consensus_velocity_rmse_mps": 0.9,
                }
                for name in (
                    "federatedCannIncrement",
                    "jointFusionWithoutCann",
                    "jointCannIncrement",
                    "jointFusionWithCannCombined",
                )
            },
        })

    evidence = _p05_evidence(payloads)

    assert evidence["coverageComparable"]
    assert evidence["pooled"]["sampleCount"] == 40
    assert evidence["pooled"]["improvementPercent"] == pytest.approx(10.0)
    assert evidence["developmentGatePassed"]
    assert not evidence["auxiliaryComparisonCanReplacePrimary"]
    assert evidence["auxiliaryExactMatchedVsSingleModality"]["radar"][
        "positionRmse"
    ]["improvementPercent"] == pytest.approx(10.0)
    assert evidence["jointMeasurementCannAblation"][
        "jointCannIncrement"
    ]["positionRmse"]["improvementPercent"] == pytest.approx(10.0)


def test_p07_keeps_mean_error_and_rmse_candidates_separate():
    matched = {
        "local_sample_count": 100,
        "consensus_sample_count": 120,
        "common_sample_count": 100,
        "local_only_sample_count": 0,
        "consensus_only_sample_count": 20,
        "local_position_mean_error_m": 10.0,
        "consensus_position_mean_error_m": 9.0,
        "local_position_rmse_m": 12.0,
        "consensus_position_rmse_m": 11.4,
        "local_velocity_mean_error_mps": 1.0,
        "consensus_velocity_mean_error_mps": 0.9,
        "local_velocity_rmse_mps": 1.2,
        "consensus_velocity_rmse_mps": 1.14,
    }
    payload = {
        "scenario": {"seed": 0},
        "matchedIndependentConsensusByCann": {
            "withoutCann": matched,
            "incrementalCann": matched,
        },
        "matchedJointIndependentConsensus": matched,
        "arms": {
            "centralized_multimodal_no_cann": {
                "matched_local_central_metrics": matched,
            },
            "fixed_topology_multimodal_no_cann": {
                "matched_local_consensus_metrics": matched,
            },
            "fixed_topology_multimodal_cann": {
                "matched_local_consensus_metrics": matched,
            },
        },
    }

    evidence = _p07_evidence([payload])
    comparison = evidence["comparisons"]["withoutCann"]

    assert comparison["coverageComparable"]
    assert comparison["positionMeanError"]["improvementPercent"] == 10.0
    assert comparison["positionRmse"]["improvementPercent"] == pytest.approx(5.0)
    assert evidence["primaryMetricFreezeStatus"] == "FROZEN_POSITION_MEAN_ERROR"
    assert evidence["centralizedAllObservationDiagnosticUpperBound"][
        "coverageComparable"
    ]
    assert evidence["adjustedJointMeasurementAlgorithm"][
        "positionMeanError"
    ]["improvementPercent"] == 10.0
