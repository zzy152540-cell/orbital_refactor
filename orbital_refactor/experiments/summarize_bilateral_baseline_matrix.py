"""Aggregate repeated bilateral baseline-matrix JSON outputs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path

import numpy as np


SUMMARY_FIELDS = (
    "final_position_rmse_m",
    "final_maximum_position_error_m",
    "final_velocity_rmse_mps",
    "final_acceleration_rmse_mps2",
    "worst_target_position_rmse_m",
    "processing_mean_ms",
    "processing_p95_ms",
    "identity_switch_count",
    "nearest_truth_disagreement_count",
    "track_replacement_count",
    "missing_truth_sample_count",
    "consensus_worst_target_position_rmse_m",
    "consensus_identity_switch_count",
    "consensus_nearest_truth_disagreement_count",
    "consensus_track_replacement_count",
    "consensus_missing_truth_sample_count",
)
P05_MINIMUM_IMPROVEMENT_PERCENT = 8.0
P07_MINIMUM_IMPROVEMENT_PERCENT = 6.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.inputs
    ]
    summary = summarize_payloads(payloads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for arm_name, arm in summary["arms"].items():
        position = arm["metrics"]["final_position_rmse_m"]
        print(
            arm_name,
            f"positionRmseMean={position['mean']:.6f}",
            f"positionRmseStd={position['std']:.6f}",
            f"identitySwitchTotal={arm['identityAudit']['switchTotal']}",
        )
    print(f"wrote {args.output}")


def summarize_payloads(payloads) -> dict:
    values = tuple(payloads)
    if not values:
        raise ValueError("At least one baseline matrix payload is required.")
    arm_names = tuple(values[0]["arms"])
    expected = set(arm_names)
    if any(set(payload.get("arms", {})) != expected for payload in values):
        raise ValueError("Every payload must contain the same baseline arms.")
    seeds = [int(payload["scenario"]["seed"]) for payload in values]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Baseline matrix seeds must be unique.")
    arms = {}
    for arm_name in arm_names:
        runs = [payload["arms"][arm_name] for payload in values]
        metrics = {
            field: _statistics(run.get(field) for run in runs)
            for field in SUMMARY_FIELDS
        }
        arms[arm_name] = {
            "metrics": metrics,
            "fullRunMetrics": {
                field: _statistics(
                    run.get("phase_metrics", {})
                    .get("FULL_RUN", {})
                    .get(field)
                    for run in runs
                )
                for field in (
                    "position_rmse_m",
                    "velocity_rmse_mps",
                    "acceleration_rmse_mps2",
                    "position_sample_count",
                    "acceleration_sample_count",
                )
            },
            "meanLocalModalityWeight": {
                modality: _statistics(
                    run.get("mean_local_modality_weight", {}).get(modality)
                    for run in runs
                )
                for modality in sorted({
                    modality
                    for run in runs
                    for modality in run.get(
                        "mean_local_modality_weight", {}
                    )
                })
            },
            "consensusFullRunMetrics": {
                field: _statistics(
                    run.get("consensus_phase_metrics", {})
                    .get("FULL_RUN", {})
                    .get(field)
                    for run in runs
                )
                for field in (
                    "position_rmse_m",
                    "velocity_rmse_mps",
                    "acceleration_rmse_mps2",
                    "position_sample_count",
                    "acceleration_sample_count",
                )
            },
            "identityAudit": {
                "switchTotal": int(sum(
                    run["identity_switch_count"] for run in runs
                )),
                "replacementTotal": int(sum(
                    run["track_replacement_count"] for run in runs
                )),
                "missingTruthSampleTotal": int(sum(
                    run["missing_truth_sample_count"] for run in runs
                )),
                "consensusSwitchTotal": int(sum(
                    run["consensus_identity_switch_count"] for run in runs
                )),
            },
            "positionRmseByTarget": _target_statistics(
                run["position_rmse_m_by_target"] for run in runs
            ),
            "consensusPositionRmseByTarget": _target_statistics(
                run["consensus_position_rmse_m_by_target"] for run in runs
            ),
        }
    local_names = tuple(values[0].get("localNodeBaselines", {}))
    if any(
        set(payload.get("localNodeBaselines", {})) != set(local_names)
        for payload in values
    ):
        raise ValueError(
            "Every payload must contain the same local-node baselines."
        )
    local_node_baselines = {
        name: _summarize_local_node_runs([
            payload["localNodeBaselines"][name] for payload in values
        ])
        for name in local_names
    }
    return {
        "schemaVersion": "bilateral-baseline-matrix-summary.v3",
        "seedCount": len(values),
        "seeds": sorted(seeds),
        "durationSeconds": values[0]["scenario"]["durationSeconds"],
        "arms": arms,
        "localNodeBaselines": local_node_baselines,
        "fourMetricEvidence": {
            "P05": _p05_evidence(values),
            "P07": _p07_evidence(values),
        },
    }


def _statistics(values) -> dict:
    raw = tuple(value for value in values if value is not None)
    if not raw:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    array = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("Summary metrics must be finite or null.")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _target_statistics(mappings) -> dict:
    values = tuple(mappings)
    target_ids = sorted({
        str(target_id) for mapping in values for target_id in mapping
    })
    return {
        target_id: _statistics(
            mapping.get(target_id) for mapping in values
        )
        for target_id in target_ids
    }


def _summarize_local_node_runs(runs) -> dict:
    sample_total = sum(
        run["phase_metrics"]["FULL_RUN"]["position_sample_count"]
        for run in runs
    )
    missing_total = sum(run["missing_truth_sample_count"] for run in runs)
    target_statistics = _target_statistics(
        run["position_rmse_m_by_target"] for run in runs
    )
    return {
        "nodeCount": int(runs[0]["node_count"]),
        "nodeIds": list(runs[0]["node_ids"]),
        "scoredSampleCounts": [
            int(run.get("scored_sample_count", 0)) for run in runs
        ],
        "scoredSampleKeySha256": [
            run.get("scored_sample_key_sha256") for run in runs
        ],
        "scoredTruthIds": [
            list(run.get("scored_truth_ids", ())) for run in runs
        ],
        "fullRunMetrics": {
            field: _statistics(
                run["phase_metrics"]["FULL_RUN"].get(field) for run in runs
            )
            for field in (
                "position_mean_error_m",
                "position_rmse_m",
                "velocity_mean_error_mps",
                "velocity_rmse_mps",
                "acceleration_mean_error_mps2",
                "acceleration_rmse_mps2",
                "position_sample_count",
                "acceleration_sample_count",
            )
        },
        "identityAudit": {
            "switchTotal": int(sum(
                run["identity_switch_count"] for run in runs
            )),
            "replacementTotal": int(sum(
                run["track_replacement_count"] for run in runs
            )),
            "missingTruthSampleTotal": int(missing_total),
        },
        "coverage": {
            "coveredTruthCount": sum(
                metrics["count"] > 0 for metrics in target_statistics.values()
            ),
            "configuredTruthCount": len(target_statistics),
            "sampleCoverageFraction": (
                float(sample_total / (sample_total + missing_total))
                if sample_total + missing_total else 0.0
            ),
        },
        "worstTargetPositionRmse": _statistics(
            run["worst_target_position_rmse_m"] for run in runs
        ),
        "positionRmseByTarget": target_statistics,
    }


def _p05_evidence(payloads) -> dict:
    baseline_name = "single_node_optical_only"
    candidate_name = "single_node_multimodal_cann"
    required = {baseline_name, candidate_name}
    if any(
        not required.issubset(payload.get("localNodeBaselines", {}))
        for payload in payloads
    ):
        return {
            "status": "NOT_AVAILABLE",
            "reason": "Single-observer paired baselines are missing.",
        }
    coverage_by_seed = []
    per_seed = []
    for payload in payloads:
        baseline = payload["localNodeBaselines"][baseline_name]
        candidate = payload["localNodeBaselines"][candidate_name]
        comparable = _same_node_sample_coverage(baseline, candidate)
        coverage_by_seed.append(comparable)
        baseline_phase = baseline["phase_metrics"]["FULL_RUN"]
        candidate_phase = candidate["phase_metrics"]["FULL_RUN"]
        per_seed.append({
            "seed": int(payload["scenario"]["seed"]),
            "coverageComparable": comparable,
            "scoredSampleCount": int(
                baseline.get("scored_sample_count", 0)
            ),
            "scoredTruthCount": len(baseline.get("scored_truth_ids", ())),
            "positionRmseImprovementPercent": _improvement_percent(
                baseline_phase["position_rmse_m"],
                candidate_phase["position_rmse_m"],
            ),
        })
    pooled = _pooled_node_pair(
        payloads, baseline_name=baseline_name, candidate_name=candidate_name,
        field="position_rmse_m", count_field="position_sample_count",
    )
    coverage_comparable = all(coverage_by_seed)
    threshold_met = bool(
        pooled["improvementPercent"] is not None
        and pooled["improvementPercent"] >= P05_MINIMUM_IMPROVEMENT_PERCENT
    )
    matched_records = [
        payload.get("matchedSingleObserverJointCannVsModality", {}).get(
            "optical"
        )
        for payload in payloads
    ]
    matched_evidence = (
        None
        if any(record is None for record in matched_records)
        else _matched_pair_evidence(
            matched_records, seeds=[
                int(payload["scenario"]["seed"]) for payload in payloads
            ],
            minimum_improvement_percent=P05_MINIMUM_IMPROVEMENT_PERCENT,
        )
    )
    joint_records = [
        payload.get("matchedSingleObserverJointCann") for payload in payloads
    ]
    joint_candidate_evidence = (
        {
            "status": "NOT_AVAILABLE",
            "reason": "Joint same-epoch multimodal candidate is missing.",
        }
        if any(record is None for record in joint_records)
        else _matched_pair_evidence(
            joint_records, seeds=[
                int(payload["scenario"]["seed"]) for payload in payloads
            ],
            minimum_improvement_percent=P05_MINIMUM_IMPROVEMENT_PERCENT,
        )
    )
    ablation_names = (
        "federatedCannIncrement",
        "jointFusionWithoutCann",
        "jointCannIncrement",
        "jointFusionWithCannCombined",
    )
    joint_ablation = {}
    for name in ablation_names:
        records = [
            payload.get("matchedSingleObserverAblation", {}).get(name)
            for payload in payloads
        ]
        joint_ablation[name] = (
            {
                "status": "NOT_AVAILABLE",
                "reason": f"P05 ablation arm {name} is missing.",
            }
            if any(record is None for record in records)
            else _matched_pair_evidence(
                records,
                seeds=[
                    int(payload["scenario"]["seed"])
                    for payload in payloads
                ],
                minimum_improvement_percent=(
                    P05_MINIMUM_IMPROVEMENT_PERCENT
                ),
            )
        )
    modality_names = ("radar", "infrared", "optical")
    auxiliary_vs_single_modality = {}
    for modality in modality_names:
        records = [
            payload.get("matchedSingleObserverCannVsModality", {}).get(
                modality
            )
            for payload in payloads
        ]
        auxiliary_vs_single_modality[modality] = (
            {
                "status": "NOT_AVAILABLE",
                "reason": (
                    "Exact matched single-observer modality comparison is "
                    "missing."
                ),
            }
            if any(record is None for record in records)
            else _matched_pair_evidence(
                records, seeds=[
                    int(payload["scenario"]["seed"]) for payload in payloads
                ],
                minimum_improvement_percent=P05_MINIMUM_IMPROVEMENT_PERCENT,
            )
        )
    return {
        "metricScope": "ONE_OBSERVER_COMMON_NODE_TARGET_TIME_SAMPLES",
        "primaryMetric": "position_rmse_m",
        "minimumImprovementPercent": P05_MINIMUM_IMPROVEMENT_PERCENT,
        "baseline": baseline_name,
        "candidate": candidate_name,
        "coverageComparable": coverage_comparable,
        "thresholdMet": threshold_met,
        "developmentGatePassed": bool(coverage_comparable and threshold_met),
        "formalAcceptanceClaimed": False,
        "perSeed": per_seed,
        "pooled": pooled,
        "exactMatchedEvidence": matched_evidence,
        "experimentalJointMeasurementCandidate": joint_candidate_evidence,
        "jointMeasurementCannAblation": joint_ablation,
        "auxiliaryExactMatchedVsSingleModality": (
            auxiliary_vs_single_modality
        ),
        "auxiliaryComparisonCanReplacePrimary": False,
    }


def _p07_evidence(payloads) -> dict:
    candidates = {
        "fixed_topology_multimodal_no_cann": "withoutCann",
        "fixed_topology_multimodal_cann": "incrementalCann",
    }
    comparisons = {}
    for arm_name, label in candidates.items():
        matched = [
            payload.get("matchedIndependentConsensusByCann", {}).get(label)
            for payload in payloads
        ]
        if any(record is None for record in matched):
            comparisons[label] = {
                "status": "NOT_AVAILABLE",
                "reason": (
                    "Cross-arm independent-local versus cooperative-consensus "
                    "matched samples are missing."
                ),
            }
            continue
        comparisons[label] = _matched_pair_evidence(
            matched, seeds=[
                int(payload["scenario"]["seed"]) for payload in payloads
            ],
        )
    central_records = [
        payload.get("arms", {}).get(
            "centralized_multimodal_no_cann", {}
        ).get("matched_local_central_metrics")
        for payload in payloads
    ]
    centralized_upper_bound = (
        {
            "status": "NOT_AVAILABLE",
            "reason": "Matched local-central samples are missing.",
        }
        if any(record is None for record in central_records)
        else _matched_pair_evidence(
            central_records, seeds=[
                int(payload["scenario"]["seed"]) for payload in payloads
            ],
        )
    )
    joint_records = [
        payload.get("matchedJointIndependentConsensus")
        for payload in payloads
    ]
    joint_algorithm_comparison = (
        {
            "status": "NOT_AVAILABLE",
            "reason": (
                "Matched joint-update independent versus cooperative samples "
                "are missing."
            ),
        }
        if any(record is None for record in joint_records)
        else _matched_pair_evidence(
            joint_records, seeds=[
                int(payload["scenario"]["seed"]) for payload in payloads
            ],
        )
    )
    return {
        "metricScope": "COMMON_NODE_TARGET_TIME_SAMPLES",
        "minimumImprovementPercent": P07_MINIMUM_IMPROVEMENT_PERCENT,
        "primaryMetricFreezeStatus": (
            "FROZEN_POSITION_MEAN_ERROR"
        ),
        "formalAcceptanceClaimed": False,
        "comparisons": comparisons,
        "adjustedJointMeasurementAlgorithm": joint_algorithm_comparison,
        "centralizedAllObservationDiagnosticUpperBound": (
            centralized_upper_bound
        ),
    }


def _matched_pair_evidence(
    records, *, seeds=None,
    minimum_improvement_percent=P07_MINIMUM_IMPROVEMENT_PERCENT,
) -> dict:
    coverage_comparable = all(
        int(record["common_sample_count"]) > 0
        and int(record["common_sample_count"])
        == int(record["local_sample_count"])
        for record in records
    )
    mean_error = _pooled_matched_pair(
        records,
        local_field="local_position_mean_error_m",
        consensus_field="consensus_position_mean_error_m",
        squared=False,
    )
    rmse = _pooled_matched_pair(
        records,
        local_field="local_position_rmse_m",
        consensus_field="consensus_position_rmse_m",
        squared=True,
    )
    velocity_mean_error = _pooled_matched_pair(
        records,
        local_field="local_velocity_mean_error_mps",
        consensus_field="consensus_velocity_mean_error_mps",
        squared=False,
    )
    velocity_rmse = _pooled_matched_pair(
        records,
        local_field="local_velocity_rmse_mps",
        consensus_field="consensus_velocity_rmse_mps",
        squared=True,
    )
    acceleration_fields_available = all(
        "local_acceleration_rmse_mps2" in record
        and "consensus_acceleration_rmse_mps2" in record
        for record in records
    )
    acceleration_rmse = (
        _pooled_matched_pair(
            records,
            local_field="local_acceleration_rmse_mps2",
            consensus_field="consensus_acceleration_rmse_mps2",
            squared=True,
            count_field="common_acceleration_sample_count",
        )
        if acceleration_fields_available else {
            "commonSampleCount": 0,
            "local": None,
            "consensus": None,
            "improvementPercent": None,
        }
    )
    acceleration_coverage_comparable = bool(
        acceleration_fields_available
        and all(
            int(record.get("common_acceleration_sample_count", 0)) > 0
            and int(record["common_acceleration_sample_count"])
            == int(record.get("local_acceleration_sample_count", -1))
            for record in records
        )
    )
    result = {
        "coverageComparable": coverage_comparable,
        "accelerationCoverageComparable": (
            acceleration_coverage_comparable
        ),
        "localOnlySampleCount": int(sum(
            record["local_only_sample_count"] for record in records
        )),
        "consensusOnlySampleCount": int(sum(
            record["consensus_only_sample_count"] for record in records
        )),
        "positionMeanError": mean_error,
        "positionRmse": rmse,
        "velocityMeanError": velocity_mean_error,
        "velocityRmse": velocity_rmse,
        "accelerationRmse": acceleration_rmse,
        "thresholdMetByCandidatePrimaryMetric": {
            "position_mean_error_m": bool(
                mean_error["improvementPercent"] is not None
                and mean_error["improvementPercent"]
                >= minimum_improvement_percent
            ),
            "position_rmse_m": bool(
                rmse["improvementPercent"] is not None
                and rmse["improvementPercent"]
                >= minimum_improvement_percent
            ),
            "velocity_mean_error_mps": bool(
                velocity_mean_error["improvementPercent"] is not None
                and velocity_mean_error["improvementPercent"]
                >= minimum_improvement_percent
            ),
            "velocity_rmse_mps": bool(
                velocity_rmse["improvementPercent"] is not None
                and velocity_rmse["improvementPercent"]
                >= minimum_improvement_percent
            ),
            "acceleration_rmse_mps2": bool(
                acceleration_coverage_comparable
                and acceleration_rmse["improvementPercent"] is not None
                and acceleration_rmse["improvementPercent"]
                >= minimum_improvement_percent
            ),
        },
    }
    if seeds is not None:
        seed_values = tuple(int(seed) for seed in seeds)
        if len(seed_values) != len(records):
            raise ValueError("Matched records and seeds must have equal length.")
        per_seed = []
        mean_improvements = []
        rmse_improvements = []
        for seed, record in zip(seed_values, records, strict=True):
            mean_improvement = _improvement_percent(
                record["local_position_mean_error_m"],
                record["consensus_position_mean_error_m"],
            )
            rmse_improvement = _improvement_percent(
                record["local_position_rmse_m"],
                record["consensus_position_rmse_m"],
            )
            acceleration_improvement = _improvement_percent(
                record.get("local_acceleration_rmse_mps2"),
                record.get("consensus_acceleration_rmse_mps2"),
            )
            mean_improvements.append(mean_improvement)
            rmse_improvements.append(rmse_improvement)
            per_seed.append({
                "seed": seed,
                "commonSampleCount": int(record["common_sample_count"]),
                "positionMeanErrorImprovementPercent": mean_improvement,
                "positionRmseImprovementPercent": rmse_improvement,
                "commonAccelerationSampleCount": int(
                    record.get("common_acceleration_sample_count", 0)
                ),
                "accelerationRmseImprovementPercent": (
                    acceleration_improvement
                ),
                "meanErrorThresholdMet": bool(
                    mean_improvement is not None
                    and mean_improvement >= minimum_improvement_percent
                ),
                "rmseThresholdMet": bool(
                    rmse_improvement is not None
                    and rmse_improvement >= minimum_improvement_percent
                ),
                "accelerationRmseThresholdMet": bool(
                    int(record.get("common_acceleration_sample_count", 0)) > 0
                    and acceleration_improvement is not None
                    and acceleration_improvement
                    >= minimum_improvement_percent
                ),
            })
        result["perSeed"] = per_seed
        result["improvementAcrossSeeds"] = {
            "positionMeanErrorPercent": _improvement_statistics(
                mean_improvements
            ),
            "positionRmsePercent": _improvement_statistics(
                rmse_improvements
            ),
        }
    return result


def _improvement_statistics(values) -> dict:
    raw = np.asarray(
        [value for value in values if value is not None], dtype=float,
    )
    if raw.size == 0:
        return {
            "count": 0, "mean": None, "std": None,
            "min": None, "max": None, "meanConfidenceInterval95": None,
        }
    if not np.all(np.isfinite(raw)):
        raise ValueError("Improvement values must be finite or null.")
    if raw.size == 1:
        interval = [float(raw[0]), float(raw[0])]
    else:
        generator = np.random.default_rng(20261019)
        resampled_means = np.mean(
            generator.choice(raw, size=(20_000, raw.size), replace=True),
            axis=1,
        )
        interval = [
            float(np.percentile(resampled_means, 2.5)),
            float(np.percentile(resampled_means, 97.5)),
        ]
    return {
        "count": int(raw.size),
        "mean": float(np.mean(raw)),
        "std": float(np.std(raw)),
        "min": float(np.min(raw)),
        "max": float(np.max(raw)),
        "meanConfidenceInterval95": interval,
        "confidenceIntervalMethod": (
            "deterministic_seed_level_percentile_bootstrap_20000"
        ),
    }


def _same_node_sample_coverage(baseline, candidate) -> bool:
    baseline_digest = baseline.get("scored_sample_key_sha256")
    candidate_digest = candidate.get("scored_sample_key_sha256")
    return bool(
        baseline_digest
        and baseline_digest == candidate_digest
        and tuple(baseline.get("node_ids", ()))
        == tuple(candidate.get("node_ids", ()))
        and int(baseline.get("scored_sample_count", 0)) > 0
        and int(baseline.get("scored_sample_count", 0))
        == int(candidate.get("scored_sample_count", -1))
    )


def _pooled_node_pair(
    payloads, *, baseline_name, candidate_name, field, count_field,
) -> dict:
    baseline_sum = 0.0
    candidate_sum = 0.0
    sample_count = 0
    for payload in payloads:
        baseline = payload["localNodeBaselines"][baseline_name][
            "phase_metrics"
        ]["FULL_RUN"]
        candidate = payload["localNodeBaselines"][candidate_name][
            "phase_metrics"
        ]["FULL_RUN"]
        count = int(baseline[count_field])
        if count != int(candidate[count_field]):
            continue
        baseline_sum += float(baseline[field]) ** 2 * count
        candidate_sum += float(candidate[field]) ** 2 * count
        sample_count += count
    baseline_value = (baseline_sum / sample_count) ** 0.5 if sample_count else None
    candidate_value = (
        (candidate_sum / sample_count) ** 0.5 if sample_count else None
    )
    return {
        "sampleCount": sample_count,
        "baseline": baseline_value,
        "candidate": candidate_value,
        "improvementPercent": _improvement_percent(
            baseline_value, candidate_value
        ),
    }


def _pooled_matched_pair(
    records, *, local_field, consensus_field, squared,
    count_field="common_sample_count",
) -> dict:
    count = sum(int(record.get(count_field, 0)) for record in records)
    if not count:
        local_value = consensus_value = None
    elif squared:
        local_value = (
            sum(
                float(record[local_field]) ** 2
                * int(record[count_field])
                for record in records
            ) / count
        ) ** 0.5
        consensus_value = (
            sum(
                float(record[consensus_field]) ** 2
                * int(record[count_field])
                for record in records
            ) / count
        ) ** 0.5
    else:
        local_value = sum(
            float(record[local_field]) * int(record[count_field])
            for record in records
        ) / count
        consensus_value = sum(
            float(record[consensus_field])
            * int(record[count_field])
            for record in records
        ) / count
    return {
        "commonSampleCount": count,
        "local": local_value,
        "consensus": consensus_value,
        "improvementPercent": _improvement_percent(
            local_value, consensus_value
        ),
    }


def _improvement_percent(baseline, candidate):
    if baseline is None or candidate is None or float(baseline) <= 0.0:
        return None
    return float((float(baseline) - float(candidate)) / float(baseline) * 100.0)


if __name__ == "__main__":
    main()
