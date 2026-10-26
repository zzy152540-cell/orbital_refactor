"""Aggregate bilateral modality-dropout and node-failure prescans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.summarize_bilateral_baseline_matrix import _statistics


ERROR_FIELDS = (
    "position_rmse_m",
    "velocity_rmse_mps",
    "acceleration_rmse_mps2",
)
NODE_FAILURE_ERROR_FIELDS = (
    "position_mean_error_m",
    "velocity_mean_error_mps",
    "acceleration_mean_error_mps2",
)
ARMS = ("withoutCann", "incrementalCann")
DROPOUT_MAX_INCREASE_PERCENT = 15.0
NODE_FAILURE_MAX_INCREASE_PERCENT = 12.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout-inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--failure-inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dropouts = [_load(path) for path in args.dropout_inputs]
    failures = [_load(path) for path in args.failure_inputs]
    summary = summarize_robustness(dropouts, failures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.output}")


def summarize_robustness(dropout_payloads, failure_payloads) -> dict:
    dropouts = tuple(dropout_payloads)
    failures = tuple(failure_payloads)
    if not dropouts or not failures:
        raise ValueError("Both dropout and failure payloads are required.")
    dropout_seeds = _unique_seeds(dropouts)
    failure_seeds = _unique_seeds(failures)
    modalities = tuple(dropouts[0]["resultsByMissingModality"])
    if any(
        set(payload["resultsByMissingModality"]) != set(modalities)
        for payload in dropouts
    ):
        raise ValueError("Dropout payloads must contain the same modalities.")
    dropout_summary = {
        modality: _dropout_modality_summary(dropouts, modality)
        for modality in modalities
    }
    failure_arms = tuple(
        arm for arm in ARMS
        if all(arm in payload["threeObserverFailure"] for payload in failures)
    )
    failure_summary = {
        "acceptanceMetricScope": "SURVIVING_NODE_CONSENSUS_ALL_TARGET_SAMPLES",
        "arms": {
            arm: _failure_arm_summary(failures, arm) for arm in failure_arms
        },
        "targetAggregateFailureVsControlPercent": {
            arm: _failure_percent_summary(failures, arm, "targetAggregate")
            for arm in failure_arms
        },
        "consensusFailureVsControlPercent": {
            arm: _failure_percent_summary(failures, arm, "consensus")
            for arm in failure_arms
        },
        "independentNodeFailureVsControlPercent": {
            arm: _failure_percent_summary(failures, arm, "independentNode")
            for arm in failure_arms
        },
        "pairedCannDeltaPercent": ({
            field: _statistics(
                _percent_change(
                    payload["threeObserverFailure"]["incrementalCann"]
                    ["consensus_phase_metrics"]["DROPOUT"][field],
                    payload["threeObserverFailure"]["withoutCann"]
                    ["consensus_phase_metrics"]["DROPOUT"][field],
                )
                for payload in failures
            )
            for field in NODE_FAILURE_ERROR_FIELDS
        } if set(ARMS).issubset(failure_arms) else {}),
    }
    for modality, record in dropout_summary.items():
        record["acceptanceMetricScope"] = (
            "POOLED_ALL_SEEDS_ALL_TARGET_TIME_SAMPLES"
        )
        record["acceptance"] = {
            arm: _acceptance_result(
                percent_statistics={
                    field: {
                        "mean": record["pooledDropoutVsControl"][arm][field]
                        ["increasePercent"]
                    }
                    for field in ERROR_FIELDS
                },
                threshold=DROPOUT_MAX_INCREASE_PERCENT,
                coverage_comparable=_dropout_coverage_comparable(
                    dropouts, modality, arm,
                ),
                primary_metric="position_rmse_m",
            )
            for arm in record["arms"]
        }
    failure_summary["acceptance"] = {
        arm: _acceptance_result(
            percent_statistics=(
                failure_summary["consensusFailureVsControlPercent"][arm]
            ),
            threshold=NODE_FAILURE_MAX_INCREASE_PERCENT,
            coverage_comparable=_failure_consensus_coverage_comparable(
                failures, arm
            ),
            primary_metric="position_mean_error_m",
        )
        for arm in failure_arms
    }
    return {
        "schemaVersion": "bilateral-robustness-summary.v3",
        "dropoutSeeds": dropout_seeds,
        "nodeFailureSeeds": failure_seeds,
        "modalityDropout": dropout_summary,
        "threeObserverFailure": failure_summary,
    }


def _acceptance_result(
    *, percent_statistics, threshold, coverage_comparable, primary_metric,
):
    threshold_by_metric = {
        field: (
            statistics["mean"] is not None
            and statistics["mean"] <= threshold
        )
        for field, statistics in percent_statistics.items()
    }
    return {
        "maximumAllowedMeanIncreasePercent": threshold,
        "primaryMetric": primary_metric,
        "coverageComparable": coverage_comparable,
        "thresholdMetByMetric": threshold_by_metric,
        "primaryThresholdMet": threshold_by_metric[primary_metric],
        "passed": bool(
            coverage_comparable and threshold_by_metric[primary_metric]
        ),
        "formalAcceptanceClaimed": False,
    }


def _dropout_coverage_comparable(payloads, modality, arm):
    for payload in payloads:
        expected = int(payload.get("scenario", {}).get("hostileTargetCount", 15))
        affected = payload["resultsByMissingModality"][modality][arm]
        control = payload.get("noDropoutControl", {}).get(arm)
        if control is None or not _dropout_pair_has_comparable_coverage(
            affected, control, expected,
        ):
            return False
    return True


def _dropout_pair_has_comparable_coverage(affected, control, expected):
    affected_phase = affected.get("phase_metrics", {}).get("DROPOUT", {})
    control_phase = control.get("phase_metrics", {}).get("DROPOUT", {})
    affected_targets = affected.get("position_rmse_m_by_target", {})
    control_targets = control.get("position_rmse_m_by_target", {})
    affected_count = int(affected_phase.get("position_sample_count", 0))
    control_count = int(control_phase.get("position_sample_count", 0))
    return bool(
        int(affected.get("final_active_track_count", -1)) == expected
        and int(control.get("final_active_track_count", -1)) == expected
        and len(affected_targets) == expected
        and len(control_targets) == expected
        and affected_count > 0
        and affected_count == control_count
    )


def _failure_consensus_coverage_comparable(payloads, arm):
    for payload in payloads:
        expected = int(payload.get("scenario", {}).get("hostileTargetCount", 15))
        affected = payload["threeObserverFailure"][arm]
        control = payload.get("noFailureControl", {}).get(arm)
        if not _arm_has_consensus_target_coverage(affected, expected):
            return False
        if control is None or not _arm_has_consensus_target_coverage(
            control, expected
        ):
            return False
    return True


def _arm_has_consensus_target_coverage(arm, expected):
    per_target = arm.get("consensus_position_rmse_m_by_target", {})
    phase = arm.get("consensus_phase_metrics", {}).get("DROPOUT", {})
    return bool(
        len(per_target) == expected
        and int(phase.get("position_sample_count", 0)) > 0
    )


def _dropout_modality_summary(payloads, modality):
    records = [payload["resultsByMissingModality"][modality] for payload in payloads]
    arms = tuple(
        arm for arm in ARMS if all(arm in record for record in records)
    )
    return {
        "arms": {
            arm: {
                "dropoutWindow": _phase_statistics(
                    (record[arm] for record in records), "DROPOUT"
                ),
                "recovery": _phase_statistics(
                    (record[arm] for record in records), "RECOVERY"
                ),
                "finalActiveTrackCount": _statistics(
                    record[arm]["final_active_track_count"] for record in records
                ),
                "retiredTrackCount": _statistics(
                    record[arm]["retired_track_count"] for record in records
                ),
                "identitySwitchCount": _statistics(
                    record[arm]["identity_switch_count"] for record in records
                ),
                "processingMeanMs": _statistics(
                    record[arm]["processing_mean_ms"] for record in records
                ),
            }
            for arm in arms
        },
        "dropoutVsControlPercent": {
            arm: {
                field: _statistics(
                    record["dropoutVsControlPercent"][arm][field]
                    for record in records
                )
                for field in ERROR_FIELDS
            }
            for arm in arms
        },
        "pooledDropoutVsControl": {
            arm: {
                field: _pooled_dropout_metric(
                    payloads, modality=modality, arm=arm, field=field,
                )
                for field in ERROR_FIELDS
            }
            for arm in arms
        },
        "pairedCannDeltaPercent": ({
            field: _statistics(
                record["pairedCannDeltaPercent"][field] for record in records
            )
            for field in ERROR_FIELDS
        } if all("pairedCannDeltaPercent" in record for record in records)
        else {}),
    }


def _pooled_dropout_metric(payloads, *, modality, arm, field):
    values = tuple(payloads)
    result = _pooled_dropout_metric_core(
        values, modality=modality, arm=arm, field=field,
    )
    if values:
        rng = np.random.default_rng(20261015)
        bootstrap = []
        for _ in range(5_000):
            indices = rng.integers(0, len(values), size=len(values))
            sample = tuple(values[int(index)] for index in indices)
            bootstrap.append(_pooled_dropout_metric_core(
                sample, modality=modality, arm=arm, field=field,
            )["increasePercent"])
        finite = np.asarray([
            value for value in bootstrap if value is not None
        ], dtype=float)
        result["pairedSeedBootstrap95Percent"] = (
            np.percentile(finite, [2.5, 97.5]).tolist()
            if finite.size else None
        )
    else:
        result["pairedSeedBootstrap95Percent"] = None
    return result


def _pooled_dropout_metric_core(payloads, *, modality, arm, field):
    affected_squared_sum = 0.0
    affected_count = 0
    control_squared_sum = 0.0
    control_count = 0
    for payload in payloads:
        affected = payload["resultsByMissingModality"][modality][arm][
            "phase_metrics"
        ]["DROPOUT"]
        control = payload["noDropoutControl"][arm]["phase_metrics"][
            "DROPOUT"
        ]
        affected_samples = _phase_sample_count(affected, field)
        control_samples = _phase_sample_count(control, field)
        affected_squared_sum += float(affected[field]) ** 2 * affected_samples
        control_squared_sum += float(control[field]) ** 2 * control_samples
        affected_count += affected_samples
        control_count += control_samples
    affected_rmse = (
        (affected_squared_sum / affected_count) ** 0.5
        if affected_count else None
    )
    control_rmse = (
        (control_squared_sum / control_count) ** 0.5
        if control_count else None
    )
    return {
        "affectedRmse": affected_rmse,
        "affectedSampleCount": affected_count,
        "controlRmse": control_rmse,
        "controlSampleCount": control_count,
        "increasePercent": _percent_change(affected_rmse, control_rmse),
    }


def _phase_sample_count(phase, field):
    if field == "acceleration_rmse_mps2":
        return int(phase.get(
            "acceleration_sample_count",
            phase.get("position_sample_count", 0),
        ))
    return int(phase.get("position_sample_count", 0))


def _failure_percent_summary(payloads, arm, scope):
    values = tuple(payloads)
    return {
        field: _statistics_with_paired_seed_bootstrap(
            (
                _percent_change(
                _failure_metric(payload, "threeObserverFailure", arm, scope, field),
                _failure_metric(payload, "noFailureControl", arm, scope, field),
                )
                for payload in values
            ),
            random_seed=20261016,
        )
        for field in NODE_FAILURE_ERROR_FIELDS
    }


def _statistics_with_paired_seed_bootstrap(values, *, random_seed):
    finite = np.asarray([
        float(value) for value in values if value is not None and np.isfinite(value)
    ], dtype=float)
    result = _statistics(finite)
    if finite.size == 0:
        result["pairedSeedBootstrapMean95Percent"] = None
        return result
    rng = np.random.default_rng(random_seed)
    bootstrap_means = np.mean(
        rng.choice(finite, size=(5_000, finite.size), replace=True), axis=1,
    )
    result["pairedSeedBootstrapMean95Percent"] = np.percentile(
        bootstrap_means, [2.5, 97.5]
    ).tolist()
    return result


def _failure_metric(payload, group, arm, scope, field):
    run = payload[group][arm]
    if scope == "targetAggregate":
        phase_metrics = run["phase_metrics"]
    elif scope == "consensus":
        phase_metrics = run["consensus_phase_metrics"]
    elif scope == "independentNode":
        phase_metrics = run["independent_node_metrics"]["phase_metrics"]
    else:
        raise ValueError(f"Unsupported node-failure metric scope: {scope}")
    return phase_metrics["DROPOUT"][field]


def _failure_arm_summary(payloads, arm):
    runs = [payload["threeObserverFailure"][arm] for payload in payloads]
    return {
        "failureWindow": _phase_statistics(runs, "DROPOUT"),
        "recovery": _phase_statistics(runs, "RECOVERY"),
        "consensusFailureWindow": _phase_statistics(
            runs, "DROPOUT", phase_key="consensus_phase_metrics"
        ),
        "consensusRecovery": _phase_statistics(
            runs, "RECOVERY", phase_key="consensus_phase_metrics"
        ),
        "finalActiveTrackCount": _statistics(
            run["final_active_track_count"] for run in runs
        ),
        "identitySwitchCount": _statistics(
            run["identity_switch_count"] for run in runs
        ),
        "consensusIdentitySwitchCount": _statistics(
            run["consensus_identity_switch_count"] for run in runs
        ),
        "recoveredLinkCount": _statistics(
            run["consensus_recovered_link_count"] for run in runs
        ),
        "processingMeanMs": _statistics(
            run["processing_mean_ms"] for run in runs
        ),
    }


def _phase_statistics(runs, phase, *, phase_key="phase_metrics"):
    values = tuple(runs)
    return {
        field: _statistics(
            run[phase_key][phase][field] for run in values
        )
        for field in ERROR_FIELDS
    }


def _unique_seeds(payloads):
    seeds = [int(payload["scenario"]["seed"]) for payload in payloads]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Robustness seeds must be unique within each scenario.")
    return sorted(seeds)


def _percent_change(value, baseline):
    if value is None or baseline in {None, 0.0}:
        return None
    return 100.0 * (float(value) / float(baseline) - 1.0)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
