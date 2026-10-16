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
        "schemaVersion": "bilateral-baseline-matrix-summary.v1",
        "seedCount": len(values),
        "seeds": sorted(seeds),
        "durationSeconds": values[0]["scenario"]["durationSeconds"],
        "arms": arms,
        "localNodeBaselines": local_node_baselines,
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
        "fullRunMetrics": {
            field: _statistics(
                run["phase_metrics"]["FULL_RUN"].get(field) for run in runs
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


if __name__ == "__main__":
    main()
