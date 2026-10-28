"""Scan continuously visible one-observer/one-target P05 pair candidates."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from experiments.bilateral_multitarget_frontend import (
    BilateralFrontendConfig,
    _matched_view_metrics,
    _run_bilateral_online_arm,
    build_bilateral_raw_frontend,
)
from experiments.run_p05_federated_ci_baseline_pilot import (
    _convergence_audit,
    _observer_view,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold-percent", type=float, default=8.0)
    parser.add_argument("--baseline-reanchor-interval", type=int)
    parser.add_argument(
        "--baseline-reanchor-covariance-scale", type=float, default=1.0,
    )
    parser.add_argument(
        "--baseline-architecture", choices=("federated-ci", "sequential"),
        default="federated-ci",
    )
    parser.add_argument("--candidate-radar-alpha", type=float, default=1.0)
    parser.add_argument("--candidate-radar-range-alpha", type=float)
    parser.add_argument("--candidate-radar-range-rate-alpha", type=float)
    parser.add_argument(
        "--candidate-radar-covariance-scale", type=float, default=1.0,
    )
    parser.add_argument("--candidate-radar-range-covariance-scale", type=float)
    parser.add_argument(
        "--candidate-radar-range-rate-covariance-scale", type=float,
    )
    parser.add_argument(
        "--candidate-adaptive-radar-innovation-sigma", type=float, default=0.0,
        help="Positive enables innovation-adaptive radar CANN blending.",
    )
    parser.add_argument("--candidate-optical-alpha", type=float, default=1.0)
    parser.add_argument("--candidate-velocity-refinement-alpha", type=float)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/current_audit/p05_one_to_one_pair_scan.json"),
    )
    args = parser.parse_args()

    frontend = build_bilateral_raw_frontend(BilateralFrontendConfig(
        duration_seconds=float(args.duration),
        sample_interval_seconds=1.0,
        random_seed=int(args.seed),
    ))
    timestamps = tuple(float(value) for value in frontend.truth.timestamps)
    observers = frontend.truth.manifest.friendly_observer_ids
    targets = frontend.truth.manifest.hostile_target_ids
    pairs = []
    for observer_id in observers:
        for target_id in targets:
            visible_count = sum(
                observer_id in frontend.visible_observers_by_target_and_timestamp[
                    (target_id, timestamp)
                ]
                for timestamp in timestamps
            )
            if visible_count == len(timestamps):
                pairs.append((observer_id, target_id, visible_count))

    records = []
    for index, (observer_id, target_id, visible_count) in enumerate(pairs, 1):
        print(f"[{index}/{len(pairs)}] {observer_id} -> {target_id}", flush=True)
        try:
            initialization_cache = {}
            collectors = {}
            for name, enable_cann, enable_local_unified in (
                (
                    "baseline", False,
                    args.baseline_architecture == "sequential",
                ),
                ("candidate", True, True),
            ):
                collector = {}
                _run_bilateral_online_arm(
                    frontend,
                    enable_cann=enable_cann,
                    enable_consensus=False,
                    enable_local_unified=enable_local_unified,
                    local_multimodal_update_mode=(
                        "sequential"
                        if name == "baseline"
                        and args.baseline_architecture == "sequential"
                        else "batch"
                    ),
                    cann_radar_nominal_blend_alpha=(
                        args.candidate_radar_alpha
                        if name == "candidate" else 1.0
                    ),
                    cann_radar_range_nominal_blend_alpha=(
                        args.candidate_radar_range_alpha
                        if name == "candidate" else None
                    ),
                    cann_radar_range_rate_nominal_blend_alpha=(
                        args.candidate_radar_range_rate_alpha
                        if name == "candidate" else None
                    ),
                    cann_radar_nominal_covariance_scale=(
                        args.candidate_radar_covariance_scale
                        if name == "candidate" else 1.0
                    ),
                    cann_radar_range_nominal_covariance_scale=(
                        args.candidate_radar_range_covariance_scale
                        if name == "candidate" else None
                    ),
                    cann_radar_range_rate_nominal_covariance_scale=(
                        args.candidate_radar_range_rate_covariance_scale
                        if name == "candidate" else None
                    ),
                    cann_radar_adaptive_nominal_blend=(
                        name == "candidate"
                        and args.candidate_adaptive_radar_innovation_sigma > 0.0
                    ),
                    cann_radar_adaptive_innovation_sigma=(
                        args.candidate_adaptive_radar_innovation_sigma
                        if name == "candidate"
                        and args.candidate_adaptive_radar_innovation_sigma > 0.0
                        else 3.0
                    ),
                    cann_optical_nominal_blend_alpha=(
                        args.candidate_optical_alpha
                        if name == "candidate" else 1.0
                    ),
                    tracking_modalities=("RADAR", "INFRARED", "OPTICAL"),
                    tracking_observer_ids=(observer_id,),
                    tracking_target_ids=(target_id,),
                    local_federated_reanchor_interval_epochs=(
                        args.baseline_reanchor_interval
                        if name == "baseline" else None
                    ),
                    local_federated_reanchor_covariance_scale=(
                        args.baseline_reanchor_covariance_scale
                        if name == "baseline" else 1.0
                    ),
                    velocity_refinement_alpha=(
                        args.candidate_velocity_refinement_alpha
                        if name == "candidate" else None
                    ),
                    sample_collector=collector,
                    iod_initialization_cache=initialization_cache,
                )
                collectors[name] = _observer_view(collector, observer_id)
            metrics = _matched_view_metrics(
                collectors["baseline"]["state"],
                collectors["candidate"]["state"],
                collectors["baseline"]["acceleration"],
                collectors["candidate"]["acceleration"],
            )
            baseline_audit = _convergence_audit(
                collectors["baseline"]["state"], duration=float(args.duration),
            )
            candidate_audit = _convergence_audit(
                collectors["candidate"]["state"], duration=float(args.duration),
            )
            baseline_converged = (
                baseline_audit["targetCount"] == 1
                and baseline_audit["targetCountWithLateNotWorseThanEarly"] == 1
            )
            candidate_converged = (
                candidate_audit["targetCount"] == 1
                and candidate_audit["targetCountWithLateNotWorseThanEarly"] == 1
            )
            all_improvements_pass = all(
                value >= args.threshold_percent for value in (
                    metrics.position_rmse_improvement_percent,
                    metrics.velocity_rmse_improvement_percent,
                    metrics.acceleration_rmse_improvement_percent,
                )
            )
            records.append({
                "observerId": observer_id,
                "targetId": target_id,
                "visibleEpochCount": visible_count,
                "failed": False,
                "metrics": asdict(metrics),
                "baselineConverged": baseline_converged,
                "candidateConverged": candidate_converged,
                "allThreeRmseImprovementsPass": all_improvements_pass,
                "fullyQualified": bool(
                    baseline_converged and candidate_converged
                    and all_improvements_pass
                ),
                "baselineConvergenceAudit": baseline_audit,
                "candidateConvergenceAudit": candidate_audit,
            })
        except (ValueError, RuntimeError) as exc:
            print(
                f"  failed: {type(exc).__name__}: {exc}", flush=True,
            )
            records.append({
                "observerId": observer_id,
                "targetId": target_id,
                "visibleEpochCount": visible_count,
                "failed": True,
                "errorType": type(exc).__name__,
                "errorMessage": str(exc),
                "metrics": None,
                "baselineConverged": False,
                "candidateConverged": False,
                "allThreeRmseImprovementsPass": False,
                "fullyQualified": False,
            })

    records.sort(key=lambda item: (
        not item["fullyQualified"],
        not item["allThreeRmseImprovementsPass"],
        item["failed"],
        -(
            item["metrics"]["position_rmse_improvement_percent"]
            if item["metrics"] is not None else float("-inf")
        ),
    ))
    payload = {
        "schemaVersion": "p05-one-to-one-pair-scan.v1",
        "acceptanceDefaultChanged": False,
        "durationSeconds": float(args.duration),
        "seed": int(args.seed),
        "thresholdPercent": float(args.threshold_percent),
        "continuousVisibilityRequired": True,
        "baselineReanchorIntervalEpochs": args.baseline_reanchor_interval,
        "baselineReanchorCovarianceScale": (
            args.baseline_reanchor_covariance_scale
        ),
        "baselineArchitecture": args.baseline_architecture,
        "candidateCausalDenoising": {
            "radarAlpha": args.candidate_radar_alpha,
            "radarRangeAlpha": args.candidate_radar_range_alpha,
            "radarRangeRateAlpha": args.candidate_radar_range_rate_alpha,
            "radarCovarianceScale": args.candidate_radar_covariance_scale,
            "radarRangeCovarianceScale": (
                args.candidate_radar_range_covariance_scale
            ),
            "radarRangeRateCovarianceScale": (
                args.candidate_radar_range_rate_covariance_scale
            ),
            "adaptiveRadarInnovationSigma": (
                args.candidate_adaptive_radar_innovation_sigma
            ),
            "opticalAlpha": args.candidate_optical_alpha,
            "velocityRefinementAlpha": (
                args.candidate_velocity_refinement_alpha
            ),
        },
        "candidatePairCount": len(pairs),
        "failedPairCount": sum(item["failed"] for item in records),
        "allThreePassCount": sum(
            item["allThreeRmseImprovementsPass"] for item in records
        ),
        "fullyQualifiedCount": sum(item["fullyQualified"] for item in records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({
        key: payload[key] for key in (
            "candidatePairCount", "failedPairCount", "allThreePassCount",
            "fullyQualifiedCount",
        )
    }, ensure_ascii=False))
    for item in records[:10]:
        metrics = item["metrics"]
        if metrics is None:
            print(
                item["observerId"], "->", item["targetId"],
                "FAILED", item["errorType"], item["errorMessage"],
            )
            continue
        print(
            item["observerId"], "->", item["targetId"],
            "P/V/A=",
            f"{metrics['position_rmse_improvement_percent']:.2f}%/",
            f"{metrics['velocity_rmse_improvement_percent']:.2f}%/",
            f"{metrics['acceleration_rmse_improvement_percent']:.2f}%",
            "baselineConverged=", item["baselineConverged"],
            "candidateConverged=", item["candidateConverged"],
        )
    print(args.output)


if __name__ == "__main__":
    main()
