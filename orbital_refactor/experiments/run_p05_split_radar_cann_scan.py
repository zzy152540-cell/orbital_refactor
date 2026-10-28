"""Scan independent range/range-rate CANN gains for strict P05 scoring."""

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
    parser.add_argument("--observer-index", type=int, default=6)
    parser.add_argument("--target-index", type=int, default=14)
    parser.add_argument(
        "--range-alphas", type=float, nargs="+",
        default=(0.6, 0.7, 0.8, 0.9),
    )
    parser.add_argument(
        "--range-rate-alphas", type=float, nargs="+",
        default=(0.6, 0.7, 0.8, 0.9, 1.0),
    )
    parser.add_argument(
        "--adaptive-sigmas", type=float, nargs="+", default=(0.0, 5.0),
        help="Zero disables innovation adaptation.",
    )
    parser.add_argument(
        "--range-covariance-scales", type=float, nargs="+", default=(1.0,),
    )
    parser.add_argument(
        "--range-rate-covariance-scales", type=float, nargs="+", default=(1.0,),
    )
    parser.add_argument(
        "--velocity-refinement-alphas", type=float, nargs="+",
        default=(0.0, 0.05, 0.075, 0.1),
    )
    parser.add_argument(
        "--position-refinement-alphas", type=float, nargs="+", default=(0.0,),
        help="Zero disables causal model-residual position refinement.",
    )
    parser.add_argument(
        "--range-rate-trend-windows", type=int, nargs="+", default=(0,),
    )
    parser.add_argument(
        "--range-rate-trend-measurement-alphas", type=float, nargs="+",
        default=(1.0,),
    )
    parser.add_argument("--optical-alpha", type=float, default=0.35)
    parser.add_argument("--threshold-percent", type=float, default=8.0)
    parser.add_argument(
        "--candidate-process-noise-accelerations", type=float, nargs="+",
        default=(1.0e-4,),
    )
    parser.add_argument(
        "--baseline-process-noise-acceleration", type=float, default=None,
        help=(
            "Optional baseline process-noise override for a paired fairness "
            "audit. Omit to retain the validated baseline default."
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "results/current_audit/p05_split_radar_cann_scan_seed0_120s.json"
        ),
    )
    args = parser.parse_args()

    frontend = build_bilateral_raw_frontend(BilateralFrontendConfig(
        duration_seconds=float(args.duration),
        sample_interval_seconds=1.0,
        random_seed=int(args.seed),
    ))
    observer_id = frontend.truth.manifest.friendly_observer_ids[
        int(args.observer_index)
    ]
    target_id = frontend.truth.manifest.hostile_target_ids[
        int(args.target_index)
    ]
    initialization_cache = {}
    baseline_collector = {}
    _run_bilateral_online_arm(
        frontend,
        enable_cann=False,
        enable_consensus=False,
        enable_local_unified=True,
        local_multimodal_update_mode="sequential",
        tracking_modalities=("RADAR", "INFRARED", "OPTICAL"),
        tracking_observer_ids=(observer_id,),
        tracking_target_ids=(target_id,),
        sample_collector=baseline_collector,
        iod_initialization_cache=initialization_cache,
        process_noise_acceleration_override=(
            args.baseline_process_noise_acceleration
        ),
    )
    baseline = _observer_view(baseline_collector, observer_id)
    baseline_audit = _convergence_audit(
        baseline["state"], duration=float(args.duration),
    )
    baseline_converged = bool(
        baseline_audit["targetCount"] == 1
        and baseline_audit["targetCountWithLateNotWorseThanEarly"] == 1
    )

    records = []
    configurations = (
        (
            range_alpha, rate_alpha, sigma, refinement_alpha,
            position_refinement_alpha, trend_window, trend_alpha,
            range_covariance_scale, rate_covariance_scale, process_noise,
        )
        for range_alpha in args.range_alphas
        for rate_alpha in args.range_rate_alphas
        for sigma in args.adaptive_sigmas
        for refinement_alpha in args.velocity_refinement_alphas
        for position_refinement_alpha in args.position_refinement_alphas
        for trend_window in args.range_rate_trend_windows
        for trend_alpha in args.range_rate_trend_measurement_alphas
        for range_covariance_scale in args.range_covariance_scales
        for rate_covariance_scale in args.range_rate_covariance_scales
        for process_noise in args.candidate_process_noise_accelerations
    )
    for index, configuration in enumerate(configurations, 1):
        (
            range_alpha, rate_alpha, sigma, refinement_alpha,
            position_refinement_alpha, trend_window, trend_alpha,
            range_covariance_scale, rate_covariance_scale, process_noise,
        ) = configuration
        print(
            f"[{index}] range={range_alpha:g} rate={rate_alpha:g} "
            f"sigma={sigma:g} velocity={refinement_alpha:g} "
            f"position={position_refinement_alpha:g} "
            f"trend={trend_window}/{trend_alpha:g} "
            f"Rscale={range_covariance_scale:g}/"
            f"{rate_covariance_scale:g} Q={process_noise:g}", flush=True,
        )
        collector = {}
        try:
            _run_bilateral_online_arm(
                frontend,
                enable_cann=True,
                enable_consensus=False,
                enable_local_unified=True,
                local_multimodal_update_mode="batch",
                cann_radar_range_nominal_blend_alpha=range_alpha,
                cann_radar_range_rate_nominal_blend_alpha=rate_alpha,
                cann_radar_range_nominal_covariance_scale=(
                    range_covariance_scale
                ),
                cann_radar_range_rate_nominal_covariance_scale=(
                    rate_covariance_scale
                ),
                cann_radar_adaptive_nominal_blend=sigma > 0.0,
                cann_radar_adaptive_innovation_sigma=(sigma if sigma > 0 else 3.0),
                cann_radar_range_rate_trend_window_epochs=trend_window,
                cann_radar_range_rate_trend_measurement_alpha=trend_alpha,
                cann_optical_nominal_blend_alpha=float(args.optical_alpha),
                tracking_modalities=("RADAR", "INFRARED", "OPTICAL"),
                tracking_observer_ids=(observer_id,),
                tracking_target_ids=(target_id,),
                velocity_refinement_alpha=(
                    refinement_alpha if refinement_alpha > 0.0 else None
                ),
                position_refinement_alpha=(
                    position_refinement_alpha
                    if position_refinement_alpha > 0.0 else None
                ),
                sample_collector=collector,
                iod_initialization_cache=initialization_cache,
                process_noise_acceleration_override=process_noise,
            )
            candidate = _observer_view(collector, observer_id)
            metrics = _matched_view_metrics(
                baseline["state"], candidate["state"],
                baseline["acceleration"], candidate["acceleration"],
            )
            audit = _convergence_audit(
                candidate["state"], duration=float(args.duration),
            )
            candidate_converged = bool(
                audit["targetCount"] == 1
                and audit["targetCountWithLateNotWorseThanEarly"] == 1
            )
            strict_pass = all(value >= args.threshold_percent for value in (
                metrics.position_rmse_improvement_percent,
                metrics.velocity_rmse_improvement_percent,
                metrics.acceleration_rmse_improvement_percent,
            ))
            records.append({
                "rangeAlpha": range_alpha,
                "rangeRateAlpha": rate_alpha,
                "adaptiveInnovationSigma": sigma,
                "velocityRefinementAlpha": refinement_alpha,
                "positionRefinementAlpha": position_refinement_alpha,
                "rangeRateTrendWindowEpochs": trend_window,
                "rangeRateTrendMeasurementAlpha": trend_alpha,
                "rangeCovarianceScale": range_covariance_scale,
                "rangeRateCovarianceScale": rate_covariance_scale,
                "processNoiseAcceleration": process_noise,
                "failed": False,
                "metrics": asdict(metrics),
                "candidateConverged": candidate_converged,
                "strictThreeMetricPass": strict_pass,
                "fullyQualified": bool(
                    baseline_converged and candidate_converged and strict_pass
                ),
                "candidateConvergenceAudit": audit,
            })
        except (ValueError, RuntimeError) as exc:
            records.append({
                "rangeAlpha": range_alpha,
                "rangeRateAlpha": rate_alpha,
                "adaptiveInnovationSigma": sigma,
                "velocityRefinementAlpha": refinement_alpha,
                "positionRefinementAlpha": position_refinement_alpha,
                "rangeRateTrendWindowEpochs": trend_window,
                "rangeRateTrendMeasurementAlpha": trend_alpha,
                "rangeCovarianceScale": range_covariance_scale,
                "rangeRateCovarianceScale": rate_covariance_scale,
                "processNoiseAcceleration": process_noise,
                "failed": True,
                "errorType": type(exc).__name__,
                "errorMessage": str(exc),
                "metrics": None,
                "candidateConverged": False,
                "strictThreeMetricPass": False,
                "fullyQualified": False,
            })

    def sort_key(item):
        if item["metrics"] is None:
            return (True, True, True, float("inf"))
        values = (
            item["metrics"]["position_rmse_improvement_percent"],
            item["metrics"]["velocity_rmse_improvement_percent"],
            item["metrics"]["acceleration_rmse_improvement_percent"],
        )
        return (
            not item["fullyQualified"],
            not item["candidateConverged"],
            item["failed"],
            -min(values),
        )

    records.sort(key=sort_key)
    payload = {
        "schemaVersion": "p05-split-radar-cann-scan.v1",
        "acceptanceDefaultChanged": False,
        "strictPerMetricThresholdPercent": float(args.threshold_percent),
        "scenario": {
            "durationSeconds": float(args.duration),
            "seed": int(args.seed),
            "observerId": observer_id,
            "targetId": target_id,
            "commonMultiobserverInitialization": True,
            "singleObserverDuringScoredFiltering": True,
        },
        "baseline": "stable sequential multimodal EKF without CANN",
        "baselineProcessNoiseAcceleration": (
            args.baseline_process_noise_acceleration
        ),
        "candidate": "joint multimodal EKF with split-channel CANN",
        "baselineConverged": baseline_converged,
        "baselineConvergenceAudit": baseline_audit,
        "fullyQualifiedCount": sum(item["fullyQualified"] for item in records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"qualified={payload['fullyQualifiedCount']}")
    for item in records[:10]:
        if item["metrics"] is None:
            continue
        metrics = item["metrics"]
        print(
            "range/rate/sigma/velocity=",
            item["rangeAlpha"], item["rangeRateAlpha"],
            item["adaptiveInnovationSigma"],
            item["velocityRefinementAlpha"],
            item["positionRefinementAlpha"],
            item["rangeRateTrendWindowEpochs"],
            item["rangeRateTrendMeasurementAlpha"],
            item["rangeCovarianceScale"],
            item["rangeRateCovarianceScale"],
            item["processNoiseAcceleration"],
            "P/V/A=",
            metrics["position_rmse_improvement_percent"],
            metrics["velocity_rmse_improvement_percent"],
            metrics["acceleration_rmse_improvement_percent"],
            "converged=", item["candidateConverged"],
        )
    print(args.output)


if __name__ == "__main__":
    main()
