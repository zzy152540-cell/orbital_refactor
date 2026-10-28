"""Scan causal CANN denoising against a stable sequential EKF baseline."""

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
    parser.add_argument("--observer-index", type=int, default=5)
    parser.add_argument("--target-index", type=int, default=1)
    parser.add_argument(
        "--radar-alphas", type=float, nargs="+",
        default=(0.2, 0.35, 0.5, 0.65, 0.8, 1.0),
    )
    parser.add_argument(
        "--optical-alphas", type=float, nargs="+",
        default=(0.35, 0.5, 0.75, 1.0),
    )
    parser.add_argument(
        "--velocity-refinement-alphas", type=float, nargs="+", default=(0.0,),
        help="Zero disables the optional causal velocity refinement.",
    )
    parser.add_argument(
        "--adaptive-radar-innovation-sigma", type=float, default=0.0,
        help="Positive enables innovation-adaptive radar CANN blending.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "results/current_audit/"
            "p05_one_to_one_cann_denoising_scan_seed0_120s.json"
        ),
    )
    args = parser.parse_args()

    frontend = build_bilateral_raw_frontend(BilateralFrontendConfig(
        duration_seconds=float(args.duration), sample_interval_seconds=1.0,
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
        enable_cann=False, enable_consensus=False,
        enable_local_unified=True,
        local_multimodal_update_mode="sequential",
        tracking_modalities=("RADAR", "INFRARED", "OPTICAL"),
        tracking_observer_ids=(observer_id,),
        tracking_target_ids=(target_id,),
        sample_collector=baseline_collector,
        iod_initialization_cache=initialization_cache,
    )
    baseline = _observer_view(baseline_collector, observer_id)

    records = []
    radar_alphas = tuple(args.radar_alphas)
    optical_alphas = tuple(args.optical_alphas)
    for radar_alpha in radar_alphas:
        covariance_scales = tuple(dict.fromkeys((
            1.0,
            radar_alpha / (2.0 - radar_alpha),
        )))
        for covariance_scale in covariance_scales:
            for optical_alpha in optical_alphas:
                for refinement_alpha in args.velocity_refinement_alphas:
                    print(
                        f"radar={radar_alpha:g} covariance={covariance_scale:g} "
                        f"optical={optical_alpha:g} "
                        f"velocity={refinement_alpha:g}", flush=True,
                    )
                    collector = {}
                    try:
                        _run_bilateral_online_arm(
                            frontend,
                            enable_cann=True, enable_consensus=False,
                            enable_local_unified=True,
                            local_multimodal_update_mode="batch",
                            cann_radar_nominal_blend_alpha=radar_alpha,
                            cann_radar_nominal_covariance_scale=covariance_scale,
                            cann_radar_adaptive_nominal_blend=(
                                args.adaptive_radar_innovation_sigma > 0.0
                            ),
                            cann_radar_adaptive_innovation_sigma=(
                                args.adaptive_radar_innovation_sigma
                                if args.adaptive_radar_innovation_sigma > 0.0
                                else 3.0
                            ),
                            cann_optical_nominal_blend_alpha=optical_alpha,
                            tracking_modalities=("RADAR", "INFRARED", "OPTICAL"),
                            tracking_observer_ids=(observer_id,),
                            tracking_target_ids=(target_id,),
                            velocity_refinement_alpha=(
                                refinement_alpha
                                if refinement_alpha > 0.0 else None
                            ),
                            sample_collector=collector,
                            iod_initialization_cache=initialization_cache,
                        )
                        candidate = _observer_view(collector, observer_id)
                        metrics = _matched_view_metrics(
                            baseline["state"], candidate["state"],
                            baseline["acceleration"], candidate["acceleration"],
                        )
                        audit = _convergence_audit(
                            candidate["state"], duration=float(args.duration),
                        )
                        records.append({
                            "radarAlpha": radar_alpha,
                            "radarCovarianceScale": covariance_scale,
                            "opticalAlpha": optical_alpha,
                            "velocityRefinementAlpha": refinement_alpha,
                            "failed": False,
                            "metrics": asdict(metrics),
                            "candidateConverged": (
                                audit["targetCount"] == 1
                                and audit[
                                    "targetCountWithLateNotWorseThanEarly"
                                ] == 1
                            ),
                            "allThreeRmsePass8Percent": all(
                                value >= 8.0 for value in (
                                    metrics.position_rmse_improvement_percent,
                                    metrics.velocity_rmse_improvement_percent,
                                    metrics.acceleration_rmse_improvement_percent,
                                )
                            ),
                            "candidateConvergenceAudit": audit,
                        })
                    except (ValueError, RuntimeError) as exc:
                        records.append({
                            "radarAlpha": radar_alpha,
                            "radarCovarianceScale": covariance_scale,
                            "opticalAlpha": optical_alpha,
                            "velocityRefinementAlpha": refinement_alpha,
                            "failed": True,
                            "errorType": type(exc).__name__,
                            "errorMessage": str(exc),
                            "metrics": None,
                            "candidateConverged": False,
                            "allThreeRmsePass8Percent": False,
                        })
    records.sort(key=lambda item: (
        not item["allThreeRmsePass8Percent"],
        not item["candidateConverged"],
        item["failed"],
        -min(
            item["metrics"]["position_rmse_improvement_percent"],
            item["metrics"]["velocity_rmse_improvement_percent"],
            item["metrics"]["acceleration_rmse_improvement_percent"],
        ),
    ) if item["metrics"] is not None else (True, True, True, float("inf")))
    payload = {
        "schemaVersion": "p05-one-to-one-cann-denoising-scan.v1",
        "acceptanceDefaultChanged": False,
        "scenario": {
            "durationSeconds": float(args.duration), "seed": int(args.seed),
            "observerId": observer_id, "targetId": target_id,
            "commonMultiobserverInitialization": True,
            "singleObserverDuringScoredFiltering": True,
        },
        "baseline": "sequential multimodal EKF without CANN",
        "candidate": "joint multimodal EKF with causal CANN denoising",
        "qualifiedCount": sum(
            item["allThreeRmsePass8Percent"] and item["candidateConverged"]
            for item in records
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"qualified={payload['qualifiedCount']}")
    for item in records[:10]:
        metrics = item["metrics"]
        if metrics is None:
            continue
        print(
            item["radarAlpha"], item["radarCovarianceScale"],
            item["opticalAlpha"], item["velocityRefinementAlpha"],
            metrics["position_rmse_improvement_percent"],
            metrics["velocity_rmse_improvement_percent"],
            metrics["acceleration_rmse_improvement_percent"],
            item["candidateConverged"],
        )
    print(args.output)


if __name__ == "__main__":
    main()
