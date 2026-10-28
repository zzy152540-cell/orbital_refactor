"""Audit P05 baselines with common IOD then one-observer/one-target filtering."""

from __future__ import annotations

import argparse
from collections import Counter
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
        "--output", type=Path,
        default=Path(
            "results/current_audit/"
            "p05_strict_one_to_one_baseline_audit_seed0_120s.json"
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
    arm_specs = {
        "federated_ci": {
            "enable_cann": False, "enable_local_unified": False,
            "mode": "batch", "modalities": ("RADAR", "INFRARED", "OPTICAL"),
            "weights": None, "anchor": "rad",
        },
        "federated_ci_no_cross_gate": {
            "enable_cann": False, "enable_local_unified": False,
            "mode": "batch", "modalities": ("RADAR", "INFRARED", "OPTICAL"),
            "weights": None, "anchor": None, "reanchor": None,
        },
        "federated_fixed_040303": {
            "enable_cann": False, "enable_local_unified": False,
            "mode": "batch", "modalities": ("RADAR", "INFRARED", "OPTICAL"),
            "weights": {"rad": 0.4, "ir": 0.3, "opt": 0.3},
            "anchor": "rad", "reanchor": None,
        },
        "federated_fixed_no_cross_gate": {
            "enable_cann": False, "enable_local_unified": False,
            "mode": "batch", "modalities": ("RADAR", "INFRARED", "OPTICAL"),
            "weights": {"rad": 0.4, "ir": 0.3, "opt": 0.3},
            "anchor": None, "reanchor": None,
        },
        "sequential_multimodal_ekf": {
            "enable_cann": False, "enable_local_unified": True,
            "mode": "sequential", "modalities": ("RADAR", "INFRARED", "OPTICAL"),
            "weights": None, "anchor": "rad", "reanchor": None,
        },
        "joint_multimodal_ekf_no_cann": {
            "enable_cann": False, "enable_local_unified": True,
            "mode": "batch", "modalities": ("RADAR", "INFRARED", "OPTICAL"),
            "weights": None, "anchor": "rad", "reanchor": None,
        },
        "radar_only_ekf": {
            "enable_cann": False, "enable_local_unified": False,
            "mode": "batch", "modalities": ("RADAR",), "weights": None,
            "anchor": "rad", "reanchor": None,
        },
        "optical_only_ekf": {
            "enable_cann": False, "enable_local_unified": False,
            "mode": "batch", "modalities": ("OPTICAL",), "weights": None,
            "anchor": "rad", "reanchor": None,
        },
        "joint_multimodal_cann": {
            "enable_cann": True, "enable_local_unified": True,
            "mode": "batch", "modalities": ("RADAR", "INFRARED", "OPTICAL"),
            "weights": None, "anchor": "rad", "reanchor": None,
        },
    }
    for interval in (2, 3, 5, 10, 20, 30):
        arm_specs[f"federated_ci_reanchor_{interval}"] = {
            "enable_cann": False, "enable_local_unified": False,
            "mode": "batch", "modalities": ("RADAR", "INFRARED", "OPTICAL"),
            "weights": None, "anchor": "rad", "reanchor": interval,
        }
    for spec in arm_specs.values():
        spec.setdefault("reanchor", None)
        spec.setdefault("reanchor_scale", 1.0)
    for interval in (2, 3):
        for scale in (2.0, 4.0, 8.0, 16.0, 32.0):
            arm_specs[
                f"federated_ci_reanchor_{interval}_scale_{scale:g}"
            ] = {
                "enable_cann": False, "enable_local_unified": False,
                "mode": "batch",
                "modalities": ("RADAR", "INFRARED", "OPTICAL"),
                "weights": None, "anchor": "rad", "reanchor": interval,
                "reanchor_scale": scale,
            }
    for minimum_weight in (0.03, 0.05, 0.1, 0.15, 0.2, 0.25):
        arm_specs[f"federated_ci_min_weight_{minimum_weight:g}"] = {
            "enable_cann": False, "enable_local_unified": False,
            "mode": "batch",
            "modalities": ("RADAR", "INFRARED", "OPTICAL"),
            "weights": None, "anchor": "rad", "reanchor": None,
            "reanchor_scale": 1.0, "ci_minimum_weight": minimum_weight,
        }
    for window_epochs in (2, 3, 5):
        arm_specs[f"joint_multimodal_cann_window_{window_epochs}"] = {
            "enable_cann": True, "enable_local_unified": True,
            "mode": "batch",
            "modalities": ("RADAR", "INFRARED", "OPTICAL"),
            "weights": None, "anchor": "rad", "reanchor": None,
            "reanchor_scale": 1.0, "ci_minimum_weight": 0.0,
            "window_epochs": window_epochs,
        }
    for spec in arm_specs.values():
        spec.setdefault("ci_minimum_weight", 0.0)
        spec.setdefault("window_epochs", 1)
    initialization_cache = {}
    views = {}
    diagnostics = {}
    for name, spec in arm_specs.items():
        print(f"running {name}", flush=True)
        collector = {}
        histories = []
        _run_bilateral_online_arm(
            frontend,
            enable_cann=spec["enable_cann"],
            enable_consensus=False,
            enable_local_unified=spec["enable_local_unified"],
            local_multimodal_update_mode=spec["mode"],
            fixed_information_weights=spec["weights"],
            local_cross_modality_anchor=spec["anchor"],
            local_federated_reanchor_interval_epochs=spec["reanchor"],
            local_federated_reanchor_covariance_scale=(
                spec["reanchor_scale"]
            ),
            local_ci_minimum_weight=spec["ci_minimum_weight"],
            local_multimodal_window_epochs=spec["window_epochs"],
            tracking_modalities=spec["modalities"],
            tracking_observer_ids=(observer_id,),
            tracking_target_ids=(target_id,),
            restrict_tracking_observers_from_start=False,
            sample_collector=collector,
            iod_initialization_cache=initialization_cache,
            local_filter_history_callback=histories.append,
        )
        views[name] = _observer_view(collector, observer_id)
        diagnostics[name] = _summarize_filter_histories(histories)

    candidate = views["joint_multimodal_cann"]
    records = {}
    for name, view in views.items():
        audit = _convergence_audit(
            view["state"], duration=float(args.duration),
        )
        record = {
            "convergenceAudit": audit,
            "filterDiagnostics": diagnostics[name],
        }
        if name != "joint_multimodal_cann":
            metrics = _matched_view_metrics(
                view["state"], candidate["state"],
                view["acceleration"], candidate["acceleration"],
            )
            record["candidateImprovement"] = asdict(metrics)
            record["allThreeRmsePass8Percent"] = all(value >= 8.0 for value in (
                metrics.position_rmse_improvement_percent,
                metrics.velocity_rmse_improvement_percent,
                metrics.acceleration_rmse_improvement_percent,
            ))
        records[name] = record

    payload = {
        "schemaVersion": "p05-one-to-one-baseline-audit.v1",
        "acceptanceDefaultChanged": False,
        "inputContract": {
            "commonMultiobserverInitialization": True,
            "singleObserverDuringScoredFiltering": True,
            "singleTarget": True,
            "sharedRenderedSensorFrames": True,
            "sharedInitializationCache": True,
        },
        "scenario": {
            "durationSeconds": float(args.duration), "seed": int(args.seed),
            "observerId": observer_id, "targetId": target_id,
        },
        "records": records,
        "stableBaselineCandidateComparisons": {
            name: asdict(_matched_view_metrics(
                views["sequential_multimodal_ekf"]["state"], view["state"],
                views["sequential_multimodal_ekf"]["acceleration"],
                view["acceleration"],
            ))
            for name, view in views.items()
            if name.startswith("joint_multimodal_cann")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(args.output)


def _summarize_filter_histories(histories) -> dict:
    statistics = {}
    weight_sums = Counter()
    weight_counts = Counter()
    for history in histories:
        for modality, counts in getattr(history, "statistics", {}).items():
            target = statistics.setdefault(
                modality, {"accepted": 0, "rejected": 0, "skipped": 0},
            )
            for name in target:
                target[name] += int(counts.get(name, 0))
        for weights in getattr(history, "ci_weight_history", ()):
            for modality, value in (weights or {}).items():
                weight_sums[modality] += float(value)
                weight_counts[modality] += 1
    return {
        "historyCount": len(histories),
        "statistics": statistics,
        "meanFusionWeight": {
            name: weight_sums[name] / weight_counts[name]
            for name in sorted(weight_sums)
        },
    }


if __name__ == "__main__":
    main()
