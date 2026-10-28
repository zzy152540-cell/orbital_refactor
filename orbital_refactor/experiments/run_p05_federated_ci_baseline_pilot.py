"""Audit P05 federated multimodal EKF/CI against joint multimodal CANN."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from experiments.bilateral_multitarget_frontend import (
    BilateralFrontendConfig,
    _matched_view_metrics,
    _run_bilateral_online_arm,
    build_bilateral_raw_frontend,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--observer-index", type=int, default=5)
    parser.add_argument(
        "--target-index", type=int, default=1,
        help="Zero-based hostile target index; exactly one target is estimated.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "results/current_audit/"
            "p05_federated_ci_vs_joint_cann_seed0_120s.json"
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
    collectors = {}
    for name, enable_cann, enable_local_unified in (
        ("federated_multimodal_ekf_ci", False, False),
        ("joint_multimodal_cann", True, True),
    ):
        collector = {}
        _run_bilateral_online_arm(
            frontend,
            enable_cann=enable_cann,
            enable_consensus=False,
            enable_local_unified=enable_local_unified,
            local_multimodal_update_mode="batch",
            tracking_modalities=("RADAR", "INFRARED", "OPTICAL"),
            tracking_observer_ids=(observer_id,),
            tracking_target_ids=(target_id,),
            sample_collector=collector,
            iod_initialization_cache=initialization_cache,
        )
        collectors[name] = collector

    baseline = _observer_view(
        collectors["federated_multimodal_ekf_ci"], observer_id,
    )
    candidate = _observer_view(
        collectors["joint_multimodal_cann"], observer_id,
    )
    comparison = _matched_view_metrics(
        baseline["state"], candidate["state"],
        baseline["acceleration"], candidate["acceleration"],
    )
    payload = {
        "schemaVersion": "p05-federated-ci-baseline-pilot.v1",
        "acceptanceDefaultChanged": False,
        "scenario": {
            "durationSeconds": float(args.duration),
            "sampleIntervalSeconds": 1.0,
            "seed": int(args.seed),
            "observerIndex": int(args.observer_index),
            "observerId": observer_id,
            "targetIndex": int(args.target_index),
            "targetId": target_id,
        },
        "comparisonContract": {
            "sharedRenderedSensorFrames": True,
            "sharedInitialization": True,
            "sharedObserverTargetAndEpochKeys": True,
            "singleObserverSingleTarget": True,
            "baseline": "federated multimodal EKF with covariance intersection",
            "candidate": "joint same-epoch multimodal EKF with CANN",
        },
        "exactMatchedComparison": asdict(comparison),
        "baselineConvergenceAudit": _convergence_audit(
            baseline["state"], duration=float(args.duration),
        ),
        "candidateConvergenceAudit": _convergence_audit(
            candidate["state"], duration=float(args.duration),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(args.output)


def _observer_view(collector, observer_id: str) -> dict:
    return {
        "state": {
            key: sample for key, sample in collector["local"].items()
            if key[0] == observer_id
        },
        "acceleration": {
            key: sample
            for key, sample in collector["localAcceleration"].items()
            if key[0] == observer_id
        },
    }


def _convergence_audit(samples, *, duration: float) -> dict:
    records = [
        (str(key[1]), float(key[2]), float(sample.position_error_m))
        for key, sample in samples.items()
    ]
    by_target: dict[str, list[tuple[float, float]]] = {}
    for target_id, timestamp, error in records:
        by_target.setdefault(target_id, []).append((timestamp, error))
    per_target = {}
    for target_id, values in sorted(by_target.items()):
        values.sort()
        timestamps = np.asarray([item[0] for item in values], dtype=float)
        errors = np.asarray([item[1] for item in values], dtype=float)
        early = errors[timestamps <= duration / 3.0]
        late = errors[timestamps > 2.0 * duration / 3.0]
        slope = float(np.polyfit(timestamps, errors, 1)[0]) if len(values) > 1 else 0.0
        per_target[target_id] = {
            "sampleCount": len(values),
            "firstTimestamp": float(timestamps[0]),
            "lastTimestamp": float(timestamps[-1]),
            "firstErrorM": float(errors[0]),
            "lastErrorM": float(errors[-1]),
            "earlyRmseM": _rmse(early),
            "lateRmseM": _rmse(late),
            "linearErrorSlopeMPerSecond": slope,
            "lateNotWorseThanEarly": bool(
                early.size and late.size and _rmse(late) <= _rmse(early)
            ),
        }
    return {
        "sampleCount": len(records),
        "targetCount": len(by_target),
        "allSamplesFinite": bool(all(np.isfinite(item[2]) for item in records)),
        "targetCountWithLateNotWorseThanEarly": sum(
            item["lateNotWorseThanEarly"] for item in per_target.values()
        ),
        "perTarget": per_target,
    }


def _rmse(values) -> float | None:
    array = np.asarray(values, dtype=float)
    return None if not array.size else float(np.sqrt(np.mean(array * array)))


if __name__ == "__main__":
    main()
