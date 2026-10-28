"""Rank frozen P05 observer candidates using baseline observability only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.bilateral_multitarget_frontend import (
    BilateralFrontendConfig,
    _run_bilateral_online_arm,
    build_bilateral_raw_frontend,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/current_audit/p05_observer_selection_seed0.json"),
    )
    args = parser.parse_args()

    frontend = build_bilateral_raw_frontend(BilateralFrontendConfig(
        duration_seconds=float(args.duration),
        sample_interval_seconds=1.0,
        random_seed=int(args.seed),
    ))
    initialization_cache = {}
    records = []
    for observer_index, observer_id in enumerate(
        frontend.truth.manifest.friendly_observer_ids
    ):
        collector = {}
        _run_bilateral_online_arm(
            frontend,
            enable_cann=False,
            enable_consensus=False,
            enable_local_unified=False,
            tracking_modalities=("OPTICAL",),
            tracking_observer_ids=(observer_id,),
            require_scored_truth=False,
            sample_collector=collector,
            iod_initialization_cache=initialization_cache,
        )
        samples = {
            key: value for key, value in collector["local"].items()
            if key[0] == observer_id
        }
        records.append(_summarize(
            observer_index, observer_id, samples,
            duration=float(args.duration),
        ))
        print(
            f"[{observer_index + 1:02d}/15] {observer_id} "
            f"samples={records[-1]['sampleCount']} "
            f"targets={records[-1]['truthTargetCount']} "
            f"longest={records[-1]['longestContinuousEpochCount']} "
            f"rmse={records[-1]['positionRmseM']:.6f}"
        )

    ranked = sorted(records, key=lambda item: (
        -item["fullDurationTargetCount"],
        -item["longestContinuousEpochCount"],
        -item["coveredEpochCount"],
        item["positionRmseM"],
        item["observerIndex"],
    ))
    payload = {
        "schemaVersion": "p05-observer-selection-pilot.v1",
        "selectionUsesCandidatePerformance": False,
        "selectionRule": (
            "maximize full-duration target count, longest continuous baseline "
            "run, covered epoch count; then minimize optical position RMSE"
        ),
        "scenario": {
            "durationSeconds": float(args.duration),
            "sampleIntervalSeconds": 1.0,
            "seed": int(args.seed),
            "modalities": ["OPTICAL"],
            "cannEnabled": False,
            "localUnifiedUpdateEnabled": False,
        },
        "recommendedObserverIndex": ranked[0]["observerIndex"],
        "recommendedObserverId": ranked[0]["observerId"],
        "ranking": ranked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(args.output)


def _summarize(observer_index, observer_id, samples, *, duration):
    timestamps = sorted({float(key[2]) for key in samples})
    errors = np.asarray([
        float(sample.position_error_m) for sample in samples.values()
    ], dtype=float)
    by_target: dict[str, list[float]] = {}
    for (_node_id, target_id, timestamp), _sample in samples.items():
        by_target.setdefault(str(target_id), []).append(float(timestamp))
    target_runs = {
        target_id: _longest_continuous_epoch_count(values)
        for target_id, values in by_target.items()
    }
    expected_last = float(duration)
    full_duration_targets = sum(
        bool(values)
        and min(values) <= 3.0
        and np.isclose(max(values), expected_last)
        and target_runs[target_id] >= max(int(round(duration)) - 2, 1)
        for target_id, values in by_target.items()
    )
    return {
        "observerIndex": int(observer_index),
        "observerId": str(observer_id),
        "sampleCount": len(samples),
        "truthTargetCount": len(by_target),
        "coveredEpochCount": len(timestamps),
        "firstTimestamp": None if not timestamps else timestamps[0],
        "lastTimestamp": None if not timestamps else timestamps[-1],
        "fullDurationTargetCount": int(full_duration_targets),
        "longestContinuousEpochCount": max(target_runs.values(), default=0),
        "positionRmseM": (
            float(np.sqrt(np.mean(np.square(errors))))
            if errors.size else float("inf")
        ),
        "positionMeanErrorM": (
            float(np.mean(errors)) if errors.size else float("inf")
        ),
        "continuousEpochCountByTarget": dict(sorted(target_runs.items())),
    }


def _longest_continuous_epoch_count(timestamps) -> int:
    values = sorted(set(float(value) for value in timestamps))
    if not values:
        return 0
    longest = current = 1
    for previous, current_time in zip(values, values[1:]):
        if np.isclose(current_time - previous, 1.0):
            current += 1
        else:
            current = 1
        longest = max(longest, current)
    return longest


if __name__ == "__main__":
    main()
