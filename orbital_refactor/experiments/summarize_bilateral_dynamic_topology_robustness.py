"""Summarize supplemental dynamic-topology R06/R08 result artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


R06_METRICS = (
    "position_rmse_m", "velocity_rmse_mps", "acceleration_rmse_mps2",
)
R08_METRICS = (
    "position_mean_error_m",
    "velocity_mean_error_mps",
    "acceleration_mean_error_mps2",
)


def _statistics(values):
    data = np.asarray(tuple(float(value) for value in values), dtype=float)
    return {
        "mean": float(np.mean(data)),
        "minimum": float(np.min(data)),
        "maximum": float(np.max(data)),
    }


def summarize(payloads):
    records = tuple(payload["record"] for payload in payloads)
    r06 = {}
    for modality in ("OPTICAL", "INFRARED", "RADAR"):
        affected_squared_sum = 0.0
        affected_count = 0
        control_squared_sum = 0.0
        control_count = 0
        equal_counts = True
        for record in records:
            item = record["r06"][modality]
            affected = item["affected"]
            control = item["control"]
            ac = int(affected["position_sample_count"])
            cc = int(control["position_sample_count"])
            equal_counts &= ac == cc and ac > 0
            affected_squared_sum += float(affected["position_rmse_m"]) ** 2 * ac
            control_squared_sum += float(control["position_rmse_m"]) ** 2 * cc
            affected_count += ac
            control_count += cc
        affected_rmse = (affected_squared_sum / affected_count) ** 0.5
        control_rmse = (control_squared_sum / control_count) ** 0.5
        pooled_increase = 100.0 * (affected_rmse / control_rmse - 1.0)
        r06[modality] = {
            "thresholdPercent": 15.0,
            "primaryMetric": "position_rmse_m",
            "sampleCountEqualForEverySeed": bool(equal_counts),
            "coverageComparable": bool(equal_counts),
            "affectedPositionSampleCount": affected_count,
            "controlPositionSampleCount": control_count,
            "pooledPositionRmseIncreasePercent": pooled_increase,
            "thresholdMetNumerically": bool(pooled_increase <= 15.0),
            "passed": bool(equal_counts and pooled_increase <= 15.0),
            "seedStatisticsByMetric": {
                metric: _statistics(
                    record["r06"][modality]["increasePercent"][metric]
                    for record in records
                )
                for metric in R06_METRICS
            },
        }

    r08 = {}
    for case in (
        "SPATIAL", "ADJACENT", "HIGH_VISIBILITY", "FROZEN_RANDOM",
    ):
        primary = _statistics(
            record["r08"][case]["increasePercent"][
                "position_mean_error_m"
            ]
            for record in records
        )
        r08[case] = {
            "thresholdPercent": 12.0,
            "primaryMetric": "position_mean_error_m",
            "meanThresholdMet": bool(primary["mean"] <= 12.0),
            "passedSeedCount": sum(
                record["r08"][case]["increasePercent"][
                    "position_mean_error_m"
                ] <= 12.0
                for record in records
            ),
            "passed": bool(primary["mean"] <= 12.0),
            "maximumActiveComponentCount": max(
                int(record["r08"][case]["topology"][
                    "activeComponentCountMax"
                ])
                for record in records
            ),
            "smallestLargestActiveComponentSize": min(
                int(record["r08"][case]["topology"][
                    "smallestLargestActiveComponentSize"
                ])
                for record in records
            ),
            "seedStatisticsByMetric": {
                metric: _statistics(
                    record["r08"][case]["increasePercent"][metric]
                    for record in records
                )
                for metric in R08_METRICS
            },
        }
    return {
        "schemaVersion": "bilateral-dynamic-topology-robustness-summary.v1",
        "role": "supplemental_only",
        "seedCount": len(records),
        "seeds": sorted(int(record["seed"]) for record in records),
        "r06": r06,
        "r08": r08,
        "allR06Passed": all(item["passed"] for item in r06.values()),
        "allR08Passed": all(item["passed"] for item in r08.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize([
        json.loads(path.read_text(encoding="utf-8")) for path in args.inputs
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
