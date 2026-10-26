"""Summarize frozen electro-optical failover dropout pilot results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


FIELDS = (
    ("position", "position_rmse_m", "position_sample_count"),
    ("velocity", "velocity_rmse_mps", "position_sample_count"),
    ("acceleration", "acceleration_rmse_mps2", "acceleration_sample_count"),
)
MODALITIES = ("OPTICAL", "INFRARED", "RADAR")
MODALITY_LABELS = {"OPTICAL": "光学", "INFRARED": "红外", "RADAR": "雷达"}
QUANTITY_LABELS = {
    "position": "位置", "velocity": "速度", "acceleration": "加速度",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--threshold-percent", type=float, default=15.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    records = [
        row
        for payload in payloads
        for row in _payload_records(payload, args.threshold_percent)
    ]

    aggregates = []
    for modality in MODALITIES:
        for quantity, _field, _sample_field in FIELDS:
            selected = [
                row for row in records
                if row["missingModality"] == modality
                and row["quantity"] == quantity
            ]
            count = sum(row["sampleCount"] for row in selected)
            control_rmse = math.sqrt(sum(
                row["sampleCount"] * row["controlRmse"] ** 2
                for row in selected
            ) / count)
            dropout_rmse = math.sqrt(sum(
                row["sampleCount"] * row["dropoutRmse"] ** 2
                for row in selected
            ) / count)
            pooled_increase = 100.0 * (dropout_rmse / control_rmse - 1.0)
            aggregates.append({
                "missingModality": modality,
                "quantity": quantity,
                "pooledControlRmse": control_rmse,
                "pooledDropoutRmse": dropout_rmse,
                "pooledIncreasePercent": pooled_increase,
                "meanSeedIncreasePercent": sum(
                    row["increasePercent"] for row in selected
                ) / len(selected),
                "worstSeedIncreasePercent": max(
                    row["increasePercent"] for row in selected
                ),
                "allSeedsPassed": all(row["passed"] for row in selected),
                "pooledPassed": pooled_increase <= args.threshold_percent,
            })

    coverage_passed = all(
        row["finalActiveTrackCount"]
        >= row["controlFinalActiveTrackCount"]
        and row["missingTruthSampleCount"]
        <= row["controlMissingTruthSampleCount"]
        for row in records
    )
    summary = {
        "schemaVersion": "electro-optical-composite-dropout-summary.v1",
        "thresholdPercent": float(args.threshold_percent),
        "seedCount": len(payloads),
        "seeds": sorted({row["seed"] for row in records}),
        "configuration": payloads[0]["scenario"],
        "independentRadialBackupEnabled": False,
        "perSeedRecords": records,
        "aggregates": aggregates,
        "coveragePassed": coverage_passed,
        "passed": (
            coverage_passed
            and all(row["allSeedsPassed"] for row in aggregates)
            and all(row["pooledPassed"] for row in aggregates)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "缺失模态", "状态量", "控制组池化RMSE", "试验组池化RMSE",
                "池化增幅(%)", "逐seed平均增幅(%)", "最坏seed增幅(%)",
                "池化判定", "逐seed判定",
            ),
        )
        writer.writeheader()
        for row in aggregates:
            writer.writerow({
                "缺失模态": MODALITY_LABELS[row["missingModality"]],
                "状态量": QUANTITY_LABELS[row["quantity"]],
                "控制组池化RMSE": row["pooledControlRmse"],
                "试验组池化RMSE": row["pooledDropoutRmse"],
                "池化增幅(%)": row["pooledIncreasePercent"],
                "逐seed平均增幅(%)": row["meanSeedIncreasePercent"],
                "最坏seed增幅(%)": row["worstSeedIncreasePercent"],
                "池化判定": "通过" if row["pooledPassed"] else "不通过",
                "逐seed判定": "通过" if row["allSeedsPassed"] else "不通过",
            })
    by_seed_path = args.output.with_name(args.output.stem + "_by_seed.csv")
    seed_fields = ["随机种子"] + [
        f"{MODALITY_LABELS[modality]}_{QUANTITY_LABELS[quantity]}增幅(%)"
        for modality in MODALITIES
        for quantity, _field, _sample_field in FIELDS
    ] + ["总体判定"]
    with by_seed_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=seed_fields)
        writer.writeheader()
        for seed in sorted({row["seed"] for row in records}):
            selected = [row for row in records if row["seed"] == seed]
            output_row = {"随机种子": seed}
            for row in selected:
                output_row[
                    f'{MODALITY_LABELS[row["missingModality"]]}_'
                    f'{QUANTITY_LABELS[row["quantity"]]}增幅(%)'
                ] = row["increasePercent"]
            output_row["总体判定"] = (
                "通过" if all(row["passed"] for row in selected) else "不通过"
            )
            writer.writerow(output_row)
    detail_path = args.output.with_name(args.output.stem + "_detail.csv")
    with detail_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    readme_path = args.output.with_name(args.output.stem + "_说明.txt")
    readme_path.write_text(
        "主CSV：按缺失模态和状态量汇总池化结果及最坏seed。\n"
        "_by_seed.csv：每个随机种子一行，适合在Excel/WPS中横向比较。\n"
        "_detail.csv：控制组、试验组和样本数的逐项长表，供审计使用。\n"
        "增幅小于0表示试验组误差低于控制组，即结果有所改善。\n"
        "位置为合同主判据；速度和派生加速度为同步报告的扩展检查。\n",
        encoding="utf-8",
    )
    summary["csvFiles"] = {
        "overview": csv_path.name,
        "bySeed": by_seed_path.name,
        "detail": detail_path.name,
        "readme": readme_path.name,
    }
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "passed": summary["passed"],
        "coveragePassed": coverage_passed,
        "aggregates": aggregates,
    }, ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")
    print(f"wrote {csv_path}")
    print(f"wrote {by_seed_path}")
    print(f"wrote {detail_path}")
    print(f"wrote {readme_path}")


def _payload_records(payload, threshold_percent):
    if "resultsByMissingModality" in payload:
        return _acceptance_payload_records(payload, threshold_percent)
    return _pilot_payload_records(payload, threshold_percent)


def _pilot_payload_records(payload, threshold_percent):
    seed = int(payload["scenario"]["seed"])
    control = payload["control"]["metrics"]
    control_tracks = int(payload["control"]["finalActiveTrackCount"])
    control_missing = int(payload["control"]["missingTruthSampleCount"])
    rows = []
    for modality in MODALITIES:
        dropout = payload["dropout"][modality]
        metrics = dropout["metrics"]
        rows.extend(_metric_rows(
            seed=seed,
            modality=modality,
            control=control,
            dropout=metrics,
            increase_by_field=dropout["increasePercent"],
            dropout_tracks=dropout["finalActiveTrackCount"],
            dropout_missing=dropout["missingTruthSampleCount"],
            control_tracks=control_tracks,
            control_missing=control_missing,
            threshold_percent=threshold_percent,
        ))
    return rows


def _acceptance_payload_records(payload, threshold_percent):
    seed = int(payload["scenario"]["seed"])
    control_arm = payload["noDropoutControl"]["incrementalCann"]
    control = control_arm["phase_metrics"]["DROPOUT"]
    rows = []
    for modality in MODALITIES:
        record = payload["resultsByMissingModality"][modality]
        dropout_arm = record["incrementalCann"]
        dropout = dropout_arm["phase_metrics"]["DROPOUT"]
        rows.extend(_metric_rows(
            seed=seed,
            modality=modality,
            control=control,
            dropout=dropout,
            increase_by_field=record["dropoutVsControlPercent"][
                "incrementalCann"
            ],
            dropout_tracks=dropout_arm["final_active_track_count"],
            dropout_missing=dropout_arm["missing_truth_sample_count"],
            control_tracks=control_arm["final_active_track_count"],
            control_missing=control_arm["missing_truth_sample_count"],
            threshold_percent=threshold_percent,
        ))
    return rows


def _metric_rows(
    *, seed, modality, control, dropout, increase_by_field,
    dropout_tracks, dropout_missing, control_tracks, control_missing,
    threshold_percent,
):
    return [{
        "seed": int(seed),
        "missingModality": modality,
        "quantity": quantity,
        "controlRmse": float(control[field]),
        "dropoutRmse": float(dropout[field]),
        "increasePercent": float(increase_by_field[field]),
        "sampleCount": int(dropout[sample_field]),
        "passed": float(increase_by_field[field]) <= threshold_percent,
        "finalActiveTrackCount": int(dropout_tracks),
        "missingTruthSampleCount": int(dropout_missing),
        "controlFinalActiveTrackCount": int(control_tracks),
        "controlMissingTruthSampleCount": int(control_missing),
    } for quantity, field, sample_field in FIELDS]


if __name__ == "__main__":
    main()
