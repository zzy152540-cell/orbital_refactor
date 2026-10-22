"""Frozen acceptance-test plans and result normalization for the Qt launcher."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import csv
import json
from pathlib import Path
from experiments.summarize_bilateral_baseline_matrix import summarize_payloads
from experiments.summarize_bilateral_robustness import (
    NODE_FAILURE_MAX_INCREASE_PERCENT,
    _acceptance_result,
    _dropout_coverage_comparable,
    _dropout_modality_summary,
    _failure_consensus_coverage_comparable,
    _failure_percent_summary,
)


QUICK_MODE = "quick"
FORMAL_MODE = "formal"
ACCEPTANCE_OUTPUT_ROOT = Path("results/acceptance_gui")
RADAR_DROPOUT_BACKUP_RANGE_SIGMA_M = 1.0
RADAR_DROPOUT_BACKUP_RANGE_RATE_SIGMA_MPS = 0.0125


@dataclass(frozen=True)
class AcceptancePreset:
    preset_id: str
    name: str
    purpose: str
    requirement: str
    result_kind: str


@dataclass(frozen=True)
class AcceptanceJob:
    label: str
    module: str
    arguments: tuple[str, ...]
    output_path: str
    role: str = "result"


@dataclass(frozen=True)
class AcceptancePlan:
    preset: AcceptancePreset
    mode: str
    output_directory: str
    duration_seconds: float | None
    seeds: tuple[int, ...]
    jobs: tuple[AcceptanceJob, ...]
    configuration_family: str
    control_family: str
    recording_path: str | None = None

    @property
    def formal(self) -> bool:
        return self.mode == FORMAL_MODE


@dataclass(frozen=True)
class AcceptanceMetric:
    metric: str
    value: float | None
    threshold: str
    passed: bool | None
    unit: str = "%"
    acceptance_required: bool = True


@dataclass(frozen=True)
class AcceptanceOutcome:
    title: str
    status: str
    metrics: tuple[AcceptanceMetric, ...]
    summary_path: str
    note: str


PRESETS = (
    AcceptancePreset(
        "P05_BASELINE", "P05 单星基准估计对照组",
        "运行P05普通单星多模态基准算法，记录其位置RMSE。",
        "记录控制组位置RMSE。", "baseline_p05_control",
    ),
    AcceptancePreset(
        "P05_FUSION", "P05 单星多模态融合试验组",
        "运行联合多模态融合+CANN算法，并与本次独立生成的P05基准组比较。",
        "位置RMSE降幅≥8%。", "baseline_p05_candidate",
    ),
    AcceptancePreset(
        "P07_INDEPENDENT", "P07 多星独立估计对照组",
        "运行各节点相互独立的本地估计，记录平均位置误差。",
        "记录控制组平均位置误差。", "baseline_p07_control",
    ),
    AcceptancePreset(
        "P07_COOPERATIVE", "P07 多星协同估计试验组",
        "运行多星Consensus-CI协同估计，并与本次独立生成的多星独立估计组比较。",
        "平均位置误差降幅≥6%。", "baseline_p07_candidate",
    ),
    AcceptancePreset(
        "R06_CONTROL", "R06 多模态正常观测对照组",
        "三种物理观测模态均正常，在冻结评分窗口内记录位置RMSE。",
        "记录无模态缺失控制组位置RMSE。", "r06_control",
    ),
    AcceptancePreset(
        "R06_OPTICAL", "R06 光学模态缺失试验组",
        "冻结窗口内仅移除光学模态。",
        "位置RMSE增幅≤15%。", "dropout",
    ),
    AcceptancePreset(
        "R06_INFRARED", "R06 红外模态缺失试验组",
        "冻结窗口内仅移除红外模态。",
        "位置RMSE增幅≤15%。", "dropout",
    ),
    AcceptancePreset(
        "R06_RADAR", "R06 雷达模态缺失试验组",
        "冻结窗口内移除雷达模态，并启用冻结的数值径向备份。",
        "位置RMSE增幅≤15%；数值径向备份不声明物理传感器性能。", "dropout",
    ),
    AcceptancePreset(
        "R08_CONTROL", "R08 多星节点正常运行对照组",
        "15个节点均正常观测并进行双向通信，记录存活节点平均位置误差。",
        "记录无节点失联控制组平均位置误差。", "r08_control",
    ),
    AcceptancePreset(
        "R08_SPATIAL", "R08 空间分散节点失联试验组",
        "节点0、5、10在冻结窗口内停止观测与双向通信。",
        "存活节点平均位置误差增幅≤12%。", "failure",
    ),
    AcceptancePreset(
        "R08_ADJACENT", "R08 拓扑相邻节点失联试验组",
        "节点0、1、2在冻结窗口内停止观测与双向通信。",
        "存活节点平均位置误差增幅≤12%。", "failure",
    ),
    AcceptancePreset(
        "R08_HIGH_VISIBILITY", "R08 高可见性节点失联试验组",
        "节点0、12、14在冻结窗口内停止观测与双向通信。",
        "存活节点平均位置误差增幅≤12%。", "failure",
    ),
    AcceptancePreset(
        "R08_RANDOM", "R08 冻结随机组合节点失联试验组",
        "每个随机种子使用验收配置中预先冻结的三节点组合。",
        "存活节点平均位置误差增幅≤12%。", "failure",
    ),
    AcceptancePreset(
        "P09", "P09 多星协同状态同步延迟",
        "15个独立节点进程通过环形UDP链路交换目标报告并执行Consensus-CI。",
        "同步延迟均值≤100 ms，且协议完整。", "latency",
    ),
    AcceptancePreset(
        "F10", "F10 双边星群可视化演示",
        "生成15颗我方观测星和15颗目标星的可视化录像并打开回放。",
        "数据契约自动检查通过，界面显示项目需人工确认。", "visualization",
    ),
)
PRESET_BY_ID = {preset.preset_id: preset for preset in PRESETS}

R08_FIXED_INDICES = {
    "R08_SPATIAL": (0, 5, 10),
    "R08_ADJACENT": (0, 1, 2),
    "R08_HIGH_VISIBILITY": (0, 12, 14),
}
R08_RANDOM_INDICES = {
    0: (7, 8, 11), 1: (6, 7, 11), 2: (1, 3, 10),
    3: (1, 2, 10), 4: (9, 13, 14),
}


def make_acceptance_output_directory(
    preset_id: str, *, root: str | Path = ACCEPTANCE_OUTPUT_ROOT,
    timestamp: datetime | None = None,
) -> Path:
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return Path(root) / preset_id.lower() / stamp


def build_acceptance_plan(
    preset_id: str, mode: str, output_directory: str | Path,
) -> AcceptancePlan:
    if preset_id not in PRESET_BY_ID:
        raise ValueError(f"Unknown acceptance preset: {preset_id}")
    if mode not in {QUICK_MODE, FORMAL_MODE}:
        raise ValueError(f"Unknown acceptance mode: {mode}")
    preset = PRESET_BY_ID[preset_id]
    output = Path(output_directory)
    duration = 20.0 if mode == QUICK_MODE else 120.0
    seeds = (0,) if mode == QUICK_MODE else (0, 1, 2, 3, 4)
    jobs: list[AcceptanceJob] = []
    configuration_family = "nominal"
    control_family = "nominal"

    if preset.result_kind.startswith("baseline_"):
        for seed in seeds:
            target = output / f"baseline_seed{seed}.json"
            jobs.append(_job(
                f"independent paired baseline seed {seed}",
                "experiments.run_bilateral_baseline_matrix",
                "--duration", duration, "--seed", seed,
                "--output", target,
                output=target,
            ))
    elif preset.result_kind == "r06_control":
        start, end = _event_window(mode)
        for seed in seeds:
            target = output / f"control_seed{seed}.json"
            jobs.append(_job(
                f"R06 normal control seed {seed}",
                "experiments.run_bilateral_acceptance_control",
                "--duration", duration, "--event-start", start,
                "--event-end", end, "--seed", seed,
                "--family", "nominal", "--output", target,
                output=target,
            ))
    elif preset.result_kind == "dropout":
        modality = preset_id.removeprefix("R06_")
        start, end = _event_window(mode)
        control_family = (
            "radar-backup" if modality == "RADAR" else "nominal"
        )
        configuration_family = control_family
        for seed in seeds:
            target = output / f"{modality.lower()}_dropout_seed{seed}.json"
            arguments: list[object] = [
                "--duration", duration, "--dropout-start", start,
                "--dropout-end", end, "--seed", seed,
                "--local-unified-multimodal-update",
                "--modalities", modality,
            ]
            if modality == "RADAR":
                arguments += [
                    "--backup-radial-range-sigma-m",
                    RADAR_DROPOUT_BACKUP_RANGE_SIGMA_M,
                    "--backup-radial-range-rate-sigma-mps",
                    RADAR_DROPOUT_BACKUP_RANGE_RATE_SIGMA_MPS,
                ]
            arguments += ["--output", target]
            jobs.append(_job(
                f"{modality} seed {seed}",
                "experiments.run_bilateral_dropout_prescan", *arguments,
                output=target,
            ))
    elif preset.result_kind == "r08_control":
        start, end = _event_window(mode)
        for seed in seeds:
            target = output / f"control_seed{seed}.json"
            jobs.append(_job(
                f"R08 normal control seed {seed}",
                "experiments.run_bilateral_acceptance_control",
                "--duration", duration, "--event-start", start,
                "--event-end", end, "--seed", seed,
                "--family", "nominal", "--output", target,
                output=target,
            ))
    elif preset.result_kind == "failure":
        start, end = _event_window(mode)
        for seed in seeds:
            indices = (
                R08_RANDOM_INDICES[seed]
                if preset_id == "R08_RANDOM"
                else R08_FIXED_INDICES[preset_id]
            )
            target = output / f"failure_seed{seed}.json"
            jobs.append(_job(
                f"seed {seed}, nodes {','.join(map(str, indices))}",
                "experiments.run_bilateral_node_failure_prescan",
                "--duration", duration, "--failure-start", start,
                "--failure-end", end, "--seed", seed,
                "--failed-observer-indices", *indices,
                "--local-unified-multimodal-update",
                "--output", target,
                output=target,
            ))
    elif preset.result_kind == "latency":
        configuration_family = "latency"
        control_family = "none"
        frames = 20 if mode == QUICK_MODE else 120
        target = output / "latency"
        jobs.append(_job(
            f"{frames} frames", "experiments.run_multitarget_distributed_node_sync_latency_acceptance",
            "--frames", frames, "--threshold-ms", 100.0,
            "--consensus-grid-points", 31, "--output", target,
            output=target / "summary.json",
        ))
        duration = None
        seeds = ()
    elif preset.result_kind == "visualization":
        configuration_family = "visualization"
        control_family = "none"
        target = output / "recording"
        jobs.append(_job(
            "bilateral visualization", "experiments.run_bilateral_visualization_recording",
            "--output", target, "--duration", duration, "--dt", 1.0, "--seed", 0,
            "--cann-node-limit", 0,
            "--radial-backup-profile", "none", "--consensus",
            output=target / "manifest.json",
        ))
        seeds = (0,)
    recording_path = None
    if preset.result_kind not in {"latency", "visualization"}:
        recording = output / "recording"
        jobs.append(_acceptance_recording_job(
            preset_id, duration=duration, output=recording,
            event_window=_event_window(mode),
        ))
        recording_path = str(recording)
    elif preset.result_kind == "visualization":
        recording_path = str(output / "recording")
    return AcceptancePlan(
        preset=preset, mode=mode, output_directory=str(output),
        duration_seconds=duration, seeds=seeds, jobs=tuple(jobs),
        configuration_family=configuration_family,
        control_family=control_family,
        recording_path=recording_path,
    )


def finalize_acceptance_plan(plan: AcceptancePlan) -> AcceptanceOutcome:
    output = Path(plan.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    payloads = [
        _load(job.output_path) for job in plan.jobs if job.role == "result"
    ]
    kind = plan.preset.result_kind
    if kind.startswith("baseline_"):
        summary = summarize_payloads(payloads)
        if kind.startswith("baseline_p05"):
            p05 = summary["fourMetricEvidence"]["P05"][
                "experimentalJointMeasurementCandidate"
            ]
            rmse = p05["positionRmse"]
            velocity_rmse = p05["velocityRmse"]
            p05_control_acceleration = summary["localNodeBaselines"][
                "single_node_multimodal_no_cann"
            ]["fullRunMetrics"]["acceleration_rmse_mps2"]["mean"]
            p05_candidate_acceleration = summary["arms"][
                "single_observer_joint_multimodal_cann"
            ]["fullRunMetrics"]["acceleration_rmse_mps2"]["mean"]
            if kind == "baseline_p05_control":
                metrics = (
                    AcceptanceMetric(
                        "P05 基准算法位置RMSE", rmse["local"],
                        "记录项", None, "m",
                    ),
                    AcceptanceMetric(
                        "P05 基准算法速度RMSE", velocity_rmse["local"],
                        "记录项", None, "m/s",
                    ),
                    AcceptanceMetric(
                        "P05 基准算法加速度RMSE（派生）",
                        p05_control_acceleration,
                        "记录项", None, "m/s²",
                    ),
                )
            else:
                value = rmse["improvementPercent"]
                metrics = (
                    AcceptanceMetric(
                        "P05 融合算法位置RMSE", rmse["consensus"],
                        "记录项", None, "m",
                    ),
                    AcceptanceMetric(
                        "P05 位置RMSE降幅", value, "≥ 8%",
                        bool(
                            p05["coverageComparable"] and value is not None
                            and value >= 8.0
                        ),
                    ),
                    AcceptanceMetric(
                        "P05 融合算法速度RMSE",
                        velocity_rmse["consensus"],
                        "记录项", None, "m/s",
                    ),
                    AcceptanceMetric(
                        "P05 速度RMSE降幅",
                        velocity_rmse["improvementPercent"],
                        "扩展检查 ≥ 8%",
                        bool(
                            p05["coverageComparable"]
                            and velocity_rmse["improvementPercent"] is not None
                            and velocity_rmse["improvementPercent"] >= 8.0
                        ),
                        "%", False,
                    ),
                    AcceptanceMetric(
                        "P05 融合算法全节点加速度RMSE（派生）",
                        p05_candidate_acceleration,
                        "记录项；与单节点控制组不同口径", None, "m/s²",
                    ),
                )
        else:
            p07 = summary["fourMetricEvidence"]["P07"][
                "adjustedJointMeasurementAlgorithm"
            ]
            mean_error = p07["positionMeanError"]
            velocity_mean_error = p07["velocityMeanError"]
            p07_control_acceleration = summary["localNodeBaselines"][
                "independent_15_node_multimodal_cann"
            ]["fullRunMetrics"]["acceleration_rmse_mps2"]["mean"]
            p07_candidate_acceleration = summary["arms"][
                "fixed_topology_joint_multimodal_cann"
            ]["consensusFullRunMetrics"]["acceleration_rmse_mps2"]["mean"]
            if kind == "baseline_p07_control":
                metrics = (
                    AcceptanceMetric(
                        "P07 多星独立估计平均位置误差",
                        mean_error["local"], "记录项", None, "m",
                    ),
                    AcceptanceMetric(
                        "P07 多星独立估计平均速度误差",
                        velocity_mean_error["local"],
                        "记录项", None, "m/s",
                    ),
                    AcceptanceMetric(
                        "P07 多星独立估计加速度RMSE（派生）",
                        p07_control_acceleration,
                        "记录项", None, "m/s²",
                    ),
                )
            else:
                value = mean_error["improvementPercent"]
                metrics = (
                    AcceptanceMetric(
                        "P07 多星协同估计平均位置误差",
                        mean_error["consensus"], "记录项", None, "m",
                    ),
                    AcceptanceMetric(
                        "P07 平均位置误差降幅", value, "≥ 6%",
                        bool(
                            p07["coverageComparable"] and value is not None
                            and value >= 6.0
                        ),
                    ),
                    AcceptanceMetric(
                        "P07 多星协同估计平均速度误差",
                        velocity_mean_error["consensus"],
                        "记录项", None, "m/s",
                    ),
                    AcceptanceMetric(
                        "P07 平均速度误差降幅",
                        velocity_mean_error["improvementPercent"],
                        "扩展检查 ≥ 6%",
                        bool(
                            p07["coverageComparable"]
                            and velocity_mean_error[
                                "improvementPercent"
                            ] is not None
                            and velocity_mean_error[
                                "improvementPercent"
                            ] >= 6.0
                        ),
                        "%", False,
                    ),
                    AcceptanceMetric(
                        "P07 多星协同估计加速度RMSE（派生）",
                        p07_candidate_acceleration,
                        "记录项；非同样本降幅判据", None, "m/s²",
                    ),
                )
    elif kind in {"r06_control", "r08_control"}:
        branch = (
            "phase_metrics" if kind == "r06_control"
            else "consensus_phase_metrics"
        )
        fields = (
            (
                ("位置RMSE", "position_rmse_m", "m"),
                ("速度RMSE", "velocity_rmse_mps", "m/s"),
                ("加速度RMSE", "acceleration_rmse_mps2", "m/s²"),
            ) if kind == "r06_control" else (
                ("平均位置误差", "position_mean_error_m", "m"),
                ("平均速度误差", "velocity_mean_error_mps", "m/s"),
                ("平均加速度误差", "acceleration_mean_error_mps2", "m/s²"),
            )
        )
        values_by_field = {
            field: [
                payload["control"]["incrementalCann"][branch]["DROPOUT"][field]
                for payload in payloads
            ]
            for _label, field, _unit in fields
        }
        summary = {
            "schemaVersion": "acceptance-gui-control.v1",
            "case": plan.preset.preset_id,
            "seeds": list(plan.seeds),
            "perSeedValues": values_by_field,
            "means": {
                field: _mean_available(values)
                for field, values in values_by_field.items()
            },
        }
        prefix = (
            "R06 模态正常控制组"
            if kind == "r06_control"
            else "R08 节点正常控制组"
        )
        metrics = tuple(
            AcceptanceMetric(
                f"{prefix}{label}", summary["means"][field],
                "记录项", None, unit,
            )
            for label, field, unit in fields
        )
    elif kind == "dropout":
        modality = plan.preset.preset_id.removeprefix("R06_")
        record = _dropout_modality_summary(payloads, modality)
        acceptance = _acceptance_result(
            percent_statistics={
                field: {"mean": values["increasePercent"]}
                for field, values in record["pooledDropoutVsControl"][
                    "incrementalCann"
                ].items()
            },
            threshold=15.0,
            coverage_comparable=_dropout_coverage_comparable(
                payloads, modality, "incrementalCann"
            ),
            primary_metric="position_rmse_m",
        )
        value = record["pooledDropoutVsControl"]["incrementalCann"][
            "position_rmse_m"
        ]["increasePercent"]
        summary = {
            "schemaVersion": "acceptance-gui-r06.v1", "modality": modality,
            "seeds": list(plan.seeds), "result": record,
            "acceptance": acceptance,
        }
        fields = (
            ("位置RMSE", "position_rmse_m", "m"),
            ("速度RMSE", "velocity_rmse_mps", "m/s"),
            ("加速度RMSE", "acceleration_rmse_mps2", "m/s²"),
        )
        metrics = tuple(
            AcceptanceMetric(
                f"R06 {modality}{label}增幅",
                record["pooledDropoutVsControl"]["incrementalCann"][field][
                    "increasePercent"
                ],
                "≤ 15%" if field == "position_rmse_m" else "扩展检查 ≤ 15%",
                bool(
                    record["pooledDropoutVsControl"]["incrementalCann"][field][
                        "increasePercent"
                    ] is not None
                    and record["pooledDropoutVsControl"]["incrementalCann"][field][
                        "increasePercent"
                    ] <= 15.0
                ),
                "%", field == "position_rmse_m",
            )
            for label, field, _unit in fields
        )
    elif kind == "failure":
        percent = _failure_percent_summary(
            payloads, "incrementalCann", "consensus"
        )
        acceptance = _acceptance_result(
            percent_statistics=percent,
            threshold=NODE_FAILURE_MAX_INCREASE_PERCENT,
            coverage_comparable=_failure_consensus_coverage_comparable(
                payloads, "incrementalCann"
            ),
            primary_metric="position_mean_error_m",
        )
        value = percent["position_mean_error_m"]["mean"]
        summary = {
            "schemaVersion": "acceptance-gui-r08.v1",
            "case": plan.preset.preset_id, "seeds": list(plan.seeds),
            "consensusFailureVsControlPercent": percent,
            "acceptance": acceptance,
        }
        fields = (
            ("平均位置误差", "position_mean_error_m"),
            ("平均速度误差", "velocity_mean_error_mps"),
            ("平均加速度误差", "acceleration_mean_error_mps2"),
        )
        metrics = tuple(
            AcceptanceMetric(
                f"R08 存活节点{label}增幅",
                percent[field]["mean"],
                "≤ 12%" if field == "position_mean_error_m" else "扩展检查 ≤ 12%",
                bool(
                    percent[field]["mean"] is not None
                    and percent[field]["mean"] <= 12.0
                ),
                "%", field == "position_mean_error_m",
            )
            for label, field in fields
        )
    elif kind == "latency":
        summary = payloads[0]
        metrics = (
            AcceptanceMetric(
                "P09 平均同步延迟", summary["mean_sync_ms"], "≤ 100 ms",
                bool(summary["threshold_met"] and summary["protocol_clean"]),
                "ms",
            ),
            AcceptanceMetric(
                "P09 P95同步延迟", summary["p95_sync_ms"], "记录项", None,
                "ms",
            ),
        )
    else:
        manifest = payloads[0]
        passed = bool(
            manifest.get("scenario_id") == "bilateral-15-friendly-15-hostile"
            and int(manifest.get("frame_count", 0)) > 0
        )
        summary = {
            "schemaVersion": "acceptance-gui-f10.v1",
            "recording": str(Path(plan.jobs[0].output_path).parent),
            "manifest": manifest,
            "automaticDataContractPassed": passed,
            "manualGuiChecklistRequired": True,
        }
        metrics = (AcceptanceMetric(
            "F10 录像数据完整性", float(manifest.get("frame_count", 0)),
            "> 0 frames", passed, "frames",
        ),)

    summary["acceptanceConfiguration"] = {
        "configurationFamily": plan.configuration_family,
        "controlFamily": plan.control_family,
        "controlPolicy": "INDEPENDENT_MATCHED_CONTROL_PER_TEST_ITEM",
        "controlReuseEnabled": False,
        "controlStorage": "EMBEDDED_IN_EACH_RESULT_FILE",
        "resultInputs": [
            job.output_path for job in plan.jobs if job.role == "result"
        ],
        "controlInputs": [
            job.output_path for job in plan.jobs
            if plan.control_family != "none" and job.role == "result"
        ],
    }
    if kind != "latency":
        summary_path = output / "summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        summary_path = Path(plan.jobs[0].output_path)
    passed_values = [
        item.passed for item in metrics
        if item.passed is not None and item.acceptance_required
    ]
    automatic_passed = bool(passed_values and all(passed_values))
    if not passed_values:
        status = "RECORDED"
    elif kind == "visualization" and automatic_passed:
        status = "PENDING_MANUAL_CHECK"
    else:
        status = "PASS" if automatic_passed else "FAIL"
    note = (
        "快速检查仅用于功能预检，不可替代正式验收。"
        if not plan.formal else "使用冻结正式配置生成。"
    )
    outcome = AcceptanceOutcome(
        title=plan.preset.name, status=status, metrics=metrics,
        summary_path=str(summary_path), note=note,
    )
    (output / "gui_outcome.json").write_text(
        json.dumps(asdict(outcome), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv_exports(output, plan, summary, metrics)
    return outcome


def describe_acceptance_plan(plan: AcceptancePlan) -> str:
    mode = "正式验收" if plan.formal else "快速检查（非正式）"
    duration = "不按仿真时长" if plan.duration_seconds is None else f"{plan.duration_seconds:g} s"
    seeds = "无随机种子" if not plan.seeds else ", ".join(map(str, plan.seeds))
    return (
        f"{plan.preset.name}\n{plan.preset.purpose}\n\n"
        f"模式：{mode}\n时长：{duration}\n随机种子：{seeds}\n"
        f"配置族：{plan.configuration_family}\n"
        f"控制组：{plan.control_family}（本测试项独立生成，不跨项目复用）\n"
        f"数值任务数：{sum(job.role == 'result' for job in plan.jobs)}\n"
        f"代表性回放：{'生成 seed 0 场景' if plan.recording_path else '无'}\n"
        f"判据：{plan.preset.requirement}\n"
        f"输出：{plan.output_directory}"
    )


def _event_window(mode: str) -> tuple[float, float]:
    return (6.0, 13.0) if mode == QUICK_MODE else (30.0, 89.0)


def _job(
    label: str, module: str, *arguments: object, output: str | Path,
    role: str = "result",
) -> AcceptanceJob:
    return AcceptanceJob(
        label=label, module=module,
        arguments=tuple(str(value) for value in arguments),
        output_path=str(output),
        role=role,
    )


def _acceptance_recording_job(
    preset_id: str, *, duration: float, output: Path,
    event_window: tuple[float, float],
) -> AcceptanceJob:
    start, end = event_window
    arguments: list[object] = [
        "--output", output, "--duration", duration, "--dt", 1.0,
        "--seed", 0, "--cann-node-limit", 0,
        "--radial-backup-profile", "none",
        "--hide-communication-topology",
    ]
    if preset_id == "P05_BASELINE":
        arguments += [
            "--tracking-observer-indices", 0,
            "--no-cann", "--no-local-unified",
        ]
    elif preset_id == "P05_FUSION":
        arguments += ["--tracking-observer-indices", 0]
    elif preset_id in {"P07_COOPERATIVE", "R08_CONTROL"}:
        arguments.remove("--hide-communication-topology")
        arguments += ["--consensus"]
    elif preset_id == "R06_CONTROL":
        # Normal-observation replay: keep all modalities enabled and do not
        # accidentally interpret CONTROL as a dropout modality.
        pass
    elif preset_id.startswith("R06_"):
        modality = preset_id.removeprefix("R06_")
        arguments += [
            "--dropout-modality", modality,
            "--dropout-start", start, "--dropout-end", end,
        ]
        if modality == "RADAR":
            profile_index = arguments.index("none")
            arguments[profile_index] = "legacy"
    elif preset_id.startswith("R08_"):
        arguments.remove("--hide-communication-topology")
        indices = R08_FIXED_INDICES.get(preset_id, R08_RANDOM_INDICES[0])
        arguments += [
            "--consensus", "--failed-observer-indices", *indices,
            "--failure-start", start, "--failure-end", end,
        ]
    return _job(
        "seed 0 representative replay",
        "experiments.run_bilateral_visualization_recording",
        *arguments, output=output / "manifest.json", role="visualization",
    )


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mean_available(values) -> float | None:
    available = [float(value) for value in values if value is not None]
    return None if not available else sum(available) / len(available)


def _write_csv_exports(output, plan, summary, metrics) -> None:
    with (output / "metrics.csv").open(
        "w", encoding="utf-8-sig", newline="",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "test_item", "metric", "value", "unit", "threshold",
            "status", "acceptance_required",
        ))
        for metric in metrics:
            status = (
                "RECORDED" if metric.passed is None
                else "PASS" if metric.passed else "FAIL"
            )
            writer.writerow((
                plan.preset.name, metric.metric, metric.value, metric.unit,
                metric.threshold, status, metric.acceptance_required,
            ))
    with (output / "summary_flat.csv").open(
        "w", encoding="utf-8-sig", newline="",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("field", "value"))
        writer.writerows(_flatten_scalars(summary))


def _flatten_scalars(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_scalars(item, name)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            name = f"{prefix}[{index}]"
            yield from _flatten_scalars(item, name)
    else:
        yield (prefix, value)
