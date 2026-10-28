"""Frozen acceptance-test plans and result normalization for the Qt launcher."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import csv
import json
import math
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
from visualization.recording import VisualizationRecordingReader


QUICK_MODE = "quick"
SINGLE_SEED_FULL_MODE = "single_seed_full"
FORMAL_MODE = "formal"
ACCEPTANCE_OUTPUT_ROOT = Path("results/acceptance_gui")
EO_RADIAL_FAILOVER_RANGE_SIGMA_M = 2.5
EO_RADIAL_FAILOVER_RANGE_RATE_SIGMA_MPS = 0.010


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
    event_window_mode: str = "NOT_APPLICABLE"
    event_start_seconds: float | None = None
    event_end_seconds: float | None = None
    event_duration_seconds: float | None = None

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
        "F03_SINGLE", "F03-1 单星多模态估计功能",
        "在冻结双边15:15场景下运行指定单节点对指定单目标的多模态估计。",
        "输出有限状态、三模态观测和CANN记录，无异常退出。",
        "single_estimation_function",
    ),
    AcceptancePreset(
        "F03_COOPERATIVE", "F03-2 多星协同估计功能",
        "在冻结双边15:15场景下运行多星Consensus-CI协同目标估计。",
        "输出有限状态、三模态、CANN及Consensus消息记录，无异常退出。",
        "cooperative_estimation_function",
    ),
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
        "三种观测模态均正常；雷达承担主径向观测，光学和红外径向代理处于备用状态。",
        "记录无模态缺失控制组位置、速度和派生加速度误差。", "r06_control",
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
        "冻结窗口内移除雷达模态，由光电径向代理按统一故障接管策略提供径向约束。",
        "位置RMSE增幅≤15%；光电径向代理为理想化验收模型，不声明物理传感器性能。", "dropout",
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
    AcceptancePreset(
        "JK01", "JK01 接口符合性测试",
        "自动检查JSON+NPZ模块接口、HTTP场景控制接口和UDP轨迹接口。",
        "接口收发、校验、异常拒绝和序列化检查全部通过。",
        "interface_conformance",
    ),
    AcceptancePreset(
        "HJ01", "HJ01 Ubuntu运行环境测试",
        "在安装后的Ubuntu系统中采集环境并执行公开接口和30星核心回归。",
        "Ubuntu环境识别、依赖加载、核心计算和回放数据生成全部通过。",
        "ubuntu_environment",
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

ACCEPTANCE_VELOCITY_REFINEMENT_ALPHA = 0.3


def make_acceptance_output_directory(
    preset_id: str, *, root: str | Path = ACCEPTANCE_OUTPUT_ROOT,
    timestamp: datetime | None = None,
) -> Path:
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return Path(root) / preset_id.lower() / stamp


def build_acceptance_plan(
    preset_id: str, mode: str, output_directory: str | Path,
    *, r06_event_start_s: float | None = None,
    r06_event_duration_s: float | None = None,
    single_seed: int = 0,
) -> AcceptancePlan:
    if preset_id not in PRESET_BY_ID:
        raise ValueError(f"Unknown acceptance preset: {preset_id}")
    if mode not in {QUICK_MODE, SINGLE_SEED_FULL_MODE, FORMAL_MODE}:
        raise ValueError(f"Unknown acceptance mode: {mode}")
    preset = PRESET_BY_ID[preset_id]
    output = Path(output_directory)
    duration = 20.0 if mode == QUICK_MODE else 120.0
    single_seed = int(single_seed)
    if single_seed not in range(5):
        raise ValueError("single_seed must be in the frozen range 0..4.")
    seeds = (
        (0, 1, 2, 3, 4) if mode == FORMAL_MODE
        else ((single_seed,) if mode == SINGLE_SEED_FULL_MODE else (0,))
    )
    jobs: list[AcceptanceJob] = []
    configuration_family = "nominal"
    control_family = "nominal"
    sidecar_recording = preset.result_kind in {
        "r06_control", "dropout", "r08_control", "failure",
    }
    is_r06 = preset.result_kind in {"r06_control", "dropout"}
    r06_window, r06_window_mode = _resolve_r06_event_window(
        mode, duration,
        start_seconds=r06_event_start_s,
        duration_seconds=r06_event_duration_s,
    )

    if preset.result_kind.startswith("baseline_"):
        acceptance_scope = {
            "baseline_p05_control": "p05-control",
            "baseline_p05_candidate": "p05-candidate",
            "baseline_p07_control": "p07-control",
            "baseline_p07_candidate": "p07-candidate",
        }[preset.result_kind]
        for seed in seeds:
            target = output / f"baseline_seed{seed}.json"
            refinement_arguments: tuple[object, ...] = ()
            if preset_id == "P05_FUSION":
                refinement_arguments = (
                    "--local-velocity-refinement-alpha",
                    ACCEPTANCE_VELOCITY_REFINEMENT_ALPHA,
                )
            elif preset_id == "P07_COOPERATIVE":
                refinement_arguments = (
                    "--consensus-velocity-refinement-alpha",
                    ACCEPTANCE_VELOCITY_REFINEMENT_ALPHA,
                )
            jobs.append(_job(
                f"independent paired baseline seed {seed}",
                "experiments.run_bilateral_baseline_matrix",
                "--duration", duration, "--seed", seed,
                "--acceptance-scope", acceptance_scope,
                *refinement_arguments,
                "--output", target,
                output=target,
            ))
    elif preset.result_kind == "r06_control":
        configuration_family = "electro-optical-failover"
        control_family = configuration_family
        start, end = r06_window
        for seed in seeds:
            target = output / f"control_seed{seed}.json"
            jobs.append(_job(
                f"R06 normal control seed {seed}",
                "experiments.run_bilateral_acceptance_control",
                "--duration", duration, "--event-start", start,
                "--event-end", end, "--seed", seed,
                "--family", "electro-optical-failover",
                *(
                    ("--recording-output", output / "recording")
                    if seed == 0 else ()
                ),
                "--output", target,
                output=target,
            ))
    elif preset.result_kind == "dropout":
        modality = preset_id.removeprefix("R06_")
        start, end = r06_window
        control_family = "electro-optical-failover"
        configuration_family = control_family
        for seed in seeds:
            target = output / f"{modality.lower()}_dropout_seed{seed}.json"
            arguments: list[object] = [
                "--duration", duration, "--dropout-start", start,
                "--dropout-end", end, "--seed", seed,
                "--acceptance-cann-only",
                "--local-unified-multimodal-update",
                "--electro-optical-radial-failover",
                "--eo-radial-range-sigma-m",
                EO_RADIAL_FAILOVER_RANGE_SIGMA_M,
                "--eo-radial-range-rate-sigma-mps",
                EO_RADIAL_FAILOVER_RANGE_RATE_SIGMA_MPS,
                "--modalities", modality,
            ]
            arguments += ["--output", target]
            if seed == 0:
                arguments += ["--recording-output", output / "recording"]
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
                "--family", "nominal", "--distributed-consensus",
                *(
                    ("--recording-output", output / "recording")
                    if seed == 0 else ()
                ),
                "--output", target,
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
                "--acceptance-cann-only",
                "--local-unified-multimodal-update",
                "--output", target,
                *(
                    ("--recording-output", output / "recording")
                    if seed == 0 else ()
                ),
                output=target,
            ))
    elif preset.result_kind in {
        "single_estimation_function", "cooperative_estimation_function",
    }:
        single = preset.result_kind == "single_estimation_function"
        configuration_family = (
            "single-observer-functional" if single
            else "multi-cooperative-functional"
        )
        control_family = "none"
        seeds = (0,)
        recording = output / "recording"
        arguments: list[object] = [
            "--output", recording, "--duration", duration,
            "--dt", 1.0, "--seed", 0, "--cann-node-limit", 0,
        ]
        if single:
            arguments += [
                "--tracking-observer-indices", 0,
                "--tracking-target-indices", 0,
                "--hide-communication-topology",
            ]
        else:
            arguments += ["--consensus"]
        jobs.append(_job(
            (
                "single observer multimodal estimation" if single
                else "multi-star cooperative estimation"
            ),
            "experiments.run_bilateral_visualization_recording",
            *arguments,
            output=recording / "manifest.json",
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
    elif preset.result_kind == "interface_conformance":
        configuration_family = "public-interface-v1"
        control_family = "none"
        target = output / "interface_conformance.json"
        jobs.append(_job(
            "public interface conformance",
            "experiments.run_interface_conformance_acceptance",
            "--output", target, output=target,
        ))
        duration = None
        seeds = ()
    elif preset.result_kind == "ubuntu_environment":
        configuration_family = "ubuntu-installed-environment"
        control_family = "none"
        target = output / "ubuntu_environment.json"
        jobs.append(_job(
            "Ubuntu environment and core regression",
            "experiments.run_ubuntu_environment_acceptance",
            "--mode", (QUICK_MODE if mode == QUICK_MODE else FORMAL_MODE),
            "--gui-launched", "--output", target,
            output=target,
        ))
        duration = None
        seeds = ()
    recording_path = None
    if (
        preset.result_kind not in {
            "latency", "visualization", "single_estimation_function",
            "cooperative_estimation_function", "interface_conformance",
            "ubuntu_environment",
        }
        and not sidecar_recording
    ):
        recording = output / "recording"
        jobs.append(_acceptance_recording_job(
            preset_id, duration=duration, output=recording,
            event_window=_event_window(mode), seed=seeds[0],
        ))
        recording_path = str(recording)
    elif preset.result_kind == "visualization":
        recording_path = str(output / "recording")
    elif preset.result_kind in {
        "single_estimation_function", "cooperative_estimation_function",
    }:
        recording_path = str(output / "recording")
    elif sidecar_recording:
        recording_path = str(output / "recording")
    return AcceptancePlan(
        preset=preset, mode=mode, output_directory=str(output),
        duration_seconds=duration, seeds=seeds, jobs=tuple(jobs),
        configuration_family=configuration_family,
        control_family=control_family,
        recording_path=recording_path,
        event_window_mode=(r06_window_mode if is_r06 else "NOT_APPLICABLE"),
        event_start_seconds=(r06_window[0] if is_r06 else None),
        event_end_seconds=(r06_window[1] if is_r06 else None),
        event_duration_seconds=(
            r06_window[1] - r06_window[0] + 1.0 if is_r06 else None
        ),
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
        if kind == "baseline_p05_control":
            control = summary["localNodeBaselines"][
                "single_node_optical_only"
            ]["fullRunMetrics"]
            metrics = (
                AcceptanceMetric(
                    "P05 基准算法位置RMSE",
                    control["position_rmse_m"]["mean"],
                    "记录项", None, "m",
                ),
                AcceptanceMetric(
                    "P05 基准算法速度RMSE",
                    control["velocity_rmse_mps"]["mean"],
                    "记录项", None, "m/s",
                ),
                AcceptanceMetric(
                    "P05 基准算法加速度RMSE（派生）",
                    control["acceleration_rmse_mps2"]["mean"],
                    "全样本记录项；共同区间降幅见试验组", None, "m/s²",
                ),
            )
        elif kind == "baseline_p05_candidate":
            p05 = summary["fourMetricEvidence"]["P05"][
                "exactMatchedEvidence"
            ]
            rmse = p05["positionRmse"]
            velocity_rmse = p05["velocityRmse"]
            acceleration_rmse = p05["accelerationRmse"]
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
                    "P05 融合算法加速度RMSE（派生、共同区间）",
                    acceleration_rmse["consensus"],
                    "记录项", None, "m/s²",
                ),
                AcceptanceMetric(
                    "P05 加速度RMSE降幅（派生、辅助判定）",
                    acceleration_rmse["improvementPercent"],
                    "辅助 ≥ 8%",
                    bool(
                        p05["accelerationCoverageComparable"]
                        and acceleration_rmse["improvementPercent"] is not None
                        and acceleration_rmse["improvementPercent"] >= 8.0
                    ),
                    "%", False,
                ),
            )
        elif kind == "baseline_p07_control":
            control = summary["localNodeBaselines"][
                "independent_15_node_joint_multimodal_cann"
            ]["fullRunMetrics"]
            metrics = (
                AcceptanceMetric(
                    "P07 多星独立估计平均位置误差",
                    control["position_mean_error_m"]["mean"],
                    "记录项", None, "m",
                ),
                AcceptanceMetric(
                    "P07 多星独立估计平均速度误差",
                    control["velocity_mean_error_mps"]["mean"],
                    "记录项", None, "m/s",
                ),
                AcceptanceMetric(
                    "P07 多星独立估计加速度RMSE（派生）",
                    control["acceleration_rmse_mps2"]["mean"],
                    "全样本记录项；共同区间降幅见试验组", None, "m/s²",
                ),
            )
        else:
            p07 = summary["fourMetricEvidence"]["P07"][
                "adjustedJointMeasurementAlgorithm"
            ]
            mean_error = p07["positionMeanError"]
            velocity_mean_error = p07["velocityMeanError"]
            acceleration_rmse = p07["accelerationRmse"]
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
                    acceleration_rmse["consensus"],
                    "共同节点-目标-连续区间", None, "m/s²",
                ),
                AcceptanceMetric(
                    "P07 加速度RMSE降幅（派生、辅助判定）",
                    acceleration_rmse["improvementPercent"],
                    "辅助 ≥ 6%",
                    bool(
                        p07["accelerationCoverageComparable"]
                        and acceleration_rmse["improvementPercent"] is not None
                        and acceleration_rmse["improvementPercent"] >= 6.0
                    ),
                    "%", False,
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
                "≤ 15%" if field == "position_rmse_m" else "辅助 ≤ 15%",
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
                "≤ 12%" if field == "position_mean_error_m" else "辅助 ≤ 12%",
                bool(
                    percent[field]["mean"] is not None
                    and percent[field]["mean"] <= 12.0
                ),
                "%", field == "position_mean_error_m",
            )
            for label, field in fields
        )
    elif kind in {
        "single_estimation_function", "cooperative_estimation_function",
    }:
        single = kind == "single_estimation_function"
        recording_path = Path(plan.jobs[0].output_path).parent
        audit = _functional_recording_audit(
            recording_path,
            expected_architecture=(
                "SINGLE_OBSERVER" if single else "MULTI_COOPERATIVE"
            ),
            require_consensus=not single,
        )
        manifest = payloads[0]
        frozen_scene = bool(
            manifest.get("scenario_id")
            == "bilateral-15-friendly-15-hostile"
        )
        passed = bool(audit["passed"] and frozen_scene)
        summary = {
            "schemaVersion": "acceptance-gui-f03.v1",
            "subtest": "F03-1" if single else "F03-2",
            "recordingAudit": audit,
            "frozenScenarioConfirmed": frozen_scene,
            "manualGuiChecklistRequired": True,
        }
        metrics = (AcceptanceMetric(
            (
                "F03-1 单星多模态估计功能" if single
                else "F03-2 多星协同估计功能"
            ),
            float(manifest.get("frame_count", 0)),
            (
                "完整有限输出" if single
                else "完整有限输出和协同消息"
            ),
            passed,
            "frames",
        ),)
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
    elif kind == "interface_conformance":
        summary = payloads[0]
        metrics = (AcceptanceMetric(
            "JK01 自动接口检查",
            float(summary.get("passedCount", 0)),
            f"= {int(summary.get('checkCount', 0))} checks",
            bool(summary.get("overallPassed")), "checks",
        ),)
    elif kind == "ubuntu_environment":
        summary = payloads[0]
        metrics = (AcceptanceMetric(
            "HJ01 Ubuntu环境和核心回归",
            float(summary.get("passedCount", 0)),
            f"= {int(summary.get('checkCount', 0))} checks",
            bool(summary.get("overallPassed")), "checks",
        ),)
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
        "eventWindow": {
            "windowMode": plan.event_window_mode,
            "dropoutStartSeconds": plan.event_start_seconds,
            "dropoutDurationSeconds": plan.event_duration_seconds,
            "dropoutEndSeconds": plan.event_end_seconds,
        },
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
    elif kind in {
        "visualization", "single_estimation_function",
        "cooperative_estimation_function",
    } and automatic_passed:
        status = "PENDING_MANUAL_CHECK"
    else:
        status = "PASS" if automatic_passed else "FAIL"
    if plan.mode == QUICK_MODE:
        note = "快速检查仅用于功能预检，不可替代正式验收。"
    elif plan.mode == SINGLE_SEED_FULL_MODE:
        note = "单种子全时长运行仅用于演示和流程检查，不可替代五种子正式验收。"
    elif plan.event_window_mode == "USER_CONFIGURED":
        note = "正式运行使用用户配置的R06窗口，不属于冻结窗口证据。"
    else:
        note = "使用冻结正式配置生成。"
    outcome = AcceptanceOutcome(
        title=plan.preset.name, status=status, metrics=metrics,
        summary_path=str(summary_path), note=note,
    )
    (output / "gui_outcome.json").write_text(
        json.dumps(asdict(outcome), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv_exports(output, plan, summary, metrics, payloads)
    return outcome


def describe_acceptance_plan(plan: AcceptancePlan) -> str:
    selected_seed = plan.seeds[0] if plan.seeds else "不适用"
    mode = {
        QUICK_MODE: "快速检查（seed 0，20 s，非正式）",
        SINGLE_SEED_FULL_MODE: (
            f"单种子全时长（seed {selected_seed}，120 s，非正式）"
        ),
        FORMAL_MODE: "正式验收（seed 0～4）",
    }[plan.mode]
    duration = "不按仿真时长" if plan.duration_seconds is None else f"{plan.duration_seconds:g} s"
    seeds = "无随机种子" if not plan.seeds else ", ".join(map(str, plan.seeds))
    window = (
        ""
        if plan.event_window_mode == "NOT_APPLICABLE" else
        f"R06窗口：{plan.event_window_mode}，"
        f"开始 {plan.event_start_seconds:g} s，"
        f"持续 {plan.event_duration_seconds:g} s，"
        f"结束采样时刻 {plan.event_end_seconds:g} s\n"
    )
    maneuver = (
        "机动设置：无机动Walker冻结场景；加速度采用二体+J2模型派生\n"
        if plan.preset.preset_id.startswith(("P05_", "P07_", "R06_", "R08_"))
        else ""
    )
    velocity_refinement = (
        "速度估计：因果二体+J2预测残差修正（alpha=0.3，"
        "残差超过0.5 m/s时重置）\n"
        if plan.preset.preset_id in {"P05_FUSION", "P07_COOPERATIVE"}
        else ""
    )
    return (
        f"{plan.preset.name}\n{plan.preset.purpose}\n\n"
        f"模式：{mode}\n时长：{duration}\n随机种子：{seeds}\n"
        f"配置族：{plan.configuration_family}\n"
        f"控制组：{plan.control_family}（本测试项独立生成，不跨项目复用）\n"
        f"{maneuver}"
        f"{velocity_refinement}"
        f"{window}"
        f"数值任务数：{sum(job.role == 'result' for job in plan.jobs)}\n"
        f"代表性回放："
        f"{'生成 seed ' + str(selected_seed) + ' 场景' if plan.recording_path and plan.seeds else '无'}\n"
        f"判据：{plan.preset.requirement}\n"
        f"输出：{plan.output_directory}"
    )


def _event_window(mode: str) -> tuple[float, float]:
    return (6.0, 13.0) if mode == QUICK_MODE else (30.0, 89.0)


def _functional_recording_audit(
    path: str | Path,
    *,
    expected_architecture: str,
    require_consensus: bool,
) -> dict:
    reader = VisualizationRecordingReader(path)
    modality_set = set()
    cann_recorded = False
    tracked_target_seen = False
    metadata = None
    for frame in reader:
        metadata = frame.metadata
        modality_set.update(item.modality for item in frame.observations)
        cann_recorded = cann_recorded or bool(frame.cann)
        tracked_target_seen = tracked_target_seen or any(
            node.valid and node.status == "HOSTILE_TARGET_TRACKED"
            for node in frame.nodes
        )
    metadata = {} if metadata is None else metadata
    diagnostics = dict(metadata.get("estimation_diagnostics", {}))
    consensus_messages = int(
        diagnostics.get("consensus_received_message_count", 0)
    )
    passed = bool(
        len(reader) > 0
        and metadata.get("estimation_architecture") == expected_architecture
        and bool(metadata.get("consensus_enabled")) == require_consensus
        and diagnostics.get("all_states_finite") is True
        and int(diagnostics.get("cann_preprocessed_observation_count", 0)) > 0
        and (not require_consensus or consensus_messages > 0)
        and modality_set == {"RADAR", "INFRARED", "OPTICAL"}
        and cann_recorded
        and tracked_target_seen
    )
    return {
        "recording": str(path),
        "estimationArchitecture": metadata.get("estimation_architecture"),
        "consensusEnabled": bool(metadata.get("consensus_enabled")),
        "modalitiesSeen": sorted(modality_set),
        "cannRecorded": cann_recorded,
        "trackedTargetSeen": tracked_target_seen,
        "diagnostics": diagnostics,
        "passed": passed,
    }


def _resolve_r06_event_window(
    mode: str,
    simulation_duration_s: float,
    *,
    start_seconds: float | None,
    duration_seconds: float | None,
) -> tuple[tuple[float, float], str]:
    if start_seconds is None and duration_seconds is None:
        return _event_window(mode), "FROZEN_DEFAULT"
    if start_seconds is None or duration_seconds is None:
        raise ValueError(
            "Custom R06 window requires both start time and duration."
        )
    start = float(start_seconds)
    duration = float(duration_seconds)
    if not math.isfinite(start) or start < 0.0:
        raise ValueError("R06 dropout start must be finite and non-negative.")
    if not math.isfinite(duration) or duration < 1.0:
        raise ValueError("R06 dropout duration must be at least 1 second.")
    if start + duration > float(simulation_duration_s) + 1e-12:
        raise ValueError("R06 dropout window exceeds the simulation duration.")
    # Acceptance runs use dt=1 s and inclusive command-line endpoints.
    return (start, start + duration - 1.0), "USER_CONFIGURED"


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
    event_window: tuple[float, float], seed: int,
) -> AcceptanceJob:
    start, end = event_window
    arguments: list[object] = [
        "--output", output, "--duration", duration, "--dt", 1.0,
        "--seed", seed, "--cann-node-limit", 0,
        "--radial-backup-profile", "none",
        "--hide-communication-topology",
    ]
    if preset_id == "P05_BASELINE":
        arguments += [
            "--tracking-observer-indices", 0,
            "--tracking-modalities", "OPTICAL",
            "--no-cann", "--no-local-unified",
        ]
    elif preset_id == "P05_FUSION":
        arguments += [
            "--tracking-observer-indices", 0,
            "--local-velocity-refinement-alpha",
            ACCEPTANCE_VELOCITY_REFINEMENT_ALPHA,
        ]
    elif preset_id in {"P07_COOPERATIVE", "R08_CONTROL"}:
        arguments.remove("--hide-communication-topology")
        arguments += ["--consensus"]
        if preset_id == "P07_COOPERATIVE":
            arguments += [
                "--consensus-velocity-refinement-alpha",
                ACCEPTANCE_VELOCITY_REFINEMENT_ALPHA,
            ]
    elif preset_id == "R06_CONTROL":
        # Normal-observation replay: keep all modalities enabled and do not
        # accidentally interpret CONTROL as a dropout modality.
        arguments += ["--electro-optical-radial-failover"]
    elif preset_id.startswith("R06_"):
        modality = preset_id.removeprefix("R06_")
        arguments += [
            "--electro-optical-radial-failover",
            "--dropout-modality", modality,
            "--dropout-start", start, "--dropout-end", end,
        ]
    elif preset_id.startswith("R08_"):
        arguments.remove("--hide-communication-topology")
        indices = R08_FIXED_INDICES.get(preset_id, R08_RANDOM_INDICES[0])
        arguments += [
            "--consensus", "--failed-observer-indices", *indices,
            "--failure-start", start, "--failure-end", end,
        ]
    return _job(
        f"seed {seed} representative replay",
        "experiments.run_bilateral_visualization_recording",
        *arguments, output=output / "manifest.json", role="visualization",
    )


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mean_available(values) -> float | None:
    available = [float(value) for value in values if value is not None]
    return None if not available else sum(available) / len(available)


def _write_csv_exports(output, plan, summary, metrics, payloads) -> None:
    with (output / "metrics.csv").open(
        "w", encoding="utf-8-sig", newline="",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "测试项目", "指标", "数值", "单位", "判定门限",
            "判定结果", "是否主验收指标",
        ))
        for metric in metrics:
            status = (
                "RECORDED" if metric.passed is None
                else "PASS" if metric.passed else "FAIL"
            )
            writer.writerow((
                plan.preset.name, metric.metric, metric.value, metric.unit,
                metric.threshold,
                {"RECORDED": "记录项", "PASS": "通过", "FAIL": "不通过"}[status],
                "是" if metric.acceptance_required else "否",
            ))
    per_seed_rows = _per_seed_report_rows(plan, payloads)
    if per_seed_rows:
        fields = tuple(dict.fromkeys(
            key for row in per_seed_rows for key in row
        ))
        with (output / "per_seed_results.csv").open(
            "w", encoding="utf-8-sig", newline="",
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(per_seed_rows)
    with (output / "summary_flat.csv").open(
        "w", encoding="utf-8-sig", newline="",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("field", "value"))
        writer.writerows(_flatten_scalars(summary))
    (output / "报告文件说明.txt").write_text(
        "metrics.csv：验收指标总览，建议首先查看。\n"
        "per_seed_results.csv：每个随机种子一行，用于检查离散性和最坏种子。\n"
        "summary.json：结构化汇总和配置来源，供复核及程序读取。\n"
        "gui_outcome.json：界面显示的最终状态。\n"
        "summary_flat.csv：完整JSON扁平展开，主要用于调试，不建议人工阅读。\n"
        "*_seed*.json：每次运行的原始结果，保留用于审计。\n"
        "增幅为负数表示试验组误差低于控制组，即结果有所改善。\n",
        encoding="utf-8",
    )


def _per_seed_report_rows(plan, payloads):
    rows = []
    for index, payload in enumerate(payloads):
        seed = plan.seeds[index] if index < len(plan.seeds) else index
        try:
            row = _per_seed_report_row(plan.preset.result_kind, payload, seed)
        except (KeyError, TypeError, ValueError, IndexError):
            row = {
                "随机种子": seed,
                "说明": "详细数据见对应原始JSON文件",
            }
        if row is not None:
            rows.append(row)
    return rows


def _per_seed_report_row(kind, payload, seed):
    if kind.startswith("baseline_"):
        return _baseline_per_seed_row(kind, payload, seed)
    if kind in {"r06_control", "r08_control"}:
        branch = (
            "phase_metrics" if kind == "r06_control"
            else "consensus_phase_metrics"
        )
        values = payload["control"]["incrementalCann"][branch]["DROPOUT"]
        fields = (
            ("位置误差", "position_rmse_m"),
            ("速度误差", "velocity_rmse_mps"),
            ("加速度误差", "acceleration_rmse_mps2"),
        ) if kind == "r06_control" else (
            ("平均位置误差", "position_mean_error_m"),
            ("平均速度误差", "velocity_mean_error_mps"),
            ("平均加速度误差", "acceleration_mean_error_mps2"),
        )
        return {
            "随机种子": seed,
            **{label: values[field] for label, field in fields},
            "判定": "记录项",
        }
    if kind == "dropout":
        modality = next(iter(payload["resultsByMissingModality"]))
        record = payload["resultsByMissingModality"][modality]
        arm = record["incrementalCann"]
        control = payload["noDropoutControl"]["incrementalCann"]
        affected_metrics = arm["phase_metrics"]["DROPOUT"]
        control_metrics = control["phase_metrics"]["DROPOUT"]
        increases = record["dropoutVsControlPercent"]["incrementalCann"]
        return _robustness_seed_row(
            seed, modality, control_metrics, affected_metrics, increases,
            final_tracks=arm["final_active_track_count"],
            missing_samples=arm["missing_truth_sample_count"],
            threshold=15.0,
        )
    if kind == "failure":
        failure = payload["threeObserverFailure"]["incrementalCann"]
        values = failure["consensus_phase_metrics"]["DROPOUT"]
        control = payload["noFailureControl"]["incrementalCann"][
            "consensus_phase_metrics"
        ]["DROPOUT"]
        increases = {
            field: _report_percent_change(values[field], control[field])
            for field in (
                "position_mean_error_m", "velocity_mean_error_mps",
                "acceleration_mean_error_mps2",
            )
        }
        position = increases["position_mean_error_m"]
        return {
            "随机种子": seed,
            "失联后平均位置误差(m)": values["position_mean_error_m"],
            "位置增幅(%)": position,
            "失联后平均速度误差(m/s)": values["velocity_mean_error_mps"],
            "速度增幅(%)": increases["velocity_mean_error_mps"],
            "失联后平均加速度误差(m/s²)": values[
                "acceleration_mean_error_mps2"
            ],
            "加速度增幅(%)": increases["acceleration_mean_error_mps2"],
            "加速度辅助判定": (
                "通过"
                if increases["acceleration_mean_error_mps2"] <= 12.0
                else "不通过"
            ),
            "判定": "通过" if position <= 12.0 else "不通过",
        }
    if kind == "latency":
        return {
            "运行": 1,
            "平均同步延迟(ms)": payload["mean_sync_ms"],
            "P95同步延迟(ms)": payload["p95_sync_ms"],
            "判定": (
                "通过" if payload["threshold_met"]
                and payload["protocol_clean"] else "不通过"
            ),
        }
    return None


def _robustness_seed_row(
    seed, modality, control, affected, increases, *, final_tracks,
    missing_samples, threshold,
):
    position = increases["position_rmse_m"]
    return {
        "随机种子": seed,
        "缺失模态": modality,
        "控制组位置RMSE(m)": control["position_rmse_m"],
        "试验组位置RMSE(m)": affected["position_rmse_m"],
        "位置增幅(%)": position,
        "控制组速度RMSE(m/s)": control["velocity_rmse_mps"],
        "试验组速度RMSE(m/s)": affected["velocity_rmse_mps"],
        "速度增幅(%)": increases["velocity_rmse_mps"],
        "控制组加速度RMSE(m/s²)": control["acceleration_rmse_mps2"],
        "试验组加速度RMSE(m/s²)": affected["acceleration_rmse_mps2"],
        "加速度增幅(%)": increases["acceleration_rmse_mps2"],
        "加速度辅助判定": (
            "通过"
            if increases["acceleration_rmse_mps2"] <= threshold
            else "不通过"
        ),
        "最终有效航迹数": final_tracks,
        "缺失真值样本数": missing_samples,
        "判定": "通过" if position <= threshold else "不通过",
    }


def _baseline_per_seed_row(kind, payload, seed):
    report = summarize_payloads([payload])
    if kind == "baseline_p05_control":
        metrics = report["localNodeBaselines"][
            "single_node_multimodal_no_cann"
        ]["fullRunMetrics"]
        return _recorded_metric_row(seed, metrics, mean_error=False)
    if kind == "baseline_p07_control":
        baselines = report["localNodeBaselines"]
        baseline_name = (
            "independent_15_node_joint_multimodal_cann"
            if "independent_15_node_joint_multimodal_cann" in baselines
            else "independent_15_node_multimodal_cann"
        )
        metrics = baselines[baseline_name]["fullRunMetrics"]
        return _recorded_metric_row(seed, metrics, mean_error=True)
    if kind == "baseline_p05_candidate":
        item = report["fourMetricEvidence"]["P05"][
            "experimentalJointMeasurementCandidate"
        ]
        position = item["positionRmse"]
        velocity = item["velocityRmse"]
        acceleration = item["accelerationRmse"]
        return {
            "随机种子": seed,
            "控制组位置RMSE(m)": position["local"],
            "试验组位置RMSE(m)": position["consensus"],
            "位置降幅(%)": position["improvementPercent"],
            "控制组速度RMSE(m/s)": velocity["local"],
            "试验组速度RMSE(m/s)": velocity["consensus"],
            "速度降幅(%)": velocity["improvementPercent"],
            "控制组加速度RMSE(m/s²)": acceleration["local"],
            "试验组加速度RMSE(m/s²)": acceleration["consensus"],
            "加速度降幅(%)": acceleration["improvementPercent"],
            "加速度辅助判定": (
                "通过" if item["accelerationCoverageComparable"]
                and acceleration["improvementPercent"] is not None
                and acceleration["improvementPercent"] >= 8.0
                else "不通过"
            ),
            "判定": (
                "通过" if item["coverageComparable"]
                and position["improvementPercent"] >= 8.0 else "不通过"
            ),
        }
    item = report["fourMetricEvidence"]["P07"][
        "adjustedJointMeasurementAlgorithm"
    ]
    position = item["positionMeanError"]
    velocity = item["velocityMeanError"]
    acceleration = item["accelerationRmse"]
    return {
        "随机种子": seed,
        "独立估计平均位置误差(m)": position["local"],
        "协同估计平均位置误差(m)": position["consensus"],
        "位置降幅(%)": position["improvementPercent"],
        "独立估计平均速度误差(m/s)": velocity["local"],
        "协同估计平均速度误差(m/s)": velocity["consensus"],
        "速度降幅(%)": velocity["improvementPercent"],
        "独立估计加速度RMSE(m/s²)": acceleration["local"],
        "协同估计加速度RMSE(m/s²)": acceleration["consensus"],
        "加速度降幅(%)": acceleration["improvementPercent"],
        "加速度辅助判定": (
            "通过" if item["accelerationCoverageComparable"]
            and acceleration["improvementPercent"] is not None
            and acceleration["improvementPercent"] >= 6.0
            else "不通过"
        ),
        "判定": (
            "通过" if item["coverageComparable"]
            and position["improvementPercent"] >= 6.0 else "不通过"
        ),
    }


def _recorded_metric_row(seed, metrics, *, mean_error):
    suffix = "mean_error" if mean_error else "rmse"
    label = "平均误差" if mean_error else "RMSE"
    return {
        "随机种子": seed,
        f"位置{label}": metrics[f"position_{suffix}_m"]["mean"],
        f"速度{label}": metrics[f"velocity_{suffix}_mps"]["mean"],
        "加速度RMSE(派生)": metrics["acceleration_rmse_mps2"]["mean"],
        "判定": "记录项",
    }


def _report_percent_change(value, baseline):
    value = float(value)
    baseline = float(baseline)
    return None if baseline == 0.0 else 100.0 * (value / baseline - 1.0)


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
