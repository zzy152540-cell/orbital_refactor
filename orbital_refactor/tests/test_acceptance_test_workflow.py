import json
from datetime import datetime
from pathlib import Path

import visualization.acceptance_test_workflow as workflow
from visualization.acceptance_test_workflow import (
    FORMAL_MODE,
    PRESETS,
    QUICK_MODE,
    build_acceptance_plan,
    finalize_acceptance_plan,
    make_acceptance_output_directory,
)


def test_formal_p05_plan_freezes_duration_and_five_seeds(tmp_path):
    plan = build_acceptance_plan("P05_BASELINE", FORMAL_MODE, tmp_path)

    assert plan.formal
    assert plan.duration_seconds == 120.0
    assert plan.seeds == (0, 1, 2, 3, 4)
    result_jobs = tuple(job for job in plan.jobs if job.role == "result")
    recording_jobs = tuple(job for job in plan.jobs if job.role == "visualization")
    assert len(result_jobs) == 5
    assert len(recording_jobs) == 1
    assert all(
        job.module == "experiments.run_bilateral_baseline_matrix"
        for job in result_jobs
    )
    assert result_jobs[0].arguments == (
        "--duration", "120.0", "--seed", "0",
        "--output",
        str(tmp_path / "baseline_seed0.json"),
    )
    assert plan.configuration_family == "nominal"


def test_p05_and_p07_control_and_candidate_are_four_separate_options(tmp_path):
    plans = [
        build_acceptance_plan(preset_id, QUICK_MODE, tmp_path / preset_id)
        for preset_id in (
            "P05_BASELINE", "P05_FUSION",
            "P07_INDEPENDENT", "P07_COOPERATIVE",
        )
    ]

    assert [plan.preset.result_kind for plan in plans] == [
        "baseline_p05_control", "baseline_p05_candidate",
        "baseline_p07_control", "baseline_p07_candidate",
    ]
    assert all(plan.jobs[0].module == (
        "experiments.run_bilateral_baseline_matrix"
    ) for plan in plans)
    assert len({plan.jobs[0].output_path for plan in plans}) == 4

    p05_recordings = [
        next(job for job in plan.jobs if job.role == "visualization")
        for plan in plans[:2]
    ]
    assert all(
        job.arguments[
            job.arguments.index("--tracking-observer-indices") + 1
        ] == "0"
        for job in p05_recordings
    )
    assert all(
        "--hide-communication-topology" in job.arguments
        for job in p05_recordings
    )
    p07_independent_recording = next(
        job for job in plans[2].jobs if job.role == "visualization"
    )
    p07_cooperative_recording = next(
        job for job in plans[3].jobs if job.role == "visualization"
    )
    assert "--tracking-observer-indices" not in (
        p07_independent_recording.arguments
    )
    assert "--consensus" not in p07_independent_recording.arguments
    assert "--consensus" in p07_cooperative_recording.arguments
    assert "--hide-communication-topology" not in (
        p07_cooperative_recording.arguments
    )


def test_dropdown_order_explicitly_lists_control_and_experiment_groups():
    assert [preset.name for preset in PRESETS[:14]] == [
        "P05 单星基准估计对照组",
        "P05 单星多模态融合试验组",
        "P07 多星独立估计对照组",
        "P07 多星协同估计试验组",
        "R06 多模态正常观测对照组",
        "R06 光学模态缺失试验组",
        "R06 红外模态缺失试验组",
        "R06 雷达模态缺失试验组",
        "R08 多星节点正常运行对照组",
        "R08 空间分散节点失联试验组",
        "R08 拓扑相邻节点失联试验组",
        "R08 高可见性节点失联试验组",
        "R08 冻结随机组合节点失联试验组",
        "P09 多星协同状态同步延迟",
    ]


def test_r06_and_r08_control_options_run_standalone_normal_controls(tmp_path):
    for preset_id in ("R06_CONTROL", "R08_CONTROL"):
        plan = build_acceptance_plan(preset_id, QUICK_MODE, tmp_path / preset_id)
        result_jobs = tuple(job for job in plan.jobs if job.role == "result")
        assert len(result_jobs) == 1
        assert len(plan.jobs) == 2
        job = result_jobs[0]
        assert job.module == "experiments.run_bilateral_acceptance_control"
        assert job.arguments[job.arguments.index("--family") + 1] == "nominal"


def test_r06_control_replay_has_no_fault_injection(tmp_path):
    plan = build_acceptance_plan("R06_CONTROL", QUICK_MODE, tmp_path)
    recording = next(job for job in plan.jobs if job.role == "visualization")

    assert "--dropout-modality" not in recording.arguments
    assert "--failed-observer-indices" not in recording.arguments
    assert "--consensus" not in recording.arguments


def test_r06_only_radar_dropout_uses_the_frozen_radial_backup(tmp_path):
    for preset_id, modality in (
        ("R06_OPTICAL", "OPTICAL"),
        ("R06_INFRARED", "INFRARED"),
        ("R06_RADAR", "RADAR"),
    ):
        plan = build_acceptance_plan(
            preset_id, FORMAL_MODE, tmp_path / modality.lower()
        )
        result_jobs = tuple(job for job in plan.jobs if job.role == "result")
        assert len(result_jobs) == 5
        assert len(plan.jobs) == 6
        arguments = result_jobs[0].arguments

        assert "--local-unified-multimodal-update" in arguments
        assert "--distributed-consensus" not in arguments
        assert arguments[arguments.index("--modalities") + 1] == modality
        assert "--control-input" not in arguments
        if modality == "RADAR":
            assert arguments[
                arguments.index("--backup-radial-range-sigma-m") + 1
            ] == "1.0"
            assert arguments[
                arguments.index("--backup-radial-range-rate-sigma-mps") + 1
            ] == "0.0125"
        else:
            assert "--backup-radial-range-sigma-m" not in arguments
            assert "--backup-radial-range-rate-sigma-mps" not in arguments
        expected_family = "radar-backup" if modality == "RADAR" else "nominal"
        assert plan.control_family == expected_family
        assert arguments[arguments.index("--dropout-start") + 1] == "30.0"
        assert arguments[arguments.index("--dropout-end") + 1] == "89.0"


def test_r08_random_plan_uses_seed_specific_frozen_node_sets(tmp_path):
    plan = build_acceptance_plan("R08_RANDOM", FORMAL_MODE, tmp_path)
    expected = ((7, 8, 11), (6, 7, 11), (1, 3, 10), (1, 2, 10), (9, 13, 14))

    result_jobs = tuple(job for job in plan.jobs if job.role == "result")
    for job, indices in zip(result_jobs, expected, strict=True):
        start = job.arguments.index("--failed-observer-indices") + 1
        assert tuple(map(int, job.arguments[start:start + 3])) == indices


def test_quick_mode_is_one_short_nonformal_run(tmp_path):
    plan = build_acceptance_plan("R06_OPTICAL", QUICK_MODE, tmp_path)

    assert not plan.formal
    assert plan.duration_seconds == 20.0
    assert plan.seeds == (0,)
    assert len(tuple(job for job in plan.jobs if job.role == "result")) == 1
    assert len(plan.jobs) == 2
    assert plan.jobs[0].arguments[
        plan.jobs[0].arguments.index("--dropout-start") + 1
    ] == "6.0"


def test_latency_result_is_normalized_for_the_gui(tmp_path):
    plan = build_acceptance_plan("P09", QUICK_MODE, tmp_path)
    summary = Path(plan.jobs[0].output_path)
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({
        "mean_sync_ms": 12.5,
        "p95_sync_ms": 18.0,
        "threshold_met": True,
        "protocol_clean": True,
    }), encoding="utf-8")

    outcome = finalize_acceptance_plan(plan)

    assert outcome.status == "PASS"
    assert outcome.metrics[0].metric == "P09 平均同步延迟"
    assert outcome.metrics[0].value == 12.5
    assert outcome.metrics[0].passed
    assert "不可替代正式验收" in outcome.note
    assert (tmp_path / "gui_outcome.json").exists()


def test_p05_uses_the_frozen_joint_multimodal_cann_comparison(
    tmp_path, monkeypatch,
):
    plan = build_acceptance_plan("P05_FUSION", QUICK_MODE, tmp_path)
    source = Path(plan.jobs[0].output_path)
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(workflow, "summarize_payloads", lambda _payloads: {
        "fourMetricEvidence": {
            "P05": {
                "pooled": {"improvementPercent": 0.0},
                "experimentalJointMeasurementCandidate": {
                    "coverageComparable": True,
                    "positionRmse": {
                        "local": 10.0, "consensus": 1.69,
                        "improvementPercent": 83.1,
                    },
                    "velocityRmse": {
                        "local": 0.10, "consensus": 0.08,
                        "improvementPercent": 20.0,
                    },
                },
            },
        },
        "localNodeBaselines": {
            "single_node_multimodal_no_cann": {
                "fullRunMetrics": {
                    "acceleration_rmse_mps2": {"mean": 0.02},
                },
            },
        },
        "arms": {
            "single_observer_joint_multimodal_cann": {
                "fullRunMetrics": {
                    "acceleration_rmse_mps2": {"mean": 0.01},
                },
            },
        },
    })

    outcome = finalize_acceptance_plan(plan)

    assert outcome.metrics[0].value == 1.69
    assert outcome.metrics[0].passed is None
    assert outcome.metrics[1].value == 83.1
    assert outcome.metrics[1].passed


def test_p07_uses_mean_error_from_joint_multistar_comparison(
    tmp_path, monkeypatch,
):
    plan = build_acceptance_plan("P07_COOPERATIVE", QUICK_MODE, tmp_path)
    source = Path(plan.jobs[0].output_path)
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(workflow, "summarize_payloads", lambda _payloads: {
        "fourMetricEvidence": {
            "P07": {
                "adjustedJointMeasurementAlgorithm": {
                    "coverageComparable": True,
                    "positionMeanError": {
                        "local": 10.0, "consensus": 8.11,
                        "improvementPercent": 18.9,
                    },
                    "positionRmse": {"improvementPercent": 14.4},
                    "velocityMeanError": {
                        "local": 0.10, "consensus": 0.08,
                        "improvementPercent": 20.0,
                    },
                },
            },
        },
        "localNodeBaselines": {
            "independent_15_node_multimodal_cann": {
                "fullRunMetrics": {
                    "acceleration_rmse_mps2": {"mean": 0.02},
                },
            },
        },
        "arms": {
            "fixed_topology_joint_multimodal_cann": {
                "consensusFullRunMetrics": {
                    "acceleration_rmse_mps2": {"mean": 0.01},
                },
            },
        },
    })

    outcome = finalize_acceptance_plan(plan)

    assert outcome.metrics[0].metric == "P07 多星协同估计平均位置误差"
    assert outcome.metrics[0].value == 8.11
    assert outcome.metrics[1].value == 18.9
    assert outcome.metrics[1].passed


def test_normal_control_option_is_recorded_without_pass_fail(tmp_path):
    plan = build_acceptance_plan("R06_CONTROL", QUICK_MODE, tmp_path)
    target = Path(plan.jobs[0].output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "control": {
            "incrementalCann": {
                "phase_metrics": {
                    "DROPOUT": {
                        "position_rmse_m": 1.25,
                        "velocity_rmse_mps": 0.02,
                        "acceleration_rmse_mps2": 0.003,
                    },
                },
                "consensus_phase_metrics": {
                    "DROPOUT": {"position_mean_error_m": 1.0},
                },
            },
        },
    }), encoding="utf-8")

    outcome = finalize_acceptance_plan(plan)

    assert outcome.status == "RECORDED"
    assert outcome.metrics[0].value == 1.25
    assert outcome.metrics[0].passed is None
    assert len(outcome.metrics) == 3
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "summary_flat.csv").exists()


def test_visualization_result_requires_manual_check_after_data_pass(tmp_path):
    plan = build_acceptance_plan("F10", QUICK_MODE, tmp_path)
    arguments = plan.jobs[0].arguments
    assert arguments[arguments.index("--radial-backup-profile") + 1] == "none"
    assert "--consensus" in arguments
    manifest = Path(plan.jobs[0].output_path)
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({
        "scenario_id": "bilateral-15-friendly-15-hostile",
        "frame_count": 21,
    }), encoding="utf-8")

    outcome = finalize_acceptance_plan(plan)

    assert outcome.status == "PENDING_MANUAL_CHECK"
    assert outcome.metrics[0].passed
    assert Path(outcome.summary_path).exists()


def test_output_directory_is_stable_and_grouped_by_test():
    output = make_acceptance_output_directory(
        "P09", timestamp=datetime(2026, 10, 21, 12, 34, 56)
    )
    assert output.as_posix() == "results/acceptance_gui/p09/20261021_123456"
