import sys

import pytest

from visualization.process_launch import module_process_command
from experiments.run_v15_app_worker import PACKAGED_WORKER_MODULES


def test_source_worker_uses_python_module_invocation(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delenv("ORBITAL_WORKER_EXECUTABLE", raising=False)

    program, arguments = module_process_command(
        "experiments.run_cross_platform_acceptance", ("--output", "result.json"),
    )

    assert program == sys.executable
    assert arguments == [
        "-m", "experiments.run_cross_platform_acceptance",
        "--output", "result.json",
    ]


def test_frozen_worker_routes_back_through_application(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.delenv("ORBITAL_WORKER_EXECUTABLE", raising=False)

    program, arguments = module_process_command(
        "experiments.run_interface_conformance_acceptance", (),
    )

    assert program == sys.executable
    assert arguments == [
        "--worker-module", "experiments.run_interface_conformance_acceptance",
    ]


def test_worker_rejects_non_experiment_module():
    with pytest.raises(ValueError, match="experiments"):
        module_process_command("os", ())


def test_dedicated_worker_executable_receives_module_directly(monkeypatch):
    monkeypatch.setenv("ORBITAL_WORKER_EXECUTABLE", "/opt/orbital/worker")

    program, arguments = module_process_command(
        "experiments.run_interface_conformance_acceptance", ("--output", "x"),
    )

    assert program.endswith("worker")
    assert arguments == [
        "experiments.run_interface_conformance_acceptance", "--output", "x",
    ]


def test_all_gui_worker_modules_are_declared_for_packaging():
    assert set(PACKAGED_WORKER_MODULES) == {
        "experiments.run_bilateral_acceptance_control",
        "experiments.run_bilateral_baseline_matrix",
        "experiments.run_bilateral_dropout_prescan",
        "experiments.run_bilateral_node_failure_prescan",
        "experiments.run_bilateral_visualization_recording",
        "experiments.run_interface_conformance_acceptance",
        "experiments.run_multitarget_distributed_node_sync_latency_acceptance",
        "experiments.run_tango_like_physical_frontend_demo",
        "experiments.run_ubuntu_environment_acceptance",
    }
