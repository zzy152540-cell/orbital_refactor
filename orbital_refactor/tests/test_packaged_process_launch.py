import sys

import pytest

from visualization.process_launch import module_process_command


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
