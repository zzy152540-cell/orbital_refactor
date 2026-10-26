import json
from pathlib import Path

from experiments.run_interface_conformance_acceptance import (
    _http_loopback,
    run_interface_conformance,
)
from visualization.acceptance_test_workflow import (
    FORMAL_MODE,
    QUICK_MODE,
    build_acceptance_plan,
    finalize_acceptance_plan,
)


def test_interface_conformance_exercises_all_public_transports():
    result = run_interface_conformance()

    assert result["overallPassed"] is True
    assert result["transportScope"] == ["JSON+NPZ", "HTTP", "IPv4-UDP"]
    assert {item["name"] for item in result["checks"]} == {
        "PUBLIC_JSON_NPZ_ROUND_TRIP",
        "INVALID_INPUT_REJECTION",
        "SCENE_CONTROL_HTTP_LOOPBACK",
        "TRAJECTORY_UDP_LOOPBACK",
    }


def test_http_loopback_bypasses_inherited_proxy(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "")

    result = _http_loopback()

    assert result["clockState"] == "READY"
    assert result["httpStatus"] == 200


def test_interface_and_ubuntu_plans_use_dedicated_workers(tmp_path):
    interface = build_acceptance_plan("JK01", QUICK_MODE, tmp_path / "jk")
    ubuntu = build_acceptance_plan("HJ01", FORMAL_MODE, tmp_path / "hj")

    assert len(interface.jobs) == len(ubuntu.jobs) == 1
    assert interface.jobs[0].module == (
        "experiments.run_interface_conformance_acceptance"
    )
    assert ubuntu.jobs[0].module == (
        "experiments.run_ubuntu_environment_acceptance"
    )
    assert ubuntu.jobs[0].arguments[:2] == ("--mode", "formal")
    assert "--gui-launched" in ubuntu.jobs[0].arguments
    assert interface.duration_seconds is None
    assert ubuntu.duration_seconds is None
    assert interface.recording_path is None
    assert ubuntu.recording_path is None


def test_diagnostic_results_are_normalized_by_acceptance_workflow(tmp_path):
    interface = build_acceptance_plan("JK01", QUICK_MODE, tmp_path / "jk")
    interface_path = Path(interface.jobs[0].output_path)
    interface_path.parent.mkdir(parents=True)
    interface_path.write_text(json.dumps({
        "schemaVersion": "interface-conformance-acceptance.v1",
        "passedCount": 4, "checkCount": 4, "overallPassed": True,
    }), encoding="utf-8")

    ubuntu = build_acceptance_plan("HJ01", FORMAL_MODE, tmp_path / "hj")
    ubuntu_path = Path(ubuntu.jobs[0].output_path)
    ubuntu_path.parent.mkdir(parents=True)
    ubuntu_path.write_text(json.dumps({
        "schemaVersion": "ubuntu-environment-acceptance.v1",
        "passedCount": 4, "checkCount": 4, "overallPassed": False,
    }), encoding="utf-8")

    assert finalize_acceptance_plan(interface).status == "PASS"
    assert finalize_acceptance_plan(ubuntu).status == "FAIL"
