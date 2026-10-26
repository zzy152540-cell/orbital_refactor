"""Audit Ubuntu and execute an installed-program core regression."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
import sys

import numpy as np
import scipy

from examples.run_cross_platform_smoke import run_smoke


def _os_release(path: Path = Path("/etc/os-release")) -> dict[str, str]:
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        name, value = line.split("=", 1)
        values[name] = value.strip().strip('"')
    return values


def run_ubuntu_environment_acceptance(
    output: Path, *, mode: str, gui_launched: bool = False,
) -> dict:
    release = _os_release()
    ubuntu = platform.system() == "Linux" and release.get("ID") == "ubuntu"
    checks = []

    def check(name, callback, *, skip=False):
        if skip:
            checks.append({"name": name, "passed": False, "skipped": True})
            return None
        try:
            details = callback()
        except Exception as exc:
            checks.append({
                "name": name, "passed": False,
                "errorType": type(exc).__name__, "message": str(exc),
            })
            return None
        checks.append({"name": name, "passed": True, "details": details})
        return details

    check("UBUNTU_PLATFORM", lambda: {
        "id": release["ID"], "versionId": release.get("VERSION_ID"),
        "prettyName": release.get("PRETTY_NAME"),
    } if ubuntu else (_ for _ in ()).throw(
        RuntimeError("This verdict must be produced on Ubuntu.")
    ))
    check("PUBLIC_INTERFACE_SMOKE", run_smoke, skip=not ubuntu)
    check("PYSIDE6_IMPORT", _pyside_version, skip=not ubuntu)
    check("GUI_LAUNCHER_ACTIVE", lambda: {
        "confirmedByLauncher": True,
        "display": os.environ.get("DISPLAY"),
        "waylandDisplay": os.environ.get("WAYLAND_DISPLAY"),
        "sessionType": os.environ.get("XDG_SESSION_TYPE"),
    } if gui_launched else (_ for _ in ()).throw(
        RuntimeError("Ubuntu test must be started from the GUI launcher.")
    ), skip=not ubuntu)
    recording = output.parent / "recording"
    check(
        "BILATERAL_30_SATELLITE_CORE_REGRESSION",
        lambda: _bilateral_regression(recording, mode=mode),
        skip=not ubuntu,
    )
    return {
        "schemaVersion": "ubuntu-environment-acceptance.v1",
        "applicable": ubuntu,
        "environment": {
            "platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(), "frozen": bool(
                getattr(sys, "frozen", False)
            ),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "osRelease": release,
        },
        "mode": mode, "checks": checks,
        "passedCount": sum(bool(item["passed"]) for item in checks),
        "checkCount": len(checks),
        "overallPassed": bool(
            ubuntu and checks and all(item["passed"] for item in checks)
        ),
    }


def _pyside_version() -> dict:
    import PySide6
    from PySide6 import QtCore
    return {"pyside6": PySide6.__version__, "qt": QtCore.qVersion()}


def _bilateral_regression(recording: Path, *, mode: str) -> dict:
    from experiments.run_bilateral_visualization_recording import (
        generate_bilateral_visualization_recording,
    )
    from visualization.recording import VisualizationRecordingReader

    duration = 8.0 if mode == "quick" else 20.0
    generate_bilateral_visualization_recording(
        recording, duration=duration, dt=1.0, seed=0,
        include_cann=True, cann_node_limit=1, enable_consensus=True,
        radial_backup_profile="none", retain_raw_frames=False,
    )
    reader = VisualizationRecordingReader(recording)
    frames = tuple(reader)
    modalities = sorted({
        item.modality for frame in frames for item in frame.observations
    })
    cann_recorded = any(frame.cann for frame in frames)
    passed = bool(
        frames and frames[0].scenario_id == "bilateral-15-friendly-15-hostile"
        and modalities == ["INFRARED", "OPTICAL", "RADAR"]
        and cann_recorded
    )
    if not passed:
        raise AssertionError("The 30-satellite recording audit failed.")
    return {
        "recording": str(recording), "frameCount": len(frames),
        "scenarioId": frames[0].scenario_id, "modalities": modalities,
        "cannRecorded": cann_recorded,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("quick", "formal"), default="quick")
    parser.add_argument("--gui-launched", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = run_ubuntu_environment_acceptance(
        args.output, mode=args.mode, gui_launched=args.gui_launched,
    )
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(args.output)
    # Platform mismatch and regression failures are reported through JSON so
    # the launcher can display a normal FAIL outcome.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
