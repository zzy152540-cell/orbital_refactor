"""Regression checks for package import boundaries in a clean interpreter."""

import subprocess
import sys


def test_adapters_import_before_interfaces_without_cycle():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from adapters.module_input_adapter import adapt_module_input; "
                "from adapters.infrared_image_adapter import InfraredCameraConfig"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
