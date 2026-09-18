"""Run the portable public-interface smoke check and print a JSON fingerprint."""

from __future__ import annotations

import json
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy

from examples.run_standard_interface import build_demo_input
from interfaces.module_serialization import (
    load_module_output,
    save_module_input,
)
from interfaces.public_api import get_public_api_info, run_module_bundle
from interfaces.state_awareness_module import StateAwarenessModule


def run_smoke() -> dict[str, object]:
    module_input = build_demo_input()
    direct = StateAwarenessModule().run(module_input)
    with tempfile.TemporaryDirectory(prefix="orbital_smoke_") as temporary:
        root = Path(temporary)
        request = save_module_input(module_input, root / "request")
        response = run_module_bundle(request, root / "response")
        restored = load_module_output(response)

    np.testing.assert_allclose(
        restored.state_output.position_estimate,
        direct.state_output.position_estimate,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        restored.state_output.covariance,
        direct.state_output.covariance,
        rtol=1e-12,
        atol=1e-12,
    )
    return {
        "ok": True,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "api": get_public_api_info(),
        "result_fingerprint": {
            "timestamp": float(restored.state_output.timestamp),
            "position_m": restored.state_output.position_estimate.tolist(),
            "velocity_mps": restored.state_output.velocity_estimate.tolist(),
            "covariance_trace": float(np.trace(restored.state_output.covariance)),
            "status": restored.runtime_status.status,
        },
    }


def main() -> int:
    try:
        payload = run_smoke()
    except Exception as exc:  # The CLI must emit a portable failure record.
        payload = {
            "ok": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
