"""Open the V15 simulation-configuration and replay launcher."""

import multiprocessing
import os
from pathlib import Path
import sys
import traceback

from experiments.run_v15_visualization_window import configure_qt_runtime


def configure_application_working_directory() -> Path:
    """Use the portable bundle directory for configs and generated results."""

    if getattr(sys, "frozen", False):
        application_root = Path(sys.executable).resolve().parent
        os.chdir(application_root)
        return application_root
    return Path.cwd()


def main() -> None:
    multiprocessing.freeze_support()
    configure_application_working_directory()
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker-module":
        from experiments.run_v15_app_worker import run_worker_module
        try:
            exit_code = run_worker_module(sys.argv[2:])
        except Exception:
            error_log = Path("results/package_worker_error.log")
            error_log.parent.mkdir(parents=True, exist_ok=True)
            error_log.write_text(traceback.format_exc(), encoding="utf-8")
            raise
        raise SystemExit(exit_code)
    gui_smoke = len(sys.argv) >= 2 and sys.argv[1] == "--gui-smoke"
    configure_qt_runtime()
    from visualization.simulation_launcher import run_simulation_launcher
    raise SystemExit(run_simulation_launcher(auto_close_ms=250 if gui_smoke else None))


if __name__ == "__main__":
    main()
