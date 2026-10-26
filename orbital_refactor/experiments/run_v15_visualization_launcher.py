"""Open the V15 simulation-configuration and replay launcher."""

import multiprocessing
import sys

from experiments.run_v15_visualization_window import configure_qt_runtime


def main() -> None:
    multiprocessing.freeze_support()
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker-module":
        from experiments.run_v15_app_worker import run_worker_module
        raise SystemExit(run_worker_module(sys.argv[2:]))
    configure_qt_runtime()
    from visualization.simulation_launcher import run_simulation_launcher
    raise SystemExit(run_simulation_launcher())


if __name__ == "__main__":
    main()
