"""Open the V15 simulation-configuration and replay launcher."""

from experiments.run_v15_visualization_window import configure_qt_runtime


def main() -> None:
    configure_qt_runtime()
    from visualization.simulation_launcher import run_simulation_launcher
    raise SystemExit(run_simulation_launcher())


if __name__ == "__main__":
    main()
