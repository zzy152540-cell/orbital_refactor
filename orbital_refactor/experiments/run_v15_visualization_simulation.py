"""Run a configured simulation, write its replay, and optionally open it."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from experiments.run_v15_dynamic_visualization_recording import (
    generate_dynamic_walker_recording,
)
from experiments.visualization_simulation_config import (
    load_visualization_simulation_config,
    next_available_recording_path,
)


def run_configured_visualization_simulation(config_path: str | Path) -> Path:
    config = load_visualization_simulation_config(config_path)
    requested = Path(config.output)
    if requested.exists():
        if not config.auto_increment_output:
            raise FileExistsError(
                "Output already exists and auto_increment_output is disabled."
            )
        config = replace(config, output=str(next_available_recording_path(requested)))
    output = generate_dynamic_walker_recording(
        Path(config.output), **config.generator_arguments,
    )
    if config.open_after_run:
        from experiments.run_v15_visualization_window import configure_qt_runtime
        configure_qt_runtime()
        from visualization.replay_window import run_replay_window
        run_replay_window(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to a v1.0 visualization-simulation JSON configuration.",
    )
    args = parser.parse_args()
    print(run_configured_visualization_simulation(args.config))


if __name__ == "__main__":
    main()
