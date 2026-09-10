from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open the V15 satellite-swarm offline replay window."
    )
    parser.add_argument(
        "recording", type=Path, nargs="?",
        default=Path(
            "results/visualization_recordings/walker20_4s_seed0_packed"
        ),
    )
    args = parser.parse_args()
    from visualization.replay_window import run_replay_window
    raise SystemExit(run_replay_window(args.recording))


if __name__ == "__main__":
    main()
