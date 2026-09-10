from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np


def benchmark_replay_window(recording: str | Path, *, offscreen=False):
    if offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from visualization.replay_model import VisualizationReplayModel
    from visualization.replay_window import VisualizationReplayWindow

    application = QApplication.instance() or QApplication([])
    model = VisualizationReplayModel(recording)
    window = VisualizationReplayWindow(model)
    window.show()
    application.processEvents()
    elapsed_ms = []
    for index in range(len(model)):
        started = perf_counter()
        window.timeline.setValue(index)
        application.processEvents()
        elapsed_ms.append(1000.0 * (perf_counter() - started))
    window.close()
    values = np.asarray(elapsed_ms)
    return {
        "frame_count": len(model),
        "mean_refresh_ms": float(np.mean(values)),
        "median_refresh_ms": float(np.median(values)),
        "p95_refresh_ms": float(np.percentile(values, 95.0)),
        "maximum_refresh_ms": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark full-frame refresh of a visualization replay."
    )
    parser.add_argument("recording", type=Path)
    parser.add_argument("--offscreen", action="store_true")
    args = parser.parse_args()
    print(json.dumps(
        benchmark_replay_window(args.recording, offscreen=args.offscreen),
        indent=2,
    ))


if __name__ == "__main__":
    main()
