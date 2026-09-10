from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


_DLL_DIRECTORY_HANDLES = []


def configure_qt_runtime() -> Path | None:
    """Locate pip-installed PySide6 plugins inside the active environment."""
    package = (
        Path(sys.prefix) / "Lib" / "site-packages" / "PySide6"
        if os.name == "nt" else
        Path(sys.prefix) / "lib" / f"python{sys.version_info.major}."
        f"{sys.version_info.minor}" / "site-packages" / "PySide6"
    )
    plugins = package / "plugins"
    platforms = plugins / "platforms"
    if not (platforms / ("qwindows.dll" if os.name == "nt" else "libqxcb.so")).is_file():
        return None
    os.environ["QT_PLUGIN_PATH"] = str(plugins)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms)
    if os.name == "nt":
        os.environ["PATH"] = f"{package}{os.pathsep}{os.environ.get('PATH', '')}"
        if hasattr(os, "add_dll_directory"):
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(package)))
    return platforms


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
    configure_qt_runtime()
    from visualization.replay_window import run_replay_window
    raise SystemExit(run_replay_window(args.recording))


if __name__ == "__main__":
    main()
