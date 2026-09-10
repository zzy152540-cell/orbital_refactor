from pathlib import Path

from experiments.run_v15_visualization_window import configure_qt_runtime


def test_qt_runtime_configuration_finds_active_environment_when_available():
    platform_path = configure_qt_runtime()
    if platform_path is not None:
        assert isinstance(platform_path, Path)
        assert platform_path.is_dir()
