from pathlib import Path

import experiments.run_v15_visualization_launcher as launcher
import visualization.simulation_launcher as simulation_launcher


def test_source_launcher_preserves_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delattr(launcher.sys, "frozen", raising=False)

    assert launcher.configure_application_working_directory() == tmp_path
    assert Path.cwd() == tmp_path


def test_frozen_launcher_uses_executable_directory(monkeypatch, tmp_path):
    executable = tmp_path / "SatelliteStateEstimation"
    # Register the original directory so pytest restores it after the launcher
    # intentionally changes process-wide state.
    monkeypatch.chdir(tmp_path.parent)
    monkeypatch.setattr(launcher.sys, "frozen", True, raising=False)
    monkeypatch.setattr(launcher.sys, "executable", str(executable))

    assert launcher.configure_application_working_directory() == tmp_path
    assert Path.cwd() == tmp_path


def test_gui_smoke_schedules_automatic_close(monkeypatch):
    scheduled = []

    class Application:
        def exec(self):
            return 0

        def quit(self):
            pass

    class Window:
        def show(self):
            pass

    application = Application()
    monkeypatch.setattr(
        simulation_launcher.QtWidgets.QApplication,
        "instance", staticmethod(lambda: application),
    )
    monkeypatch.setattr(
        simulation_launcher, "VisualizationSimulationLauncher", Window,
    )
    monkeypatch.setattr(
        simulation_launcher.QtCore.QTimer, "singleShot",
        staticmethod(lambda delay, callback: scheduled.append((delay, callback))),
    )

    assert simulation_launcher.run_simulation_launcher(auto_close_ms=250) == 0
    assert scheduled == [(250, application.quit)]
