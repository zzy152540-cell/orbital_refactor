import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from visualization.simulation_launcher import VisualizationSimulationLauncher


def test_constellation_fields_follow_selected_scenario():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = VisualizationSimulationLauncher()

    assert not window.total_satellites.isHidden()
    assert window.friendly_total_satellites.isHidden()
    assert window.hostile_total_satellites.isHidden()

    window.scenario_type.setCurrentIndex(
        window.scenario_type.findData("BILATERAL_CONFIGURABLE")
    )
    app.processEvents()
    assert window.total_satellites.isHidden()
    assert not window.friendly_total_satellites.isHidden()
    assert not window.hostile_total_satellites.isHidden()
    assert window.friendly_total_satellites.isEnabled()

    window.scenario_type.setCurrentIndex(
        window.scenario_type.findData("BILATERAL_15X15")
    )
    app.processEvents()
    assert not window.friendly_total_satellites.isHidden()
    assert window.friendly_total_satellites.value() == 15
    assert window.hostile_total_satellites.value() == 15
    assert not window.friendly_total_satellites.isEnabled()
    assert not window.hostile_total_satellites.isEnabled()

    window.close()
