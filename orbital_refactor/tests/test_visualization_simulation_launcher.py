import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from visualization.simulation_launcher import VisualizationSimulationLauncher
from visualization.acceptance_test_workflow import FORMAL_MODE, SINGLE_SEED_FULL_MODE


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
    assert window.maximum_range_km.isEnabled()
    assert window.bilateral_process_noise.isEnabled()
    assert window.radar_enabled.isEnabled()
    assert window.infrared_enabled.isEnabled()
    assert window.optical_enabled.isEnabled()
    assert window.modality_dropout.isEnabled()
    assert window.estimation_architecture.isEnabled()
    assert not window.tracking_observer_number.isEnabled()
    assert not window.tracking_target_number.isEnabled()
    window.estimation_architecture.setCurrentIndex(
        window.estimation_architecture.findData("SINGLE_OBSERVER")
    )
    app.processEvents()
    assert window.tracking_observer_number.isEnabled()
    assert window.tracking_target_number.isEnabled()
    assert not window.dropout_start_s.isEnabled()
    assert not window.dropout_duration_s.isEnabled()
    window.modality_dropout.setCurrentIndex(
        window.modality_dropout.findData("RADAR")
    )
    app.processEvents()
    assert window.dropout_start_s.isEnabled()
    assert window.dropout_duration_s.isEnabled()
    assert not window.process_noise.isEnabled()
    assert not window.absolute_dropout.isEnabled()
    assert not window.replay_window.isEnabled()
    assert not window.communication_profile.isEnabled()

    window.scenario_type.setCurrentIndex(
        window.scenario_type.findData("BILATERAL_15X15")
    )
    app.processEvents()
    assert not window.friendly_total_satellites.isHidden()
    assert window.friendly_total_satellites.value() == 15
    assert window.hostile_total_satellites.value() == 15
    assert not window.friendly_total_satellites.isEnabled()
    assert not window.hostile_total_satellites.isEnabled()
    assert not window.maximum_range_km.isEnabled()
    assert not window.bilateral_process_noise.isEnabled()
    assert not window.radar_enabled.isEnabled()
    assert not window.modality_dropout.isEnabled()
    assert not window.estimation_architecture.isEnabled()
    assert not window.tracking_observer_number.isEnabled()
    assert not window.tracking_target_number.isEnabled()
    assert window.estimation_architecture.currentData() == "MULTI_COOPERATIVE"

    window.scenario_type.setCurrentIndex(
        window.scenario_type.findData("BILATERAL_CONFIGURABLE")
    )
    app.processEvents()
    assert window.estimation_architecture.currentData() == "SINGLE_OBSERVER"
    assert window.dropout_start_s.isEnabled()

    window.close()


def test_single_seed_full_acceptance_mode_is_default():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = VisualizationSimulationLauncher()

    assert window.acceptance_mode.currentData() == SINGLE_SEED_FULL_MODE
    plan = window._build_acceptance_plan_from_ui()
    assert plan.duration_seconds == 120.0
    assert plan.seeds == (0,)

    window.acceptance_test.setCurrentIndex(
        window.acceptance_test.findData("P07_COOPERATIVE")
    )
    app.processEvents()
    window.acceptance_single_seed.setValue(4)
    plan = window._build_acceptance_plan_from_ui()
    assert plan.seeds == (4,)

    window.acceptance_mode.setCurrentIndex(
        window.acceptance_mode.findData(FORMAL_MODE)
    )
    app.processEvents()
    assert not window.acceptance_single_seed.isEnabled()
    assert window._build_acceptance_plan_from_ui().seeds == (0, 1, 2, 3, 4)

    window.close()


def test_r06_acceptance_window_defaults_and_custom_controls():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = VisualizationSimulationLauncher()
    window.acceptance_test.setCurrentIndex(
        window.acceptance_test.findData("R06_RADAR")
    )
    app.processEvents()

    assert window.acceptance_default_r06_window.isChecked()
    assert window.acceptance_dropout_start_s.value() == 30.0
    assert window.acceptance_dropout_duration_s.value() == 60.0
    assert not window.acceptance_dropout_start_s.isEnabled()

    window.acceptance_default_r06_window.setChecked(False)
    window.acceptance_dropout_start_s.setValue(4.0)
    window.acceptance_dropout_duration_s.setValue(10.0)
    app.processEvents()
    assert window.acceptance_dropout_start_s.isEnabled()
    plan = window._build_acceptance_plan_from_ui()
    assert plan.event_window_mode == "USER_CONFIGURED"
    assert plan.event_start_seconds == 4.0
    assert plan.event_end_seconds == 13.0

    window.acceptance_mode.setCurrentIndex(
        window.acceptance_mode.findData("formal")
    )
    window.acceptance_default_r06_window.setChecked(True)
    app.processEvents()
    assert window.acceptance_dropout_start_s.value() == 30.0
    assert window.acceptance_dropout_duration_s.value() == 60.0
    window.close()
