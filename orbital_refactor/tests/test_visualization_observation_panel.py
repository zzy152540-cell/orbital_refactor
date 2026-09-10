import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtWidgets import QApplication

from visualization.data_contract import VisualObservation
from visualization.replay_window import ModalityObservationPanel


def test_observation_panel_distinguishes_raw_and_numeric_only_data():
    application = QApplication.instance() or QApplication([])
    panel = ModalityObservationPanel("OPTICAL")
    numeric = VisualObservation(
        observer_id="sat_01", target_id="sat_02", modality="OPTICAL",
        measurement=np.array([0.1, 0.2]),
        covariance_diagonal=np.array([1e-4, 1e-4]),
        visible=True, valid=True, processing_status="ACCEPTED", frame="BODY",
    )
    panel.set_observation(numeric)
    application.processEvents()
    assert not panel.image_plot.isVisible()
    assert "not recorded" in panel.details.toPlainText()

    raw = VisualObservation(
        observer_id="sat_01", target_id="sat_02", modality="OPTICAL",
        measurement=np.array([0.1, 0.2]),
        covariance_diagonal=np.array([1e-4, 1e-4]),
        visible=True, valid=True, processing_status="ACCEPTED", frame="BODY",
        raw_sensor_data=np.ones((16, 16)), raw_data_kind="GRAYSCALE_IMAGE",
    )
    panel.set_observation(raw)
    panel.show()
    application.processEvents()
    assert panel.image_plot.isVisible()
    assert "GRAYSCALE_IMAGE" in panel.details.toPlainText()
    panel.close()
