import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtWidgets import QApplication

from visualization.data_contract import VisualObservation
from visualization.replay_window import (
    CANNObservationPanel,
    MODALITY_VISUAL_STYLE,
    ModalityObservationPanel,
)


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
    assert "optical grayscale intensity" in panel.details.toPlainText()
    assert panel.visual_style["symbol"] == "o"
    assert panel.image_item.lut is not None
    panel.close()


@pytest.mark.parametrize(
    ("modality", "symbol"),
    (("RADAR", "s"), ("INFRARED", "t"), ("OPTICAL", "o")),
)
def test_observation_panels_use_distinct_modality_styles(modality, symbol):
    application = QApplication.instance() or QApplication([])
    panel = ModalityObservationPanel(modality)

    assert panel.visual_style["symbol"] == symbol
    assert panel.visual_style["color"] == MODALITY_VISUAL_STYLE[modality][
        "color"
    ]
    assert modality in panel.modality_badge.text()
    panel.close()
    application.processEvents()


def test_cann_place_heatmap_uses_fixed_single_hue_contrast():
    application = QApplication.instance() or QApplication([])
    panel = CANNObservationPanel()
    direction = SimpleNamespace(
        activity=np.array([0.1, 1.0, 0.2]), decoded_value=np.array([0.25]),
        concentration=0.8, bump_width=0.2, status="OK",
    )
    rt = SimpleNamespace(
        activity=np.vstack((np.linspace(0.1, 1.0, 8),) * 2),
        decoded_value=np.array([10.0, -20.0]), anchor_age=0.0, status="OK",
    )
    place = SimpleNamespace(
        activity=np.array([
            [0.01, 0.04, 0.01],
            [0.04, 0.25, 0.04],
            [0.01, 0.04, 0.01],
        ]),
        concentration=0.25, status="OK",
    )

    panel._set_absolute_phase_ring(
        model=SimpleNamespace(node_ids=("target-0",)),
        frame=SimpleNamespace(timestamp=0.0),
        node_id="target-0",
        snapshots={
            "DIRECTION_RING": direction,
            "RT_LINE_PAIR": rt,
            "PLACE_CELL_RT": place,
        },
    )
    application.processEvents()

    assert np.sum(panel.place_image.image) == pytest.approx(1.0)
    assert np.max(panel.place_image.image) == pytest.approx(0.25 / 0.45)
    assert panel.place_plot.getPlotItem().titleLabel.text == (
        "RT place-cell CANN"
    )
    image_rect = panel.place_image.mapRectToParent(
        panel.place_image.boundingRect()
    )
    assert image_rect.left() == pytest.approx(-0.5)
    assert image_rect.top() == pytest.approx(-0.5)
    assert image_rect.width() == pytest.approx(3.0)
    assert "fixed 0-0.35 single-hue scale" in panel.details.toPlainText()
    panel.close()
