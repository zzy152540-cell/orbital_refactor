"""PySide6 offline replay window for satellite-swarm visualization records."""

from __future__ import annotations

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from visualization.replay_model import VisualizationReplayModel


MODALITY_VISUAL_STYLE = {
    "RADAR": {
        "color": "#27b7d8",
        "symbol": "s",
        "description": "range-Doppler power",
        "stops": ("#03051a", "#123b6d", "#1c8ca8", "#d8f3dc"),
    },
    "INFRARED": {
        "color": "#ff7f0e",
        "symbol": "t",
        "description": "infrared intensity",
        "stops": ("#090014", "#6a176e", "#dd513a", "#fff3b0"),
    },
    "OPTICAL": {
        "color": "#9aa0a6",
        "symbol": "o",
        "description": "optical grayscale intensity",
        "stops": ("#000000", "#555555", "#bdbdbd", "#ffffff"),
    },
}


def _lookup_table(colors, size=256):
    positions = np.linspace(0.0, 1.0, len(colors))
    return pg.ColorMap(positions, colors).getLookupTable(nPts=size)


def _robust_image_levels(values) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, (1.0, 99.5))
    if not high > low:
        padding = max(abs(float(high)) * 0.05, 1.0)
        return float(low - padding), float(high + padding)
    return float(low), float(high)


class ModalityObservationPanel(QtWidgets.QWidget):
    def __init__(self, modality: str) -> None:
        super().__init__()
        self.modality = modality
        self.visual_style = MODALITY_VISUAL_STYLE.get(
            modality, MODALITY_VISUAL_STYLE["OPTICAL"],
        )
        layout = QtWidgets.QVBoxLayout(self)
        self.modality_badge = QtWidgets.QLabel(
            f"{modality} raw observation · "
            f"{self.visual_style['description']}"
        )
        self.modality_badge.setStyleSheet(
            "font-weight: 600; padding: 4px 8px; "
            f"border-left: 5px solid {self.visual_style['color']};"
        )
        layout.addWidget(self.modality_badge)
        self.image_plot = pg.PlotWidget(
            title=f"{modality} raw sensor frame",
        )
        self.image_plot.setAspectLocked(True)
        self.image_plot.hideAxis("left")
        self.image_plot.hideAxis("bottom")
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.image_item.setLookupTable(
            _lookup_table(self.visual_style["stops"])
        )
        self.image_plot.addItem(self.image_item)
        self.peak_marker = pg.ScatterPlotItem(
            symbol=self.visual_style["symbol"], size=13,
            pen=pg.mkPen("#ffffff", width=2), brush=pg.mkBrush(0, 0, 0, 0),
        )
        self.image_plot.addItem(self.peak_marker)
        self.image_plot.setMinimumHeight(180)
        layout.addWidget(self.image_plot, 1)
        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(150)
        layout.addWidget(self.details)
        self.set_observation(None)

    def set_observation(self, observation) -> None:
        if observation is None:
            self.image_item.clear()
            self.peak_marker.setData([], [])
            self.image_plot.setVisible(False)
            self.details.setPlainText(
                f"{self.modality}\nNo observation for the selected link "
                "at this epoch."
            )
            return
        raw = observation.raw_sensor_data
        self.image_plot.setVisible(raw is not None)
        if raw is not None:
            image = np.asarray(raw, dtype=float)
            self.image_item.setImage(
                image, autoLevels=False, levels=_robust_image_levels(image),
            )
            finite = np.isfinite(image)
            if image.ndim == 2 and np.any(finite):
                peak_image = np.where(finite, image, -np.inf)
                peak_row, peak_column = np.unravel_index(
                    int(np.argmax(peak_image)), image.shape,
                )
                self.peak_marker.setData(
                    [float(peak_column)], [float(peak_row)]
                )
            else:
                self.peak_marker.setData([], [])
        measurement = np.array2string(
            observation.measurement, precision=6, separator=", ",
        )
        covariance = np.array2string(
            observation.covariance_diagonal, precision=4, separator=", ",
        )
        raw_status = (
            f"{observation.raw_data_kind}, shape={raw.shape}"
            if raw is not None else
            "not recorded (numeric simulation path)"
        )
        nis = "unavailable" if observation.nis is None else f"{observation.nis:.4f}"
        self.details.setPlainText(
            f"{observation.modality}: {observation.observer_id} -> "
            f"{observation.target_id}\n"
            f"Measurement [{observation.frame}]: {measurement}\n"
            f"Covariance diagonal: {covariance}\n"
            f"Visible / valid: {observation.visible} / {observation.valid}\n"
            f"NIS: {nis}\n"
            f"Processing: {observation.processing_status}\n"
            f"Raw sensor data: {raw_status}\n"
            f"Display: {self.visual_style['description']} | "
            f"white {self.visual_style['symbol']} marks frame maximum"
        )


class CANNObservationPanel(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._view_context = None
        layout = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel(
            "Target posterior CANN: absolute orbital phase and local RT offset"
        ))
        self.view_selector = QtWidgets.QComboBox()
        self.view_selector.addItems((
            "Absolute phase ring", "Absolute ECI position", "Position history",
        ))
        self.view_selector.hide()
        controls.addStretch(1)
        layout.addLayout(controls)

        self.views = QtWidgets.QStackedWidget()
        layout.addWidget(self.views, 1)
        self.view_selector.currentIndexChanged.connect(self._change_view)

        current = QtWidgets.QWidget()
        current_layout = QtWidgets.QGridLayout(current)
        self.direction_plot = self._activity_plot(
            "CANN absolute ECI position", "ECI axis (X/Y/Z)", None,
        )
        self.direction_curve = self.direction_plot.plot(
            pen=pg.mkPen("#4aa3df", width=2), symbol="o",
        )
        self.direction_marker = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("#ff7f0e", width=2),
        )
        self.direction_plot.addItem(self.direction_marker)
        current_layout.addWidget(self.direction_plot, 0, 0)

        self.place_plot = pg.PlotWidget(title="RT place-cell CANN")
        self.place_plot.setAspectLocked(True)
        self.place_plot.showGrid(x=True, y=True, alpha=0.35)
        self.place_image = pg.ImageItem(axisOrder="row-major")
        self.place_image.setLookupTable(_lookup_table((
            "#02090b", "#063944", "#08778a", "#16afc3", "#d7fbff",
        )))
        self.place_plot.addItem(self.place_image)
        self.place_plot.setLabel(
            "bottom", "T offset", units="km",
        )
        self.place_plot.setLabel(
            "left", "R offset", units="km",
        )
        current_layout.addWidget(self.place_plot, 0, 1)

        self.radial_plot = self._activity_plot(
            "Radial Line CANN", "line neuron", None,
        )
        self.radial_curve = self.radial_plot.plot(
            pen=pg.mkPen("#e85d75", width=2),
        )
        current_layout.addWidget(self.radial_plot, 1, 0)

        self.along_plot = self._activity_plot(
            "Along-track Line CANN", "line neuron", None,
        )
        self.along_curve = self.along_plot.plot(
            pen=pg.mkPen("#55b76e", width=2),
        )
        current_layout.addWidget(self.along_plot, 1, 1)
        current_layout.setRowStretch(0, 1)
        current_layout.setRowStretch(1, 1)
        current_layout.setColumnStretch(0, 1)
        current_layout.setColumnStretch(1, 1)
        self.views.addWidget(current)

        history = QtWidgets.QWidget()
        history_layout = QtWidgets.QVBoxLayout(history)
        self.history_plot = pg.PlotWidget(
            title="CANN decoded absolute ECI position history",
        )
        self.direction_image = pg.ImageItem()
        self.history_plot.addItem(self.direction_image)
        self.direction_cursor = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("#ff7f0e", width=2),
        )
        self.history_plot.addItem(self.direction_cursor)
        self.position_history_curves = (
            self.history_plot.plot(pen=pg.mkPen("#d62728", width=2), name="X"),
            self.history_plot.plot(pen=pg.mkPen("#2ca02c", width=2), name="Y"),
            self.history_plot.plot(pen=pg.mkPen("#1f77b4", width=2), name="Z"),
        )
        self.history_plot.addLegend()
        self.history_plot.setLabel("bottom", "simulation time", units="s")
        self.history_plot.setLabel("left", "ECI position", units="km")
        history_layout.addWidget(self.history_plot)
        self.views.addWidget(history)

        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(125)
        layout.addWidget(self.details)

    @staticmethod
    def _activity_plot(title: str, bottom_label: str, units: str | None):
        plot = pg.PlotWidget(title=title)
        plot.setLabel("bottom", bottom_label, units=units)
        plot.setLabel("left", "normalized activity")
        plot.showGrid(x=True, y=True, alpha=0.18)
        plot.setYRange(0.0, 1.05, padding=0.0)
        return plot

    def _clear_activity(self) -> None:
        self.direction_curve.setData([], [])
        self.direction_marker.hide()
        self.radial_curve.setData([], [])
        self.along_curve.setData([], [])
        self.direction_image.clear()
        self.place_image.clear()
        for curve in self.position_history_curves:
            curve.clear()

    def _change_view(self, index: int) -> None:
        self.views.setCurrentIndex(1 if index == 2 else 0)
        if self._view_context is not None:
            self._show_selected_view()

    def _update_place_heatmap(self, activity):
        """Show normalized place-cell activity on a fixed display scale."""

        place_activity = np.asarray(activity, dtype=float)
        positive = np.where(
            np.isfinite(place_activity), np.maximum(place_activity, 0.0), 0.0,
        )
        total = float(np.sum(positive))
        normalized = positive / total if total > 0.0 else positive
        self.place_image.setImage(
            normalized, autoLevels=False, levels=(0.0, 0.35),
        )
        self.place_image.setRect(QtCore.QRectF(
            -0.5, -0.5,
            float(place_activity.shape[1]),
            float(place_activity.shape[0]),
        ))
        peak_row, peak_column = np.unravel_index(
            int(np.argmax(positive)), positive.shape,
        )
        self.place_plot.setXRange(
            -0.5, place_activity.shape[1] - 0.5, padding=0.02,
        )
        self.place_plot.setYRange(
            -0.5, place_activity.shape[0] - 0.5, padding=0.02,
        )
        if place_activity.shape == (3, 3):
            ticks = [[(0, "-2"), (1, "0"), (2, "+2")]]
            self.place_plot.getAxis("bottom").setTicks(ticks)
            self.place_plot.getAxis("left").setTicks(ticks)
        self.place_plot.setTitle("RT place-cell CANN")
        return peak_row, peak_column

    def set_frame(self, *, model, frame, node_id: str) -> None:
        snapshots = {
            item.representation: item for item in frame.cann
            if item.node_id == node_id and item.available
        }
        self._view_context = (model, frame, node_id, snapshots)
        self._show_selected_view()

    def _show_selected_view(self) -> None:
        model, frame, node_id, snapshots = self._view_context
        required = ("DIRECTION_RING", "RT_LINE_PAIR", "PLACE_CELL_RT")
        if any(name not in snapshots for name in required):
            self._clear_activity()
            self.direction_plot.setTitle("Orbital-phase CANN")
            self.details.setPlainText(
                "Target-state CANN is unavailable for the selected object "
                "at this epoch."
            )
            return
        self._set_absolute_phase_ring(
            model=model, frame=frame, node_id=node_id, snapshots=snapshots,
        )

    def _set_absolute_phase_ring(
        self, *, model, frame, node_id, snapshots,
    ) -> None:
        direction = snapshots["DIRECTION_RING"]
        rt = snapshots["RT_LINE_PAIR"]
        place = snapshots["PLACE_CELL_RT"]
        direction_activity = np.asarray(direction.activity, dtype=float)
        try:
            node_index = model.node_ids.index(node_id)
        except ValueError:
            node_index = 0
        node_color = pg.intColor(
            node_index, hues=max(len(model.node_ids), 1), values=1,
        )
        self.direction_curve.setPen(pg.mkPen(node_color, width=2))
        self.direction_curve.setSymbol(None)
        self.direction_marker.show()
        self.direction_plot.setTitle("Orbital-phase CANN")
        self.direction_plot.setLabel(
            "bottom", "absolute phase", units="deg",
        )
        self.direction_plot.setLabel("left", "normalized activity")
        self.direction_plot.getAxis("bottom").setTicks(None)
        self.direction_plot.disableAutoRange(axis="y")
        self.direction_plot.setYRange(0.0, 1.05, padding=0.0)
        self.direction_image.show()
        for curve in self.position_history_curves:
            curve.clear()
        direction_peak = max(float(np.max(direction_activity)), 1e-15)
        direction_axis = np.linspace(
            0.0, 360.0, direction_activity.size, endpoint=False,
        )
        self.direction_curve.setData(
            direction_axis, direction_activity / direction_peak,
        )
        self.direction_marker.setValue(
            float(np.rad2deg(direction.decoded_value[0]) % 360.0),
        )
        self.direction_plot.setXRange(0.0, 360.0, padding=0.0)

        rt_activity = np.asarray(rt.activity, dtype=float)
        radial_peak = max(float(np.max(rt_activity[0])), 1e-15)
        along_peak = max(float(np.max(rt_activity[1])), 1e-15)
        self.radial_curve.setData(rt_activity[0] / radial_peak)
        self.along_curve.setData(rt_activity[1] / along_peak)
        self.radial_plot.setXRange(0.0, rt_activity.shape[1] - 1, padding=0.0)
        self.along_plot.setXRange(0.0, rt_activity.shape[1] - 1, padding=0.0)
        peak_row, peak_column = self._update_place_heatmap(place.activity)

        phase_deg = float(np.rad2deg(direction.decoded_value[0]) % 360.0)
        peak_neuron = int(np.argmax(direction_activity))
        absolute = snapshots.get("ABSOLUTE_POSITION_ECI")
        if absolute is None:
            absolute_text = "CANN decoded ECI position: unavailable"
        else:
            position_km = np.asarray(absolute.decoded_value, dtype=float) / 1000.0
            absolute_text = (
                "CANN decoded ECI position: "
                f"X={position_km[0]:.3f}, Y={position_km[1]:.3f}, "
                f"Z={position_km[2]:.3f} km"
            )
        self.details.setPlainText(
            f"Tracked hostile target: {node_id}\n"
            f"Absolute argument of latitude: {phase_deg:.4f} deg | "
            f"peak neuron: {peak_neuron}/{direction_activity.size - 1}\n"
            f"Phase zero: ascending node | "
            f"concentration: {direction.concentration:.6f} | "
            f"width: {direction.bump_width:.6f}\n"
            f"RT decoded: R={rt.decoded_value[0]:.3f} m, "
            f"T={rt.decoded_value[1]:.3f} m | anchor age: "
            f"{rt.anchor_age:.1f} s\n"
            f"{absolute_text}\n"
            f"Input: target filter posterior | place peak: "
            f"{place.concentration:.6f} | peak cell: "
            f"R={peak_row}, T={peak_column}\n"
            f"Heatmap display: normalized activity, fixed 0-0.35 "
            f"single-hue scale (display only)\n"
            f"Status: direction={direction.status}, RT={rt.status}, "
            f"place={place.status}"
        )

    def _set_absolute_position(
        self, *, model, frame, node_id, snapshots,
    ) -> None:
        absolute = snapshots["ABSOLUTE_POSITION_ECI"]
        rt = snapshots["RT_LINE_PAIR"]
        place = snapshots["PLACE_CELL_RT"]
        position_km = np.asarray(absolute.decoded_value, dtype=float) / 1000.0
        self.direction_marker.hide()
        self.direction_plot.setTitle("CANN decoded absolute ECI position")
        self.direction_plot.setLabel("bottom", "ECI axis (X/Y/Z)")
        self.direction_plot.setLabel("left", "position", units="km")
        self.direction_plot.getAxis("bottom").setTicks([
            [(0, "X"), (1, "Y"), (2, "Z")]
        ])
        self.direction_curve.setPen(pg.mkPen("#4aa3df", width=2))
        self.direction_curve.setSymbol("o")
        self.direction_plot.enableAutoRange(axis="y", enable=True)
        self.direction_curve.setData(np.arange(3), position_km)
        self.direction_plot.setXRange(-0.25, 2.25, padding=0.0)

        rt_activity = np.asarray(rt.activity, dtype=float)
        self.radial_curve.setData(
            rt_activity[0] / max(float(np.max(rt_activity[0])), 1e-15)
        )
        self.along_curve.setData(
            rt_activity[1] / max(float(np.max(rt_activity[1])), 1e-15)
        )
        self.radial_plot.setXRange(0.0, rt_activity.shape[1] - 1, padding=0.0)
        self.along_plot.setXRange(0.0, rt_activity.shape[1] - 1, padding=0.0)
        self._update_place_heatmap(place.activity)

        history = model.cann_decoded_history(node_id, "ABSOLUTE_POSITION_ECI")
        self.direction_image.hide()
        self.history_plot.setTitle("CANN decoded absolute ECI position history")
        self.history_plot.setLabel("bottom", "simulation time", units="s")
        self.history_plot.setLabel("left", "ECI position", units="km")
        if history is not None:
            for axis, curve in enumerate(self.position_history_curves):
                curve.setData(model.timestamps, history[:, axis] / 1000.0)
        self.direction_cursor.setValue(frame.timestamp)
        conversion_error = absolute.metrics.get(
            "position_error_to_posterior_m", float("nan")
        )
        self.details.setPlainText(
            f"Node: {node_id}\n"
            f"CANN absolute ECI: X={position_km[0]:.3f} km, "
            f"Y={position_km[1]:.3f} km, Z={position_km[2]:.3f} km\n"
            f"RT decoded: R={rt.decoded_value[0]:.3f} m, "
            f"T={rt.decoded_value[1]:.3f} m | anchor age: "
            f"{rt.anchor_age:.1f} s\n"
            f"CANN-to-posterior position difference: "
            f"{conversion_error:.3f} m\n"
            f"Status: absolute={absolute.status}, RT={rt.status}, "
            f"place={place.status}"
        )


class VisualizationReplayWindow(QtWidgets.QMainWindow):
    def __init__(self, model: VisualizationReplayModel) -> None:
        super().__init__()
        application = QtWidgets.QApplication.instance()
        if application is not None:
            application.setFont(QtGui.QFont("Arial", 10))
        self.model = model
        first_frame = model.frame(0)
        role_by_id = first_frame.metadata.get("node_role_by_id", {})
        configured_target_ids = tuple(
            first_frame.metadata.get("estimation_target_ids", ())
        )
        self.target_ids = configured_target_ids or tuple(
            node.node_id for node in first_frame.nodes
            if role_by_id.get(node.node_id) == "HOSTILE_TARGET"
            or node.status.startswith("HOSTILE")
        )
        if not self.target_ids:
            self.target_ids = model.node_ids
        self._current_index = 0
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._earth_surface_km = self._build_earth_surface()
        self.setWindowTitle(
            f"Satellite Swarm State Estimation Replay - "
            f"{model.reader.manifest.scenario_id}"
        )
        self.resize(1500, 900)
        self._build_ui()
        self._set_frame(0)

    @property
    def current_index(self) -> int:
        return self._current_index

    def _build_ui(self) -> None:
        first_frame = self.model.frame(0)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        controls = QtWidgets.QHBoxLayout()
        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.clicked.connect(self._toggle_playback)
        self.step_button = QtWidgets.QPushButton("Step")
        self.step_button.clicked.connect(self._advance)
        self.target_selector = QtWidgets.QComboBox()
        self.target_selector.addItems(self.target_ids)
        preferred_target = str(
            first_frame.metadata.get("preferred_target_id", "")
        )
        if preferred_target in self.target_ids:
            self.target_selector.setCurrentText(preferred_target)
        self.target_selector.currentTextChanged.connect(self._refresh_selected_node)
        self.speed_selector = QtWidgets.QComboBox()
        self.speed_selector.addItems(("0.5x", "1x", "2x", "5x", "10x"))
        self.speed_selector.setCurrentText("1x")
        self.time_label = QtWidgets.QLabel()
        controls.addWidget(self.play_button)
        controls.addWidget(self.step_button)
        controls.addWidget(QtWidgets.QLabel("Tracked hostile target:"))
        controls.addWidget(self.target_selector)
        controls.addWidget(QtWidgets.QLabel("Speed:"))
        controls.addWidget(self.speed_selector)
        controls.addStretch(1)
        controls.addWidget(self.time_label)
        root.addLayout(controls)

        layers = QtWidgets.QHBoxLayout()
        layers.addWidget(QtWidgets.QLabel("3-D layers:"))
        layer_defaults = (
            ("earth", "Earth", True),
            ("satellites", "Satellites", True),
            ("observations", "Observation links", True),
            ("configured", "Configured topology", False),
            ("active", "Active topology", True),
            ("delivered", "Delivered flow", True),
            ("delayed", "Delayed flow", True),
            ("truth", "Truth trajectory", True),
            ("estimate", "Estimate trajectory", True),
        )
        self.layer_checkboxes = {}
        for key, label, checked in layer_defaults:
            checkbox = QtWidgets.QCheckBox(label)
            checkbox.setChecked(checked)
            checkbox.toggled.connect(self._refresh_selected_node)
            self.layer_checkboxes[key] = checkbox
            layers.addWidget(checkbox)
        layers.addStretch(1)
        root.addLayout(layers)

        self.timeline = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.timeline.setRange(0, len(self.model) - 1)
        self.timeline.valueChanged.connect(self._set_frame)
        root.addWidget(self.timeline)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)
        left = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes((950, 550))

        self.orbit_figure = Figure(figsize=(8, 6), tight_layout=True)
        self.orbit_canvas = FigureCanvasQTAgg(self.orbit_figure)
        self.orbit_axis = self.orbit_figure.add_subplot(111, projection="3d")
        left.addWidget(self.orbit_canvas)

        error_panel = QtWidgets.QWidget()
        error_layout = QtWidgets.QVBoxLayout(error_panel)
        self.error_tabs = QtWidgets.QTabWidget()
        error_layout.addWidget(self.error_tabs)
        specifications = {
            "position": ("Target position error", "position error", "m"),
            "velocity": ("Target velocity error", "velocity error", "m/s"),
            "acceleration": (
                "Target acceleration error (derived)",
                "acceleration error", "m/s²",
            ),
        }
        self.error_plots = {}
        self.node_error_curves = {}
        self.fleet_error_curves = {}
        self.time_cursors = {}
        for quantity, (title, axis_label, units) in specifications.items():
            plot = pg.PlotWidget(title=title)
            plot.setLabel("bottom", "simulation time", units="s")
            plot.setLabel("left", axis_label, units=units)
            plot.showGrid(x=True, y=True, alpha=0.3)
            selected_curve = plot.plot(
                pen=pg.mkPen("#d62728", width=2), name="selected target",
            )
            fleet_curve = plot.plot(
                pen=pg.mkPen("#1f77b4", width=2), name="target-fleet RMSE",
            )
            cursor = pg.InfiniteLine(
                angle=90, movable=False, pen=pg.mkPen("#777777"),
            )
            plot.addItem(cursor)
            plot.addLegend()
            tab_label = {
                "position": "Position error",
                "velocity": "Velocity error",
                "acceleration": "Acceleration error (derived)",
            }[quantity]
            self.error_tabs.addTab(plot, tab_label)
            self.error_plots[quantity] = plot
            self.node_error_curves[quantity] = selected_curve
            self.fleet_error_curves[quantity] = fleet_curve
            self.time_cursors[quantity] = cursor
        self.error_plot = self.error_plots["position"]
        self.node_error_curve = self.node_error_curves["position"]
        self.fleet_error_curve = self.fleet_error_curves["position"]
        self.time_cursor = self.time_cursors["position"]
        left.addWidget(error_panel)

        self.information_tabs = QtWidgets.QTabWidget()
        self.summary = QtWidgets.QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.information_tabs.addTab(self.summary, "Overview")
        self.observation_panels = {
            modality: ModalityObservationPanel(modality)
            for modality in ("RADAR", "INFRARED", "OPTICAL")
        }
        for modality, panel in self.observation_panels.items():
            self.information_tabs.addTab(panel, modality.title())
            tab_index = self.information_tabs.indexOf(panel)
            self.information_tabs.tabBar().setTabTextColor(
                tab_index,
                QtGui.QColor(MODALITY_VISUAL_STYLE[modality]["color"]),
            )
        self.cann_panel = CANNObservationPanel()
        self.information_tabs.addTab(self.cann_panel, "Target state CANN")
        right.addWidget(self.information_tabs)

        self.node_table = QtWidgets.QTableWidget()
        self.node_table.setColumnCount(6)
        self.node_table.setHorizontalHeaderLabels(
            ("Target", "Position error (m)", "Velocity error (m/s)",
             "Position 3-sigma (m)", "Track status", "CANN status")
        )
        self.node_table.horizontalHeader().setStretchLastSection(True)
        right.addWidget(self.node_table)

        self.event_table = QtWidgets.QTableWidget()
        self.event_table.setColumnCount(4)
        self.event_table.setHorizontalHeaderLabels(
            ("Time (s)", "Severity", "Event", "Description")
        )
        self.event_table.horizontalHeader().setStretchLastSection(True)
        self.event_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.event_table.cellClicked.connect(self._jump_to_event)
        self._populate_events()
        right.addWidget(self.event_table)

    def _set_frame(self, index: int) -> None:
        self._current_index = int(index)
        if self.timeline.value() != self._current_index:
            self.timeline.setValue(self._current_index)
        frame = self.model.frame(self._current_index)
        self.time_label.setText(
            f"t = {frame.timestamp:.1f} s   "
            f"epoch {frame.epoch_index + 1}/{len(self.model)}"
        )
        self._draw_orbit(frame)
        self._draw_errors(frame.timestamp)
        self._populate_status(frame)

    def _draw_orbit(self, frame) -> None:
        axis = self.orbit_axis
        axis.clear()
        positions = {
            node.node_id: (
                node.truth_state[:3] if node.truth_state is not None
                else node.estimate_state[:3]
            ) / 1000.0
            for node in frame.nodes
        }
        selected = self.target_selector.currentText()
        values = np.asarray(list(positions.values()))
        if self._layer_visible("earth"):
            self._draw_earth(axis)
        legend_handles = []
        if self._layer_visible("satellites"):
            role_groups = (
                ("FRIENDLY", "#1f77b4", "friendly observers"),
                ("HOSTILE", "#d62728", "hostile targets"),
            )
            categorized = False
            for prefix, color, label in role_groups:
                role_values = np.asarray([
                    positions[node.node_id] for node in frame.nodes
                    if node.status.startswith(prefix)
                ])
                if not role_values.size:
                    continue
                categorized = True
                axis.scatter(
                    role_values[:, 0], role_values[:, 1], role_values[:, 2],
                    s=24, color=color, label=label,
                )
                legend_handles.append(Line2D(
                    [], [], marker="o", linestyle="None", color=color,
                    label=label,
                ))
            if not categorized:
                axis.scatter(values[:, 0], values[:, 1], values[:, 2], s=22,
                             color="#1f77b4", label="satellites")
                legend_handles.append(Line2D(
                    [], [], marker="o", linestyle="None", color="#1f77b4",
                    label="satellites",
                ))
        edge_styles = {
            "CONFIGURED_TOPOLOGY": ("#9e9e9e", ":", 0.45),
            "ACTIVE_TOPOLOGY": ("#1f77b4", "-", 0.85),
            "ACTUAL_INFORMATION_FLOW": ("#2ca02c", "-", 1.0),
            "PENDING_INFORMATION_FLOW": ("#ffbf00", "--", 0.9),
            "OBSERVATION_LINK": ("#9467bd", "--", 0.78),
        }
        edge_layers = {
            "CONFIGURED_TOPOLOGY": "configured",
            "ACTIVE_TOPOLOGY": "active",
            "ACTUAL_INFORMATION_FLOW": "delivered",
            "PENDING_INFORMATION_FLOW": "delayed",
            "OBSERVATION_LINK": "observations",
        }
        edge_labels = {
            "CONFIGURED_TOPOLOGY": "configured topology",
            "ACTIVE_TOPOLOGY": "active topology",
            "ACTUAL_INFORMATION_FLOW": "delivered flow",
            "PENDING_INFORMATION_FLOW": "delayed flow",
            "OBSERVATION_LINK": "friendly -> hostile observation",
        }
        visible_edge_types = set()
        observing_source_ids = set()
        for edge in frame.edges:
            if not self._layer_visible(edge_layers.get(edge.edge_type, "active")):
                continue
            if (
                edge.edge_type == "OBSERVATION_LINK"
                and edge.target_node_id != selected
            ):
                continue
            left, right = positions[edge.source_node_id], positions[edge.target_node_id]
            color, linestyle, alpha = edge_styles.get(
                edge.edge_type, ("#7f7f7f", "--", 0.6)
            )
            if edge.status != "ACTIVE":
                color, linestyle, alpha = "#d62728", "--", 0.8
            axis.plot(
                (left[0], right[0]), (left[1], right[1]), (left[2], right[2]),
                color=color, linestyle=linestyle,
                linewidth=(
                    1.7 if edge.edge_type == "ACTUAL_INFORMATION_FLOW"
                    else 1.35 if edge.edge_type == "OBSERVATION_LINK"
                    else 0.8
                ),
                alpha=alpha,
            )
            if edge.edge_type == "OBSERVATION_LINK":
                observing_source_ids.add(edge.source_node_id)
                delta = right - left
                arrow_start = left + 0.80 * delta
                arrow_delta = 0.14 * delta
                axis.quiver(
                    arrow_start[0], arrow_start[1], arrow_start[2],
                    arrow_delta[0], arrow_delta[1], arrow_delta[2],
                    color=color, alpha=alpha, linewidth=1.35,
                    arrow_length_ratio=0.35,
                )
            visible_edge_types.add(edge.edge_type)
        if observing_source_ids and self._layer_visible("satellites"):
            observer_positions = np.asarray([
                positions[node_id] for node_id in sorted(observing_source_ids)
            ])
            axis.scatter(
                observer_positions[:, 0], observer_positions[:, 1],
                observer_positions[:, 2], s=58, facecolors="none",
                edgecolors="#9467bd", linewidths=1.5,
            )
        for edge_type in edge_styles:
            if edge_type not in visible_edge_types:
                continue
            color, linestyle, alpha = edge_styles[edge_type]
            legend_handles.append(Line2D(
                [], [], color=color, linestyle=linestyle, alpha=alpha,
                linewidth=(
                    1.7 if edge_type == "ACTUAL_INFORMATION_FLOW"
                    else 1.35 if edge_type == "OBSERVATION_LINK"
                    else 0.8
                ),
                label=edge_labels[edge_type],
            ))
        if selected:
            truth, estimate = self.model.node_position_history(selected)
            visible = slice(0, self._current_index + 1)
            if self._layer_visible("truth"):
                axis.plot(*(truth[visible].T / 1000.0), color="#17becf",
                          linewidth=1.8)
                legend_handles.append(Line2D(
                    [], [], color="#17becf", linewidth=1.8,
                    label=f"{selected} truth",
                ))
            if self._layer_visible("estimate"):
                axis.plot(*(estimate[visible].T / 1000.0), color="#d62728",
                          linestyle="--", linewidth=1.4)
                legend_handles.append(Line2D(
                    [], [], color="#d62728", linestyle="--", linewidth=1.4,
                    label=f"{selected} estimate",
                ))
            point = positions[selected]
            if self._layer_visible("satellites"):
                axis.scatter(*point, s=85, color="#ff7f0e", edgecolor="black")
        self._equalize_3d_axes(axis, values)
        axis.set_title(f"Bilateral constellation and observations of {selected}")
        axis.set_xlabel("ECI x (km)")
        axis.set_ylabel("ECI y (km)")
        axis.set_zlabel("ECI z (km)")
        if legend_handles:
            axis.legend(handles=legend_handles, loc="upper right", fontsize=7)
        self.orbit_canvas.draw_idle()

    def _layer_visible(self, key: str) -> bool:
        return self.layer_checkboxes[key].isChecked()

    @staticmethod
    def _build_earth_surface():
        radius_km = 6378.137
        longitude = np.linspace(0.0, 2.0 * np.pi, 25)
        latitude = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 13)
        longitude, latitude = np.meshgrid(longitude, latitude)
        cosine = np.cos(latitude)
        return (
            radius_km * cosine * np.cos(longitude),
            radius_km * cosine * np.sin(longitude),
            radius_km * np.sin(latitude),
        )

    def _draw_earth(self, axis) -> None:
        x, y, z = self._earth_surface_km
        axis.plot_surface(
            x, y, z, color="#4f93ce", alpha=0.38,
            edgecolor=(0.85, 0.94, 1.0, 0.28), linewidth=0.25,
            antialiased=True, shade=True,
        )

    @staticmethod
    def _equalize_3d_axes(axis, positions) -> None:
        center = np.mean(positions, axis=0)
        radius = max(float(np.ptp(positions, axis=0).max()) / 2.0, 1.0)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))

    def _draw_errors(self, timestamp: float) -> None:
        selected = self.target_selector.currentText()
        for quantity in ("position", "velocity", "acceleration"):
            node_error = self.model.node_error_history(selected, quantity)
            fleet_error = self.model.fleet_error_rmse_history(
                quantity, self.target_ids,
            )
            self.node_error_curves[quantity].setData(
                self.model.timestamps, node_error,
            )
            self.fleet_error_curves[quantity].setData(
                self.model.timestamps, fleet_error,
            )
            self.time_cursors[quantity].setValue(timestamp)

    def _refresh_error_plot(self, _index: int | None = None) -> None:
        if hasattr(self, "time_cursors"):
            self._draw_errors(self.model.frame(self._current_index).timestamp)

    def _populate_status(self, frame) -> None:
        target = self.target_selector.currentText()
        selected_observations = tuple(
            item for item in frame.observations if item.target_id == target
        )
        counts = {
            modality: sum(item.modality == modality for item in selected_observations)
            for modality in ("RADAR", "INFRARED", "OPTICAL")
        }
        fleet_rmse = frame.metadata.get("fleet_position_rmse_m")
        try:
            fleet_rmse = float(fleet_rmse)
        except (TypeError, ValueError):
            fleet_rmse = float("nan")
        fleet_rmse_text = (
            f"{fleet_rmse:.3f} m" if np.isfinite(fleet_rmse) else "N/A"
        )
        self.summary.setPlainText(
            f"Scenario: {frame.scenario_id}\n"
            f"Run: {frame.run_id}\n"
            f"Schema: {frame.schema_version}\n"
            f"Estimation architecture: "
            f"{frame.metadata.get('estimation_architecture', 'n/a')}\n"
            f"Estimation observer scope: "
            f"{frame.metadata.get('estimation_observer_scope', 'n/a')}\n"
            f"Estimation target scope: "
            f"{frame.metadata.get('estimation_target_scope', 'n/a')}\n"
            f"Current estimation phase: "
            f"{frame.metadata.get('estimation_phase', 'n/a')}\n"
            f"Displayed posterior: "
            f"{frame.metadata.get('display_estimation_scope', 'n/a')}"
            f" @ {frame.metadata.get('display_source_node_id', 'n/a')}\n"
            f"Consensus-CI enabled: "
            f"{frame.metadata.get('consensus_enabled', False)}\n"
            f"Nodes: {len(frame.nodes)}\n"
            f"Configured / active / flow edges: "
            f"{sum(item.edge_type == 'CONFIGURED_TOPOLOGY' for item in frame.edges)} / "
            f"{sum(item.edge_type == 'ACTIVE_TOPOLOGY' for item in frame.edges)} / "
            f"{sum(item.edge_type == 'ACTUAL_INFORMATION_FLOW' for item in frame.edges)}\n"
            f"Observations this epoch: {len(frame.observations)}\n"
            f"Tracked hostile target: {target or 'none'}\n"
            f"Active observing friendly nodes: "
            f"{len({item.observer_id for item in selected_observations})}\n"
            f"RADAR / INFRARED / OPTICAL: "
            f"{counts['RADAR']} / {counts['INFRARED']} / {counts['OPTICAL']}\n"
            f"Fleet position RMSE: {fleet_rmse_text}\n"
            f"Topology version: {frame.metadata.get('topology_version', 'n/a')}\n"
            f"Observation messages available / accepted / dropped / delayed: "
            f"{frame.metadata.get('transmitted_message_count', 'n/a')} / "
            f"{frame.metadata.get('delivered_message_count', 'n/a')} / "
            f"{frame.metadata.get('dropped_message_count', 'n/a')} / "
            f"{frame.metadata.get('delayed_message_count', 'n/a')}\n"
            f"Consensus messages received (whole run): "
            f"{frame.metadata.get('estimation_diagnostics', {}).get('consensus_received_message_count', 'n/a')}\n"
            f"Resynchronized links: "
            f"{frame.metadata.get('resynchronization_count', 'n/a')}"
        )
        by_modality = {}
        for modality in ("RADAR", "INFRARED", "OPTICAL"):
            candidates = [
                item for item in selected_observations
                if item.modality == modality
            ]
            if candidates:
                by_modality[modality] = max(
                    candidates,
                    key=lambda item: (item.valid, item.visible),
                )
        for modality, panel in self.observation_panels.items():
            panel.set_observation(by_modality.get(modality))
        self.cann_panel.set_frame(
            model=self.model, frame=frame, node_id=target,
        )
        navigation = {item.node_id: item for item in frame.navigation}
        targets = [
            node for node in frame.nodes if node.node_id in self.target_ids
        ]
        self.node_table.setRowCount(len(targets))
        for row, node in enumerate(targets):
            error = (
                float("nan") if node.truth_state is None or not node.valid else
                float(np.linalg.norm(node.estimate_state[:3] - node.truth_state[:3]))
            )
            velocity_error = (
                float("nan") if node.truth_state is None or not node.valid else
                float(np.linalg.norm(
                    node.estimate_state[3:6] - node.truth_state[3:6]
                ))
            )
            three_sigma = 3.0 * np.sqrt(np.sum(node.covariance_diagonal[:3]))
            nav = navigation.get(node.node_id)
            values = (
                node.node_id, f"{error:.3f}", f"{velocity_error:.5f}",
                f"{three_sigma:.3f}", node.status,
                "n/a" if nav is None else nav.status,
            )
            for column, value in enumerate(values):
                self.node_table.setItem(row, column, QtWidgets.QTableWidgetItem(value))

    def _populate_events(self) -> None:
        events = self.model.events
        self.event_table.setRowCount(len(events))
        for row, event in enumerate(events):
            values = (
                f"{event.timestamp:.1f}", event.severity,
                event.event_type, event.description,
            )
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, event.timestamp)
                self.event_table.setItem(row, column, item)

    def _jump_to_event(self, row: int, _column: int) -> None:
        item = self.event_table.item(row, 0)
        if item is not None:
            self.timeline.setValue(self.model.nearest_index(
                float(item.data(QtCore.Qt.ItemDataRole.UserRole))
            ))

    def _refresh_selected_node(self) -> None:
        self._set_frame(self._current_index)

    def _toggle_playback(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self.play_button.setText("Play")
            return
        if self._current_index == len(self.model) - 1:
            self.timeline.setValue(0)
        self._timer.start(self._timer_interval_ms())
        self.play_button.setText("Pause")

    def _timer_interval_ms(self) -> int:
        if len(self.model) < 2:
            return 1000
        speed = float(self.speed_selector.currentText().rstrip("x"))
        physical_ms = 1000.0 * float(np.median(np.diff(self.model.timestamps)))
        return max(20, round(physical_ms / speed))

    def _advance(self) -> None:
        if self._current_index >= len(self.model) - 1:
            self._timer.stop()
            self.play_button.setText("Play")
            return
        self.timeline.setValue(self._current_index + 1)


def run_replay_window(recording_path) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = VisualizationReplayWindow(VisualizationReplayModel(recording_path))
    window.show()
    return app.exec()
