"""PySide6 offline replay window for satellite-swarm visualization records."""

from __future__ import annotations

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from visualization.replay_model import VisualizationReplayModel


class ModalityObservationPanel(QtWidgets.QWidget):
    def __init__(self, modality: str) -> None:
        super().__init__()
        self.modality = modality
        layout = QtWidgets.QVBoxLayout(self)
        self.image_plot = pg.PlotWidget()
        self.image_plot.setAspectLocked(True)
        self.image_plot.hideAxis("left")
        self.image_plot.hideAxis("bottom")
        self.image_item = pg.ImageItem()
        self.image_plot.addItem(self.image_item)
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
            self.image_plot.setVisible(False)
            self.details.setPlainText(
                f"{self.modality}\nNo observation for the selected link "
                "at this epoch."
            )
            return
        raw = observation.raw_sensor_data
        self.image_plot.setVisible(raw is not None)
        if raw is not None:
            self.image_item.setImage(np.asarray(raw), autoLevels=True)
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
            f"Raw sensor data: {raw_status}"
        )


class CANNObservationPanel(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("CANN view:"))
        self.view_selector = QtWidgets.QComboBox()
        self.view_selector.addItems(("Current neural activity", "Direction history"))
        controls.addWidget(self.view_selector)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.views = QtWidgets.QStackedWidget()
        layout.addWidget(self.views, 1)
        self.view_selector.currentIndexChanged.connect(self.views.setCurrentIndex)

        current = QtWidgets.QWidget()
        current_layout = QtWidgets.QGridLayout(current)
        self.direction_plot = self._activity_plot(
            "Direction cells", "preferred orbital phase", "degree",
        )
        self.direction_curve = self.direction_plot.plot(
            pen=pg.mkPen("#4aa3df", width=2),
        )
        self.direction_marker = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("#ff7f0e", width=2),
        )
        self.direction_plot.addItem(self.direction_marker)
        current_layout.addWidget(self.direction_plot, 0, 0)

        self.place_plot = pg.PlotWidget(title="RT place-cell activity")
        self.place_plot.setAspectLocked(True)
        self.place_image = pg.ImageItem()
        self.place_plot.addItem(self.place_image)
        self.place_plot.setLabel("bottom", "along-track cell")
        self.place_plot.setLabel("left", "radial cell")
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
        self.views.addWidget(current)

        history = QtWidgets.QWidget()
        history_layout = QtWidgets.QVBoxLayout(history)
        self.history_plot = pg.PlotWidget(
            title="Direction Ring: time-neuron activity",
        )
        self.direction_image = pg.ImageItem()
        self.history_plot.addItem(self.direction_image)
        self.direction_cursor = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("#ff7f0e", width=2),
        )
        self.history_plot.addItem(self.direction_cursor)
        self.history_plot.setLabel("bottom", "simulation time", units="s")
        self.history_plot.setLabel("left", "ring neuron")
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
        self.direction_curve.clear()
        self.radial_curve.clear()
        self.along_curve.clear()
        self.direction_image.clear()
        self.place_image.clear()

    def set_frame(self, *, model, frame, node_id: str) -> None:
        snapshots = {
            item.representation: item for item in frame.cann
            if item.node_id == node_id and item.available
        }
        required = ("DIRECTION_RING", "RT_LINE_PAIR", "PLACE_CELL_RT")
        if any(name not in snapshots for name in required):
            self._clear_activity()
            self.details.setPlainText(
                "CANN navigation display is not enabled in this recording."
            )
            return
        direction = snapshots["DIRECTION_RING"]
        rt = snapshots["RT_LINE_PAIR"]
        place = snapshots["PLACE_CELL_RT"]
        direction_activity = np.asarray(direction.activity, dtype=float)
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
        self.place_image.setImage(np.asarray(place.activity), autoLevels=True)

        history = model.cann_activity_history(node_id, "DIRECTION_RING")
        if history is not None:
            image = history
            self.direction_image.setImage(
                image, autoLevels=True,
                rect=QtCore.QRectF(
                    float(model.timestamps[0]), 0.0,
                    max(float(model.timestamps[-1] - model.timestamps[0]), 1.0),
                    float(image.shape[1]),
                ),
            )
        self.direction_cursor.setValue(frame.timestamp)
        phase_deg = float(np.rad2deg(direction.decoded_value[0]))
        self.details.setPlainText(
            f"Node: {node_id}\n"
            f"Direction phase: {phase_deg:.4f} deg | "
            f"concentration: {direction.concentration:.6f} | "
            f"width: {direction.bump_width:.6f}\n"
            f"RT decoded: R={rt.decoded_value[0]:.3f} m, "
            f"T={rt.decoded_value[1]:.3f} m | anchor age: "
            f"{rt.anchor_age:.1f} s\n"
            f"Place decoded: phase={np.rad2deg(place.decoded_value[0]):.4f} deg, "
            f"R={place.decoded_value[1]:.3f} m, "
            f"T={place.decoded_value[2]:.3f} m | "
            f"peak={place.concentration:.6f}\n"
            f"Status: direction={direction.status}, RT={rt.status}, "
            f"place={place.status}"
        )


class VisualizationReplayWindow(QtWidgets.QMainWindow):
    def __init__(self, model: VisualizationReplayModel) -> None:
        super().__init__()
        application = QtWidgets.QApplication.instance()
        if application is not None:
            application.setFont(QtGui.QFont("Arial", 10))
        self.model = model
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
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        controls = QtWidgets.QHBoxLayout()
        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.clicked.connect(self._toggle_playback)
        self.step_button = QtWidgets.QPushButton("Step")
        self.step_button.clicked.connect(self._advance)
        self.node_selector = QtWidgets.QComboBox()
        self.node_selector.addItems(self.model.node_ids)
        self.node_selector.currentTextChanged.connect(self._refresh_selected_node)
        self.target_selector = QtWidgets.QComboBox()
        self.target_selector.currentTextChanged.connect(self._refresh_selected_node)
        self.speed_selector = QtWidgets.QComboBox()
        self.speed_selector.addItems(("0.5x", "1x", "2x", "5x", "10x"))
        self.speed_selector.setCurrentText("1x")
        self.time_label = QtWidgets.QLabel()
        controls.addWidget(self.play_button)
        controls.addWidget(self.step_button)
        controls.addWidget(QtWidgets.QLabel("Satellite:"))
        controls.addWidget(self.node_selector)
        controls.addWidget(QtWidgets.QLabel("Observed target:"))
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

        self.error_plot = pg.PlotWidget(title="Position estimation error")
        self.error_plot.setLabel("bottom", "simulation time", units="s")
        self.error_plot.setLabel("left", "position error", units="m")
        self.error_plot.showGrid(x=True, y=True, alpha=0.3)
        self.node_error_curve = self.error_plot.plot(
            pen=pg.mkPen("#d62728", width=2), name="selected satellite",
        )
        self.fleet_error_curve = self.error_plot.plot(
            pen=pg.mkPen("#1f77b4", width=2), name="fleet RMSE",
        )
        self.time_cursor = pg.InfiniteLine(angle=90, movable=False,
                                           pen=pg.mkPen("#333333"))
        self.error_plot.addItem(self.time_cursor)
        self.error_plot.addLegend()
        left.addWidget(self.error_plot)

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
        self.cann_panel = CANNObservationPanel()
        self.information_tabs.addTab(self.cann_panel, "CANN navigation")
        right.addWidget(self.information_tabs)

        self.node_table = QtWidgets.QTableWidget()
        self.node_table.setColumnCount(5)
        self.node_table.setHorizontalHeaderLabels(
            ("Node", "Position error (m)", "3-sigma (m)", "Abs nav", "Status")
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
        self._update_target_selector(frame)
        self.time_label.setText(
            f"t = {frame.timestamp:.1f} s   "
            f"epoch {frame.epoch_index + 1}/{len(self.model)}"
        )
        self._draw_orbit(frame)
        self._draw_errors(frame.timestamp)
        self._populate_status(frame)

    def _update_target_selector(self, frame) -> None:
        observer = self.node_selector.currentText()
        targets = sorted({
            item.target_id for item in frame.observations
            if item.observer_id == observer
        })
        previous = self.target_selector.currentText()
        self.target_selector.blockSignals(True)
        self.target_selector.clear()
        self.target_selector.addItems(targets)
        if previous in targets:
            self.target_selector.setCurrentText(previous)
        self.target_selector.blockSignals(False)

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
        selected = self.node_selector.currentText()
        values = np.asarray(list(positions.values()))
        if self._layer_visible("earth"):
            self._draw_earth(axis)
        legend_handles = []
        if self._layer_visible("satellites"):
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
        }
        edge_layers = {
            "CONFIGURED_TOPOLOGY": "configured",
            "ACTIVE_TOPOLOGY": "active",
            "ACTUAL_INFORMATION_FLOW": "delivered",
            "PENDING_INFORMATION_FLOW": "delayed",
        }
        edge_labels = {
            "CONFIGURED_TOPOLOGY": "configured topology",
            "ACTIVE_TOPOLOGY": "active topology",
            "ACTUAL_INFORMATION_FLOW": "delivered flow",
            "PENDING_INFORMATION_FLOW": "delayed flow",
        }
        visible_edge_types = set()
        for edge in frame.edges:
            if not self._layer_visible(edge_layers.get(edge.edge_type, "active")):
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
                linewidth=(1.7 if edge.edge_type == "ACTUAL_INFORMATION_FLOW" else 0.8),
                alpha=alpha,
            )
            visible_edge_types.add(edge.edge_type)
        for edge_type in edge_styles:
            if edge_type not in visible_edge_types:
                continue
            color, linestyle, alpha = edge_styles[edge_type]
            legend_handles.append(Line2D(
                [], [], color=color, linestyle=linestyle, alpha=alpha,
                linewidth=(1.7 if edge_type == "ACTUAL_INFORMATION_FLOW" else 0.8),
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
        axis.set_title("Walker constellation: configured / active / information flow")
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
        selected = self.node_selector.currentText()
        node_error = self.model.node_position_error_history(selected)
        self.node_error_curve.setData(self.model.timestamps, node_error)
        self.fleet_error_curve.setData(
            self.model.timestamps, self.model.fleet_position_rmse_history,
        )
        self.time_cursor.setValue(timestamp)

    def _populate_status(self, frame) -> None:
        selected = self.node_selector.currentText()
        target = self.target_selector.currentText()
        selected_observations = tuple(
            item for item in frame.observations if item.observer_id == selected
        )
        counts = {
            modality: sum(item.modality == modality for item in selected_observations)
            for modality in ("RADAR", "INFRARED", "OPTICAL")
        }
        self.summary.setPlainText(
            f"Scenario: {frame.scenario_id}\n"
            f"Run: {frame.run_id}\n"
            f"Schema: {frame.schema_version}\n"
            f"Nodes: {len(frame.nodes)}\n"
            f"Configured / active / flow edges: "
            f"{sum(item.edge_type == 'CONFIGURED_TOPOLOGY' for item in frame.edges)} / "
            f"{sum(item.edge_type == 'ACTIVE_TOPOLOGY' for item in frame.edges)} / "
            f"{sum(item.edge_type == 'ACTUAL_INFORMATION_FLOW' for item in frame.edges)}\n"
            f"Observations this epoch: {len(frame.observations)}\n"
            f"Selected link: {selected} -> {target or 'none'}\n"
            f"RADAR / INFRARED / OPTICAL: "
            f"{counts['RADAR']} / {counts['INFRARED']} / {counts['OPTICAL']}\n"
            f"Fleet position RMSE: "
            f"{frame.metadata.get('fleet_position_rmse_m', float('nan')):.3f} m\n"
            f"Topology version: {frame.metadata.get('topology_version', 'n/a')}\n"
            f"Sent / delivered / dropped / delayed: "
            f"{frame.metadata.get('transmitted_message_count', 'n/a')} / "
            f"{frame.metadata.get('delivered_message_count', 'n/a')} / "
            f"{frame.metadata.get('dropped_message_count', 'n/a')} / "
            f"{frame.metadata.get('delayed_message_count', 'n/a')}\n"
            f"Resynchronized links: "
            f"{frame.metadata.get('resynchronization_count', 'n/a')}"
        )
        by_modality = {
            item.modality: item for item in selected_observations
            if item.target_id == target
        }
        for modality, panel in self.observation_panels.items():
            panel.set_observation(by_modality.get(modality))
        self.cann_panel.set_frame(
            model=self.model, frame=frame, node_id=selected,
        )
        navigation = {item.node_id: item for item in frame.navigation}
        self.node_table.setRowCount(len(frame.nodes))
        for row, node in enumerate(frame.nodes):
            error = (
                float("nan") if node.truth_state is None else
                float(np.linalg.norm(node.estimate_state[:3] - node.truth_state[:3]))
            )
            three_sigma = 3.0 * np.sqrt(np.sum(node.covariance_diagonal[:3]))
            nav = navigation.get(node.node_id)
            values = (
                node.node_id, f"{error:.3f}", f"{three_sigma:.3f}",
                "yes" if nav and nav.absolute_navigation_available else "no",
                node.status,
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
