"""Qt setup window for configured simulation and replay launch."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from experiments.run_v15_dynamic_visualization_recording import (
    generate_dynamic_walker_recording,
)
from experiments.visualization_simulation_config import (
    VisualizationSimulationConfig,
    load_visualization_simulation_config,
    next_available_recording_path,
    save_visualization_simulation_config,
)
from visualization.replay_model import VisualizationReplayModel
from visualization.replay_window import VisualizationReplayWindow


class SimulationWorker(QtCore.QThread):
    completed = QtCore.Signal(str)
    failed = QtCore.Signal(str)
    progressed = QtCore.Signal(str, int, float)

    def __init__(self, config: VisualizationSimulationConfig, parent=None):
        super().__init__(parent)
        self.config = config

    def run(self) -> None:
        try:
            output = generate_dynamic_walker_recording(
                Path(self.config.output), **self.config.generator_arguments,
                progress_callback=self._report_progress,
            )
        except Exception as exc:  # surfaced to the GUI; traceback stays in logs
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.completed.emit(str(output))

    def _report_progress(self, stage, fraction, elapsed) -> None:
        self.progressed.emit(str(stage), round(100.0 * fraction), float(elapsed))


class VisualizationSimulationLauncher(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._worker = None
        self._replay_window = None
        self.setWindowTitle("Satellite Swarm Simulation Setup")
        self.resize(760, 600)
        self._build_ui()
        self.set_config(VisualizationSimulationConfig())

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        tabs = QtWidgets.QTabWidget()
        root.addWidget(tabs, 1)

        self.total_satellites = self._integer(2, 100)
        self.plane_count = self._integer(1, 100)
        self.phasing = self._integer(0, 99)
        self.altitude_km = self._decimal(100.0, 100000.0, 1)
        self.inclination_deg = self._decimal(0.0, 180.0, 2)
        self.duration_s = self._decimal(2.0, 86400.0, 1)
        self.dt_s = self._decimal(0.1, 3600.0, 2)
        self.maximum_range_km = self._decimal(1.0, 100000.0, 1)
        self.seed = self._integer(0, 2_000_000_000)
        self.absolute_dropout = QtWidgets.QCheckBox(
            "Interrupt absolute navigation on two nodes"
        )
        self.process_noise = self._decimal(1e-12, 1.0, 12)
        self.radar_enabled = QtWidgets.QCheckBox("Enable radar measurements")
        self.infrared_enabled = QtWidgets.QCheckBox("Enable infrared measurements")
        self.optical_enabled = QtWidgets.QCheckBox("Enable optical measurements")
        self.radar_range_sigma = self._decimal(0.001, 10000.0, 3)
        self.radar_rate_sigma = self._decimal(0.0001, 1000.0, 4)
        self.infrared_sigma = self._decimal(0.0001, 90.0, 4)
        self.optical_sigma = self._decimal(1e-8, 1.0, 8)
        self.absolute_sigma = self._decimal(0.001, 10000.0, 3)
        self.replay_window = self._decimal(0.1, 10000.0, 1)
        self.max_pinned_age = self._decimal(0.1, 10000.0, 1)
        self.communication_profile = QtWidgets.QComboBox()
        self.communication_profile.addItems(
            ("nominal", "mild", "moderate", "aggressive")
        )
        self.link_suspension = QtWidgets.QCheckBox(
            "Suspend one persistent link during the severe phase"
        )
        self.cann_enabled = QtWidgets.QCheckBox("Record navigation-cell states")
        self.cann_node_limit = self._integer(0, 100)
        self.cann_node_limit.setSpecialValueText("All nodes")
        self.open_after_run = QtWidgets.QCheckBox("Open replay after simulation")
        self.auto_increment_output = QtWidgets.QCheckBox(
            "Automatically create a new directory when output exists"
        )
        self.output = QtWidgets.QLineEdit()
        output_row = QtWidgets.QHBoxLayout()
        output_row.addWidget(self.output, 1)
        browse_output = QtWidgets.QPushButton("Browse...")
        browse_output.clicked.connect(self._browse_output)
        output_row.addWidget(browse_output)

        self._add_tab(tabs, "Constellation", (
            ("Total satellites", self.total_satellites),
            ("Orbital planes", self.plane_count),
            ("Walker phasing", self.phasing),
            ("Altitude (km)", self.altitude_km),
            ("Inclination (deg)", self.inclination_deg),
        ))
        self._add_tab(tabs, "Simulation & dynamics", (
            ("Duration (s)", self.duration_s),
            ("Time step (s)", self.dt_s),
            ("Random seed", self.seed),
            ("Navigation fault", self.absolute_dropout),
            ("Process acceleration noise", self.process_noise),
        ))
        self._add_tab(tabs, "Measurements & visibility", (
            ("Radar", self.radar_enabled),
            ("Infrared", self.infrared_enabled),
            ("Optical", self.optical_enabled),
            ("Maximum link range (km)", self.maximum_range_km),
            ("Radar range sigma (m)", self.radar_range_sigma),
            ("Radar range-rate sigma (m/s)", self.radar_rate_sigma),
            ("Infrared angle sigma (deg)", self.infrared_sigma),
            ("Optical image sigma", self.optical_sigma),
        ))
        self._add_tab(tabs, "Filter & integrity", (
            ("Absolute navigation sigma (m)", self.absolute_sigma),
            ("Replay history window (s)", self.replay_window),
            ("Maximum pinned age (s)", self.max_pinned_age),
            ("Integrity policy", QtWidgets.QLabel("Validated baseline (fixed)")),
        ))
        self._add_tab(tabs, "Communication & topology", (
            ("Degradation profile", self.communication_profile),
            ("Topology", QtWidgets.QLabel("Walker persistent topology")),
            ("Fault injection", self.link_suspension),
        ))
        self._add_tab(tabs, "CANN & output", (
            ("CANN", self.cann_enabled),
            ("CANN node limit", self.cann_node_limit),
            ("Recording output", output_row),
            ("Existing output", self.auto_increment_output),
            ("After run", self.open_after_run),
        ))

        buttons = QtWidgets.QHBoxLayout()
        self.load_button = QtWidgets.QPushButton("Load config")
        self.save_button = QtWidgets.QPushButton("Save config")
        self.open_button = QtWidgets.QPushButton("Open recording")
        self.run_button = QtWidgets.QPushButton("Run simulation")
        self.load_button.clicked.connect(self._load_config)
        self.save_button.clicked.connect(self._save_config)
        self.open_button.clicked.connect(self._open_recording_dialog)
        self.run_button.clicked.connect(self._run_simulation)
        for button in (
            self.load_button, self.save_button, self.open_button, self.run_button,
        ):
            buttons.addWidget(button)
        root.addLayout(buttons)
        self.status = QtWidgets.QPlainTextEdit()
        self.status.setReadOnly(True)
        self.status.setMaximumHeight(110)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        root.addWidget(self.progress_bar)
        root.addWidget(self.status)

    @staticmethod
    def _integer(minimum, maximum):
        widget = QtWidgets.QSpinBox()
        widget.setRange(minimum, maximum)
        return widget

    @staticmethod
    def _decimal(minimum, maximum, decimals):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        return widget

    @staticmethod
    def _add_tab(tabs, title, fields):
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        for label, field in fields:
            form.addRow(label, field)
        form.addItem(QtWidgets.QSpacerItem(
            1, 1, QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Expanding,
        ))
        tabs.addTab(page, title)

    def config(self) -> VisualizationSimulationConfig:
        return VisualizationSimulationConfig(
            total_satellites=self.total_satellites.value(),
            plane_count=self.plane_count.value(), phasing=self.phasing.value(),
            altitude_km=self.altitude_km.value(),
            inclination_deg=self.inclination_deg.value(),
            duration_s=self.duration_s.value(), dt_s=self.dt_s.value(),
            maximum_range_km=self.maximum_range_km.value(),
            seed=self.seed.value(),
            enable_absolute_navigation_dropout=self.absolute_dropout.isChecked(),
            process_noise_acceleration=self.process_noise.value(),
            radar_enabled=self.radar_enabled.isChecked(),
            infrared_enabled=self.infrared_enabled.isChecked(),
            optical_enabled=self.optical_enabled.isChecked(),
            radar_range_sigma_m=self.radar_range_sigma.value(),
            radar_range_rate_sigma_mps=self.radar_rate_sigma.value(),
            infrared_angle_sigma_deg=self.infrared_sigma.value(),
            optical_image_sigma=self.optical_sigma.value(),
            absolute_navigation_sigma_m=self.absolute_sigma.value(),
            replay_history_window_s=self.replay_window.value(),
            max_pinned_age_s=self.max_pinned_age.value(),
            communication_profile=self.communication_profile.currentText(),
            enable_link_suspension=self.link_suspension.isChecked(),
            cann_enabled=self.cann_enabled.isChecked(),
            cann_node_limit=self.cann_node_limit.value(),
            output=self.output.text().strip(),
            open_after_run=self.open_after_run.isChecked(),
            auto_increment_output=self.auto_increment_output.isChecked(),
        )

    def set_config(self, config: VisualizationSimulationConfig) -> None:
        self.total_satellites.setValue(config.total_satellites)
        self.plane_count.setValue(config.plane_count)
        self.phasing.setValue(config.phasing)
        self.altitude_km.setValue(config.altitude_km)
        self.inclination_deg.setValue(config.inclination_deg)
        self.duration_s.setValue(config.duration_s)
        self.dt_s.setValue(config.dt_s)
        self.maximum_range_km.setValue(config.maximum_range_km)
        self.seed.setValue(config.seed)
        self.absolute_dropout.setChecked(config.enable_absolute_navigation_dropout)
        self.process_noise.setValue(config.process_noise_acceleration)
        self.radar_enabled.setChecked(config.radar_enabled)
        self.infrared_enabled.setChecked(config.infrared_enabled)
        self.optical_enabled.setChecked(config.optical_enabled)
        self.radar_range_sigma.setValue(config.radar_range_sigma_m)
        self.radar_rate_sigma.setValue(config.radar_range_rate_sigma_mps)
        self.infrared_sigma.setValue(config.infrared_angle_sigma_deg)
        self.optical_sigma.setValue(config.optical_image_sigma)
        self.absolute_sigma.setValue(config.absolute_navigation_sigma_m)
        self.replay_window.setValue(config.replay_history_window_s)
        self.max_pinned_age.setValue(config.max_pinned_age_s)
        self.communication_profile.setCurrentText(config.communication_profile)
        self.link_suspension.setChecked(config.enable_link_suspension)
        self.cann_enabled.setChecked(config.cann_enabled)
        self.cann_node_limit.setValue(config.cann_node_limit)
        self.output.setText(config.output)
        self.open_after_run.setChecked(config.open_after_run)
        self.auto_increment_output.setChecked(config.auto_increment_output)

    def _load_config(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load simulation configuration", "configs", "JSON (*.json)"
        )
        if not path:
            return
        try:
            self.set_config(load_visualization_simulation_config(path))
            self.status.setPlainText(f"Loaded configuration: {path}")
        except Exception as exc:
            self._show_error(exc)

    def _save_config(self) -> None:
        try:
            config = self.config()
        except Exception as exc:
            self._show_error(exc)
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save simulation configuration", "configs/visualization.json",
            "JSON (*.json)",
        )
        if path:
            save_visualization_simulation_config(config, path)
            self.status.setPlainText(f"Saved configuration: {path}")

    def _browse_output(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select recording parent directory", "results"
        )
        if path:
            self.output.setText(str(Path(path) / "visualization_recording"))

    def _open_recording_dialog(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open visualization recording", "results/visualization_recordings"
        )
        if path:
            self._open_replay(path)

    def _run_simulation(self) -> None:
        try:
            config = self.config()
        except Exception as exc:
            self._show_error(exc)
            return
        requested_output = Path(config.output)
        if requested_output.exists():
            if not config.auto_increment_output:
                self._show_error(FileExistsError(
                    "Output already exists. Select a new recording directory or "
                    "enable automatic directory numbering."
                ))
                return
            selected_output = next_available_recording_path(requested_output)
            config = replace(config, output=str(selected_output))
            self.output.setText(str(selected_output))
        self._set_running(True)
        self.status.setPlainText(
            "Simulation running in the background. This may take about one minute."
        )
        self._worker = SimulationWorker(config, self)
        self._worker.completed.connect(self._simulation_completed)
        self._worker.failed.connect(self._simulation_failed)
        self._worker.progressed.connect(self._simulation_progressed)
        self._worker.start()

    def _simulation_progressed(self, stage: str, percent: int, elapsed: float) -> None:
        labels = {
            "constellation_and_topology": "Generating constellation and topology",
            "constellation_and_topology_complete": "Constellation and topology complete",
            "measurements_and_transport_inputs_complete": "Measurements and transport inputs complete",
            "online_filter_complete": "Online filtering complete",
            "direction_ring_complete": "Direction Ring CANN complete",
            "rt_line_complete": "RT Line CANN complete",
            "place_cells_complete": "Place-cell encoding complete",
            "cann_snapshot_conversion_complete": "CANN snapshots complete",
            "cann_history_complete": "CANN navigation history complete",
            "cann_disabled": "CANN disabled",
            "recording_write_started": "Writing replay recording",
            "recording_complete": "Replay recording complete",
        }
        self.progress_bar.setValue(percent)
        self.status.setPlainText(
            f"{labels.get(stage, stage)}\nProgress: {percent}% | elapsed: {elapsed:.1f} s"
        )

    def _simulation_completed(self, output: str) -> None:
        config = self._worker.config
        self._set_running(False)
        self.progress_bar.setValue(100)
        self.status.setPlainText(f"Simulation completed: {output}")
        if config.open_after_run:
            self._open_replay(output)

    def _simulation_failed(self, message: str) -> None:
        self._set_running(False)
        self.status.setPlainText(f"Simulation failed: {message}")

    def _set_running(self, running: bool) -> None:
        for button in (
            self.load_button, self.save_button, self.open_button, self.run_button,
        ):
            button.setEnabled(not running)
        self.run_button.setText("Running..." if running else "Run simulation")

    def _open_replay(self, path: str) -> None:
        try:
            self._replay_window = VisualizationReplayWindow(
                VisualizationReplayModel(path)
            )
        except Exception as exc:
            self._show_error(exc)
            return
        self._replay_window.show()

    def _show_error(self, exc: Exception) -> None:
        self.status.setPlainText(f"Error: {type(exc).__name__}: {exc}")


def run_simulation_launcher() -> int:
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = VisualizationSimulationLauncher()
    window.show()
    return application.exec()
