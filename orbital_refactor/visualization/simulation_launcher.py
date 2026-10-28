"""Qt setup window for configured simulation and replay launch."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from experiments.run_v15_dynamic_visualization_recording import (
    generate_dynamic_walker_recording,
)
from experiments.run_bilateral_visualization_recording import (
    generate_bilateral_visualization_recording,
)
from experiments.visualization_simulation_config import (
    VisualizationSimulationConfig,
    load_visualization_simulation_config,
    next_available_recording_path,
    save_visualization_simulation_config,
)
from visualization.acceptance_test_workflow import (
    FORMAL_MODE,
    PRESETS,
    QUICK_MODE,
    SINGLE_SEED_FULL_MODE,
    build_acceptance_plan,
    describe_acceptance_plan,
    finalize_acceptance_plan,
    make_acceptance_output_directory,
)
from visualization.replay_model import VisualizationReplayModel
from visualization.replay_window import VisualizationReplayWindow
from visualization.process_launch import module_process_command


MAX_PARALLEL_ACCEPTANCE_JOBS = 2


class SimulationWorker(QtCore.QThread):
    completed = QtCore.Signal(str)
    failed = QtCore.Signal(str)
    progressed = QtCore.Signal(str, int, float)

    def __init__(self, config: VisualizationSimulationConfig, parent=None):
        super().__init__(parent)
        self.config = config

    def run(self) -> None:
        try:
            if self.config.scenario_type in {
                "BILATERAL_CONFIGURABLE", "BILATERAL_15X15",
            }:
                output = generate_bilateral_visualization_recording(
                    Path(self.config.output),
                    **self.config.bilateral_generator_arguments,
                    progress_callback=self._report_progress,
                )
            else:
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
        self._acceptance_processes = {}
        self._acceptance_plan = None
        self._acceptance_job_index = 0
        self._acceptance_completed_jobs = 0
        self.setWindowTitle("Satellite Swarm Simulation Setup")
        self.resize(860, 680)
        self._build_ui()
        self.set_config(VisualizationSimulationConfig())

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        tabs = QtWidgets.QTabWidget()
        root.addWidget(tabs, 1)

        self.total_satellites = self._integer(2, 100)
        self.scenario_type = QtWidgets.QComboBox()
        self.scenario_type.addItem("Walker cooperative constellation", "WALKER_COOPERATIVE")
        self.scenario_type.addItem(
            "Bilateral: configurable friendly + hostile Walker fleets",
            "BILATERAL_CONFIGURABLE",
        )
        self.scenario_type.addItem("Bilateral: 15 friendly + 15 hostile", "BILATERAL_15X15")
        self.scenario_type.currentIndexChanged.connect(
            self._apply_scenario_controls
        )
        self.plane_count = self._integer(1, 100)
        self.phasing = self._integer(0, 99)
        self.altitude_km = self._decimal(100.0, 100000.0, 1)
        self.inclination_deg = self._decimal(0.0, 180.0, 2)
        self.friendly_total_satellites = self._integer(2, 100)
        self.friendly_plane_count = self._integer(1, 100)
        self.friendly_phasing = self._integer(0, 99)
        self.friendly_altitude_km = self._decimal(100.0, 100000.0, 1)
        self.friendly_inclination_deg = self._decimal(0.0, 180.0, 2)
        self.hostile_total_satellites = self._integer(2, 100)
        self.hostile_plane_count = self._integer(1, 100)
        self.hostile_phasing = self._integer(0, 99)
        self.hostile_altitude_km = self._decimal(100.0, 100000.0, 1)
        self.hostile_inclination_deg = self._decimal(0.0, 180.0, 2)
        self.duration_s = self._decimal(2.0, 86400.0, 1)
        self.dt_s = self._decimal(0.1, 3600.0, 2)
        self.maximum_range_km = self._decimal(1.0, 100000.0, 1)
        self.topology_audit_max_duration_s = self._decimal(1.0, 86400.0, 1)
        self.seed = self._integer(0, 2_000_000_000)
        self.absolute_dropout = QtWidgets.QCheckBox(
            "Interrupt absolute navigation on two nodes"
        )
        self.process_noise = self._decimal(1e-12, 1.0, 12)
        self.bilateral_process_noise = self._decimal(1e-12, 1.0, 12)
        self.radar_enabled = QtWidgets.QCheckBox("Enable radar measurements")
        self.infrared_enabled = QtWidgets.QCheckBox("Enable infrared measurements")
        self.optical_enabled = QtWidgets.QCheckBox("Enable optical measurements")
        self.modality_dropout = QtWidgets.QComboBox()
        self.modality_dropout.addItem("No modality dropout", "NONE")
        self.modality_dropout.addItem("Radar dropout", "RADAR")
        self.modality_dropout.addItem("Infrared dropout", "INFRARED")
        self.modality_dropout.addItem("Optical dropout", "OPTICAL")
        self.dropout_start_s = self._decimal(0.0, 86400.0, 1)
        self.dropout_duration_s = self._decimal(0.1, 86400.0, 1)
        self.modality_dropout.currentIndexChanged.connect(
            self._apply_dropout_controls
        )
        self.radar_range_sigma = self._decimal(0.001, 10000.0, 3)
        self.radar_rate_sigma = self._decimal(0.0001, 1000.0, 4)
        self.infrared_sigma = self._decimal(0.0001, 90.0, 4)
        self.optical_sigma = self._decimal(1e-8, 1.0, 8)
        self.absolute_sigma = self._decimal(0.001, 10000.0, 3)
        self.estimation_architecture = QtWidgets.QComboBox()
        self.estimation_architecture.addItem(
            "Single-observer multimodal estimation", "SINGLE_OBSERVER"
        )
        self.estimation_architecture.addItem(
            "Multi-observer independent estimation", "MULTI_INDEPENDENT"
        )
        self.estimation_architecture.addItem(
            "Multi-star cooperative estimation (Consensus-CI)",
            "MULTI_COOPERATIVE",
        )
        self._last_configurable_estimation_architecture = "MULTI_INDEPENDENT"
        self.tracking_observer_number = self._integer(1, 100)
        self.tracking_target_number = self._integer(1, 100)
        self.estimation_architecture.currentIndexChanged.connect(
            self._apply_estimation_controls
        )
        self.estimation_architecture.currentIndexChanged.connect(
            self._remember_configurable_estimation_architecture
        )
        self.friendly_total_satellites.valueChanged.connect(
            self._apply_estimation_controls
        )
        self.hostile_total_satellites.valueChanged.connect(
            self._apply_estimation_controls
        )
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
            "Automatically create a timestamped directory when output exists"
        )
        self.output = QtWidgets.QLineEdit()
        output_row = QtWidgets.QHBoxLayout()
        output_row.addWidget(self.output, 1)
        browse_output = QtWidgets.QPushButton("Browse...")
        browse_output.clicked.connect(self._browse_output)
        output_row.addWidget(browse_output)

        _, self.constellation_form = self._add_tab(tabs, "Constellation", (
            ("Scenario", self.scenario_type),
            ("Walker total satellites", self.total_satellites),
            ("Walker orbital planes", self.plane_count),
            ("Walker phasing", self.phasing),
            ("Walker altitude (km)", self.altitude_km),
            ("Walker inclination (deg)", self.inclination_deg),
            ("Friendly satellites", self.friendly_total_satellites),
            ("Friendly orbital planes", self.friendly_plane_count),
            ("Friendly Walker phasing", self.friendly_phasing),
            ("Friendly altitude (km)", self.friendly_altitude_km),
            ("Friendly inclination (deg)", self.friendly_inclination_deg),
            ("Hostile satellites", self.hostile_total_satellites),
            ("Hostile orbital planes", self.hostile_plane_count),
            ("Hostile Walker phasing", self.hostile_phasing),
            ("Hostile altitude (km)", self.hostile_altitude_km),
            ("Hostile inclination (deg)", self.hostile_inclination_deg),
        ))
        self._add_tab(tabs, "Simulation & dynamics", (
            ("Duration (s)", self.duration_s),
            ("Time step (s)", self.dt_s),
            ("Random seed", self.seed),
            ("Navigation fault", self.absolute_dropout),
            ("Process acceleration noise", self.process_noise),
            ("Bilateral tracking process noise", self.bilateral_process_noise),
        ))
        self._add_tab(tabs, "Measurements & visibility", (
            ("Radar", self.radar_enabled),
            ("Infrared", self.infrared_enabled),
            ("Optical", self.optical_enabled),
            ("Modality fault", self.modality_dropout),
            ("Dropout start (s)", self.dropout_start_s),
            ("Dropout duration (s)", self.dropout_duration_s),
            ("Maximum observation/link range (km)", self.maximum_range_km),
            ("Radar range sigma (m)", self.radar_range_sigma),
            ("Radar range-rate sigma (m/s)", self.radar_rate_sigma),
            ("Infrared angle sigma (deg)", self.infrared_sigma),
            ("Optical image sigma", self.optical_sigma),
        ))
        self._add_tab(tabs, "Filter & integrity", (
            ("Estimation architecture", self.estimation_architecture),
            ("Single observer number (1-based)", self.tracking_observer_number),
            ("Single target number (1-based)", self.tracking_target_number),
            ("Absolute navigation sigma (m)", self.absolute_sigma),
            ("Replay history window (s)", self.replay_window),
            ("Maximum pinned age (s)", self.max_pinned_age),
            ("Integrity policy", QtWidgets.QLabel("Validated baseline (fixed)")),
        ))
        self._add_tab(tabs, "Communication & topology", (
            ("Degradation profile", self.communication_profile),
            ("Topology", QtWidgets.QLabel("Walker persistent topology")),
            ("Topology audit maximum duration (s)", self.topology_audit_max_duration_s),
            ("Fault injection", self.link_suspension),
        ))
        self._add_tab(tabs, "CANN & output", (
            ("CANN", self.cann_enabled),
            ("CANN node limit", self.cann_node_limit),
            ("Recording output", output_row),
            ("Existing output", self.auto_increment_output),
            ("After run", self.open_after_run),
        ))
        self._build_acceptance_tab(tabs)

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

    def _build_acceptance_tab(self, tabs) -> None:
        page = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(page)
        form = QtWidgets.QFormLayout()
        self.acceptance_test = QtWidgets.QComboBox()
        for preset in PRESETS:
            self.acceptance_test.addItem(preset.name, preset.preset_id)
        self.acceptance_mode = QtWidgets.QComboBox()
        self.acceptance_mode.addItem(
            "快速预检（seed 0，20 s）", QUICK_MODE
        )
        self.acceptance_mode.addItem(
            "现场验收（单种子可选，120 s）",
            SINGLE_SEED_FULL_MODE,
        )
        self.acceptance_mode.addItem(
            "重复性验证（seed 0～4，120 s）", FORMAL_MODE
        )
        self.acceptance_mode.setCurrentIndex(
            self.acceptance_mode.findData(SINGLE_SEED_FULL_MODE)
        )
        self.acceptance_single_seed = self._integer(0, 4)
        self.acceptance_single_seed.setValue(0)
        self.acceptance_default_r06_window = QtWidgets.QCheckBox(
            "使用默认R06缺失窗口"
        )
        self.acceptance_default_r06_window.setChecked(True)
        self.acceptance_dropout_start_s = self._decimal(0.0, 86400.0, 1)
        self.acceptance_dropout_duration_s = self._decimal(1.0, 86400.0, 1)
        self.acceptance_output = QtWidgets.QLineEdit()
        output_row = QtWidgets.QHBoxLayout()
        output_row.addWidget(self.acceptance_output, 1)
        browse = QtWidgets.QPushButton("浏览...")
        browse.clicked.connect(self._browse_acceptance_output)
        output_row.addWidget(browse)
        form.addRow("测试项目", self.acceptance_test)
        form.addRow("运行方式", self.acceptance_mode)
        form.addRow("随机种子", self.acceptance_single_seed)
        form.addRow("R06缺失窗口", self.acceptance_default_r06_window)
        form.addRow("R06缺失开始时间（s）", self.acceptance_dropout_start_s)
        form.addRow(
            "R06缺失持续时间（s）", self.acceptance_dropout_duration_s
        )
        form.addRow("结果目录", output_row)
        root.addLayout(form)

        self.acceptance_description = QtWidgets.QPlainTextEdit()
        self.acceptance_description.setReadOnly(True)
        self.acceptance_description.setMaximumHeight(155)
        root.addWidget(self.acceptance_description)

        controls = QtWidgets.QHBoxLayout()
        self.acceptance_run_button = QtWidgets.QPushButton("运行所选测试")
        self.acceptance_stop_button = QtWidgets.QPushButton("停止")
        self.acceptance_open_button = QtWidgets.QPushButton("打开结果目录")
        self.acceptance_stop_button.setEnabled(False)
        self.acceptance_run_button.clicked.connect(self._run_acceptance_test)
        self.acceptance_stop_button.clicked.connect(self._stop_acceptance_test)
        self.acceptance_open_button.clicked.connect(self._open_acceptance_output)
        controls.addWidget(self.acceptance_run_button)
        controls.addWidget(self.acceptance_stop_button)
        controls.addWidget(self.acceptance_open_button)
        root.addLayout(controls)

        self.acceptance_progress = QtWidgets.QProgressBar()
        self.acceptance_progress.setRange(0, 100)
        root.addWidget(self.acceptance_progress)
        self.acceptance_results = QtWidgets.QTreeWidget()
        self.acceptance_results.setHeaderLabels(
            ("指标", "数值", "判据", "结果")
        )
        self.acceptance_results.setRootIsDecorated(False)
        self.acceptance_results.setAlternatingRowColors(True)
        root.addWidget(self.acceptance_results, 1)
        self.acceptance_log = QtWidgets.QPlainTextEdit()
        self.acceptance_log.setReadOnly(True)
        self.acceptance_log.setMaximumBlockCount(1000)
        self.acceptance_log.setMaximumHeight(135)
        root.addWidget(self.acceptance_log)
        tabs.addTab(page, "Acceptance tests")

        self.acceptance_test.currentIndexChanged.connect(
            self._refresh_acceptance_plan
        )
        self.acceptance_mode.currentIndexChanged.connect(
            self._refresh_acceptance_plan
        )
        self.acceptance_single_seed.valueChanged.connect(
            self._refresh_acceptance_description
        )
        self.acceptance_default_r06_window.toggled.connect(
            self._refresh_acceptance_window_controls
        )
        self.acceptance_dropout_start_s.valueChanged.connect(
            self._refresh_acceptance_description
        )
        self.acceptance_dropout_duration_s.valueChanged.connect(
            self._refresh_acceptance_description
        )
        self._refresh_acceptance_plan()

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
        return page, form

    def _set_constellation_fields_visible(self, fields, visible) -> None:
        for field in fields:
            label = self.constellation_form.labelForField(field)
            if label is not None:
                label.setVisible(visible)
            field.setVisible(visible)

    def config(self) -> VisualizationSimulationConfig:
        return VisualizationSimulationConfig(
            scenario_type=self.scenario_type.currentData(),
            total_satellites=self.total_satellites.value(),
            plane_count=self.plane_count.value(), phasing=self.phasing.value(),
            altitude_km=self.altitude_km.value(),
            inclination_deg=self.inclination_deg.value(),
            friendly_total_satellites=self.friendly_total_satellites.value(),
            friendly_plane_count=self.friendly_plane_count.value(),
            friendly_phasing=self.friendly_phasing.value(),
            friendly_altitude_km=self.friendly_altitude_km.value(),
            friendly_inclination_deg=self.friendly_inclination_deg.value(),
            hostile_total_satellites=self.hostile_total_satellites.value(),
            hostile_plane_count=self.hostile_plane_count.value(),
            hostile_phasing=self.hostile_phasing.value(),
            hostile_altitude_km=self.hostile_altitude_km.value(),
            hostile_inclination_deg=self.hostile_inclination_deg.value(),
            duration_s=self.duration_s.value(), dt_s=self.dt_s.value(),
            maximum_range_km=self.maximum_range_km.value(),
            seed=self.seed.value(),
            enable_absolute_navigation_dropout=self.absolute_dropout.isChecked(),
            process_noise_acceleration=self.process_noise.value(),
            bilateral_process_noise_acceleration=(
                self.bilateral_process_noise.value()
            ),
            radar_enabled=self.radar_enabled.isChecked(),
            infrared_enabled=self.infrared_enabled.isChecked(),
            optical_enabled=self.optical_enabled.isChecked(),
            dropout_modality=self.modality_dropout.currentData(),
            dropout_start_s=self.dropout_start_s.value(),
            dropout_duration_s=self.dropout_duration_s.value(),
            radar_range_sigma_m=self.radar_range_sigma.value(),
            radar_range_rate_sigma_mps=self.radar_rate_sigma.value(),
            infrared_angle_sigma_deg=self.infrared_sigma.value(),
            optical_image_sigma=self.optical_sigma.value(),
            absolute_navigation_sigma_m=self.absolute_sigma.value(),
            estimation_architecture=self.estimation_architecture.currentData(),
            tracking_observer_index=self.tracking_observer_number.value() - 1,
            tracking_target_index=self.tracking_target_number.value() - 1,
            replay_history_window_s=self.replay_window.value(),
            max_pinned_age_s=self.max_pinned_age.value(),
            communication_profile=self.communication_profile.currentText(),
            topology_audit_max_duration_s=(
                self.topology_audit_max_duration_s.value()
            ),
            enable_link_suspension=self.link_suspension.isChecked(),
            cann_enabled=self.cann_enabled.isChecked(),
            cann_node_limit=self.cann_node_limit.value(),
            output=self.output.text().strip(),
            open_after_run=self.open_after_run.isChecked(),
            auto_increment_output=self.auto_increment_output.isChecked(),
        )

    def set_config(self, config: VisualizationSimulationConfig) -> None:
        scenario_index = self.scenario_type.findData(config.scenario_type)
        self.scenario_type.setCurrentIndex(max(0, scenario_index))
        self.total_satellites.setValue(config.total_satellites)
        self.plane_count.setValue(config.plane_count)
        self.phasing.setValue(config.phasing)
        self.altitude_km.setValue(config.altitude_km)
        self.inclination_deg.setValue(config.inclination_deg)
        self.friendly_total_satellites.setValue(
            config.friendly_total_satellites
        )
        self.friendly_plane_count.setValue(config.friendly_plane_count)
        self.friendly_phasing.setValue(config.friendly_phasing)
        self.friendly_altitude_km.setValue(config.friendly_altitude_km)
        self.friendly_inclination_deg.setValue(
            config.friendly_inclination_deg
        )
        self.hostile_total_satellites.setValue(
            config.hostile_total_satellites
        )
        self.hostile_plane_count.setValue(config.hostile_plane_count)
        self.hostile_phasing.setValue(config.hostile_phasing)
        self.hostile_altitude_km.setValue(config.hostile_altitude_km)
        self.hostile_inclination_deg.setValue(
            config.hostile_inclination_deg
        )
        self.duration_s.setValue(config.duration_s)
        self.dt_s.setValue(config.dt_s)
        self.maximum_range_km.setValue(config.maximum_range_km)
        self.seed.setValue(config.seed)
        self.absolute_dropout.setChecked(config.enable_absolute_navigation_dropout)
        self.process_noise.setValue(config.process_noise_acceleration)
        self.bilateral_process_noise.setValue(
            config.bilateral_process_noise_acceleration
        )
        self.radar_enabled.setChecked(config.radar_enabled)
        self.infrared_enabled.setChecked(config.infrared_enabled)
        self.optical_enabled.setChecked(config.optical_enabled)
        self.modality_dropout.setCurrentIndex(max(
            0, self.modality_dropout.findData(config.dropout_modality),
        ))
        self.dropout_start_s.setValue(config.dropout_start_s)
        self.dropout_duration_s.setValue(config.dropout_duration_s)
        self.radar_range_sigma.setValue(config.radar_range_sigma_m)
        self.radar_rate_sigma.setValue(config.radar_range_rate_sigma_mps)
        self.infrared_sigma.setValue(config.infrared_angle_sigma_deg)
        self.optical_sigma.setValue(config.optical_image_sigma)
        self.absolute_sigma.setValue(config.absolute_navigation_sigma_m)
        self.estimation_architecture.setCurrentIndex(max(
            0,
            self.estimation_architecture.findData(
                config.estimation_architecture
            ),
        ))
        self.tracking_observer_number.setValue(
            config.tracking_observer_index + 1
        )
        self.tracking_target_number.setValue(
            config.tracking_target_index + 1
        )
        self.replay_window.setValue(config.replay_history_window_s)
        self.max_pinned_age.setValue(config.max_pinned_age_s)
        self.communication_profile.setCurrentText(config.communication_profile)
        self.topology_audit_max_duration_s.setValue(
            config.topology_audit_max_duration_s
        )
        self.link_suspension.setChecked(config.enable_link_suspension)
        self.cann_enabled.setChecked(config.cann_enabled)
        self.cann_node_limit.setValue(config.cann_node_limit)
        self.output.setText(config.output)
        self.open_after_run.setChecked(config.open_after_run)
        self.auto_increment_output.setChecked(config.auto_increment_output)
        self._apply_scenario_controls()

    def _apply_scenario_controls(self) -> None:
        scenario = self.scenario_type.currentData()
        bilateral = scenario in {
            "BILATERAL_CONFIGURABLE", "BILATERAL_15X15",
        }
        frozen_bilateral = scenario == "BILATERAL_15X15"
        configurable_bilateral = scenario == "BILATERAL_CONFIGURABLE"
        walker_fields = (
            self.total_satellites, self.plane_count, self.phasing,
            self.altitude_km, self.inclination_deg,
        )
        bilateral_fields = (
            self.friendly_total_satellites, self.friendly_plane_count,
            self.friendly_phasing, self.friendly_altitude_km,
            self.friendly_inclination_deg, self.hostile_total_satellites,
            self.hostile_plane_count, self.hostile_phasing,
            self.hostile_altitude_km, self.hostile_inclination_deg,
        )
        self._set_constellation_fields_visible(walker_fields, not bilateral)
        self._set_constellation_fields_visible(bilateral_fields, bilateral)
        for widget in walker_fields:
            widget.setEnabled(not bilateral)
        for widget in bilateral_fields:
            widget.setEnabled(scenario == "BILATERAL_CONFIGURABLE")
        for widget in (
            self.absolute_dropout,
            self.radar_range_sigma, self.radar_rate_sigma,
            self.infrared_sigma, self.optical_sigma, self.absolute_sigma,
            self.replay_window, self.max_pinned_age,
            self.communication_profile, self.topology_audit_max_duration_s,
            self.link_suspension,
        ):
            widget.setEnabled(not bilateral)
        self.process_noise.setEnabled(not bilateral)
        self.bilateral_process_noise.setEnabled(configurable_bilateral)
        self.estimation_architecture.setEnabled(configurable_bilateral)
        if frozen_bilateral:
            with QtCore.QSignalBlocker(self.estimation_architecture):
                self.estimation_architecture.setCurrentIndex(
                    self.estimation_architecture.findData("MULTI_COOPERATIVE")
                )
        elif configurable_bilateral:
            with QtCore.QSignalBlocker(self.estimation_architecture):
                self.estimation_architecture.setCurrentIndex(max(
                    0, self.estimation_architecture.findData(
                        self._last_configurable_estimation_architecture
                    ),
                ))
        for widget in (
            self.maximum_range_km, self.radar_enabled,
            self.infrared_enabled, self.optical_enabled,
        ):
            widget.setEnabled(not frozen_bilateral)
        self.modality_dropout.setEnabled(configurable_bilateral)
        bilateral_tracking_help = (
            "For bilateral target tracking, enabled modalities are applied "
            "after the common multimodal target-initialization stage."
        )
        for widget in (
            self.radar_enabled, self.infrared_enabled, self.optical_enabled,
        ):
            widget.setToolTip(
                bilateral_tracking_help if bilateral else ""
            )
        self.bilateral_process_noise.setToolTip(
            "Target-tracker process acceleration noise for the configurable "
            "bilateral scene. The frozen 15:15 scene uses 1e-4."
        )
        self._apply_estimation_controls()
        self._apply_dropout_controls()
        if frozen_bilateral:
            self.friendly_total_satellites.setValue(15)
            self.friendly_plane_count.setValue(3)
            self.friendly_phasing.setValue(1)
            self.friendly_altitude_km.setValue(700.0)
            self.friendly_inclination_deg.setValue(53.0)
            self.hostile_total_satellites.setValue(15)
            self.hostile_plane_count.setValue(3)
            self.hostile_phasing.setValue(1)
            self.hostile_altitude_km.setValue(705.0)
            self.hostile_inclination_deg.setValue(53.0)
        if bilateral:
            if "bilateral" not in self.output.text().lower():
                self.output.setText(
                    "results/visualization_recordings/bilateral30_configured"
                )

    def _apply_dropout_controls(self) -> None:
        enabled = (
            self.scenario_type.currentData() == "BILATERAL_CONFIGURABLE"
            and self.modality_dropout.currentData() != "NONE"
        )
        self.dropout_start_s.setEnabled(enabled)
        self.dropout_duration_s.setEnabled(enabled)
        help_text = (
            "The interval is applied to free bilateral simulation only. "
            "An outage before target initialization may prevent track birth."
        )
        self.modality_dropout.setToolTip(help_text)
        self.dropout_start_s.setToolTip(help_text)
        self.dropout_duration_s.setToolTip(help_text)

    def _apply_estimation_controls(self) -> None:
        configurable_bilateral = (
            self.scenario_type.currentData() == "BILATERAL_CONFIGURABLE"
        )
        self.tracking_observer_number.setMaximum(max(
            1, self.friendly_total_satellites.value()
        ))
        self.tracking_target_number.setMaximum(max(
            1, self.hostile_total_satellites.value()
        ))
        single = (
            configurable_bilateral
            and self.estimation_architecture.currentData()
            == "SINGLE_OBSERVER"
        )
        self.tracking_observer_number.setEnabled(single)
        self.tracking_target_number.setEnabled(single)
        self.estimation_architecture.setToolTip(
            "Single-observer mode performs common multimodal initialization "
            "for only the selected hostile target, then restricts continued "
            "tracking to the selected friendly observer. The frozen 15:15 scene always "
            "uses multi-star cooperative Consensus-CI."
        )

    def _remember_configurable_estimation_architecture(self) -> None:
        if self.scenario_type.currentData() != "BILATERAL_CONFIGURABLE":
            return
        value = self.estimation_architecture.currentData()
        if value:
            self._last_configurable_estimation_architecture = str(value)

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

    def _browse_acceptance_output(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select acceptance result directory", "results"
        )
        if path:
            self.acceptance_output.setText(path)
            self._refresh_acceptance_description()

    def _refresh_acceptance_plan(self) -> None:
        preset_id = self.acceptance_test.currentData()
        mode = self.acceptance_mode.currentData()
        if not preset_id or not mode:
            return
        self._refresh_acceptance_window_controls()
        self.acceptance_single_seed.setEnabled(
            self._acceptance_uses_selected_seed()
        )
        self.acceptance_output.setText(str(
            make_acceptance_output_directory(preset_id)
        ))
        self._refresh_acceptance_description()

    def _refresh_acceptance_window_controls(self) -> None:
        preset_id = self.acceptance_test.currentData()
        mode = self.acceptance_mode.currentData()
        if not preset_id or not mode:
            return
        is_r06 = str(preset_id).startswith("R06_")
        self.acceptance_default_r06_window.setEnabled(is_r06)
        if self.acceptance_default_r06_window.isChecked():
            start, duration = (
                (6.0, 8.0) if mode == QUICK_MODE else (30.0, 60.0)
            )
            self.acceptance_dropout_start_s.setValue(start)
            self.acceptance_dropout_duration_s.setValue(duration)
        custom_enabled = (
            is_r06 and not self.acceptance_default_r06_window.isChecked()
        )
        self.acceptance_dropout_start_s.setEnabled(custom_enabled)
        self.acceptance_dropout_duration_s.setEnabled(custom_enabled)
        self._refresh_acceptance_description()

    def _acceptance_uses_selected_seed(self) -> bool:
        preset_id = str(self.acceptance_test.currentData() or "")
        return (
            self.acceptance_mode.currentData() == SINGLE_SEED_FULL_MODE
            and preset_id.startswith(("P05_", "P07_", "R06_", "R08_"))
        )

    def _refresh_acceptance_description(self) -> None:
        try:
            plan = self._build_acceptance_plan_from_ui()
            self.acceptance_description.setPlainText(
                describe_acceptance_plan(plan)
            )
        except Exception as exc:
            self.acceptance_description.setPlainText(
                f"测试配置无效：{type(exc).__name__}: {exc}"
            )

    def _run_acceptance_test(self) -> None:
        try:
            plan = self._build_acceptance_plan_from_ui()
            output = Path(plan.output_directory)
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(
                    "Result directory already contains files. Select a new directory."
                )
            output.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self._show_acceptance_error(exc)
            return
        self._acceptance_plan = plan
        self._acceptance_job_index = 0
        self._acceptance_completed_jobs = 0
        self._acceptance_processes.clear()
        self.acceptance_results.clear()
        self.acceptance_log.clear()
        self.acceptance_progress.setValue(0)
        self._set_acceptance_running(True)
        self.acceptance_log.appendPlainText(
            f"开始运行：{plan.preset.name}；{self.acceptance_mode.currentText()}"
        )
        self._start_next_acceptance_job()

    def _build_acceptance_plan_from_ui(self):
        preset_id = self.acceptance_test.currentData()
        custom_r06 = (
            str(preset_id).startswith("R06_")
            and not self.acceptance_default_r06_window.isChecked()
        )
        return build_acceptance_plan(
            preset_id,
            self.acceptance_mode.currentData(),
            self.acceptance_output.text().strip(),
            single_seed=self.acceptance_single_seed.value(),
            r06_event_start_s=(
                self.acceptance_dropout_start_s.value()
                if custom_r06 else None
            ),
            r06_event_duration_s=(
                self.acceptance_dropout_duration_s.value()
                if custom_r06 else None
            ),
        )

    def _start_next_acceptance_job(self) -> None:
        plan = self._acceptance_plan
        if plan is None:
            return
        while (
            self._acceptance_job_index < len(plan.jobs)
            and len(self._acceptance_processes) < MAX_PARALLEL_ACCEPTANCE_JOBS
        ):
            job_index = self._acceptance_job_index
            self._acceptance_job_index += 1
            self._start_acceptance_job(job_index)
        if (
            self._acceptance_completed_jobs >= len(plan.jobs)
            and not self._acceptance_processes
        ):
            self._finish_acceptance_plan()

    def _start_acceptance_job(self, job_index: int) -> None:
        plan = self._acceptance_plan
        if plan is None:
            return
        job = plan.jobs[job_index]
        self.acceptance_log.appendPlainText(
            f"\n[{job_index + 1}/{len(plan.jobs)}] {job.label}"
        )
        process = QtCore.QProcess(self)
        process.setWorkingDirectory(str(Path.cwd()))
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            environment.insert(name, "1")
        process.setProcessEnvironment(environment)
        process.setProcessChannelMode(
            QtCore.QProcess.ProcessChannelMode.MergedChannels
        )
        process.readyReadStandardOutput.connect(
            lambda process=process, index=job_index:
                self._read_acceptance_output(process, index)
        )
        process.finished.connect(
            lambda exit_code, exit_status, process=process, index=job_index:
                self._acceptance_job_finished(
                    process, index, exit_code, exit_status,
                )
        )
        process.errorOccurred.connect(
            lambda error, process=process, index=job_index:
                self._acceptance_process_error(process, index, error)
        )
        self._acceptance_processes[process] = job_index
        program, arguments = module_process_command(job.module, job.arguments)
        process.start(program, arguments)

    def _read_acceptance_output(self, process, job_index: int) -> None:
        if process not in self._acceptance_processes:
            return
        text = bytes(
            process.readAllStandardOutput()
        ).decode("utf-8", errors="replace").rstrip()
        if text:
            prefix = f"[{job_index + 1}] "
            self.acceptance_log.appendPlainText(
                prefix + text.replace("\n", "\n" + prefix)
            )

    def _acceptance_job_finished(
        self, process, job_index: int, exit_code, exit_status,
    ) -> None:
        if process not in self._acceptance_processes:
            return
        self._read_acceptance_output(process, job_index)
        self._acceptance_processes.pop(process, None)
        process.deleteLater()
        plan = self._acceptance_plan
        if plan is None:
            return
        if exit_code != 0 or exit_status != QtCore.QProcess.ExitStatus.NormalExit:
            self._abort_acceptance_jobs(
                f"测试已停止：进程退出码为 {exit_code}。"
            )
            return
        job = plan.jobs[job_index]
        if not Path(job.output_path).exists():
            self._abort_acceptance_jobs(
                f"测试已停止：未生成预期结果 {job.output_path}"
            )
            return
        self._acceptance_completed_jobs += 1
        self.acceptance_progress.setValue(round(
            90 * self._acceptance_completed_jobs / len(plan.jobs)
        ))
        self._start_next_acceptance_job()

    def _acceptance_process_error(self, process, job_index: int, error) -> None:
        if process not in self._acceptance_processes:
            return
        if error == QtCore.QProcess.ProcessError.Crashed:
            return
        self.acceptance_log.appendPlainText(
            f"任务 {job_index + 1} 发生错误：{error}"
        )
        if error == QtCore.QProcess.ProcessError.FailedToStart:
            self._abort_acceptance_jobs(
                f"测试已停止：任务 {job_index + 1} 无法启动。"
            )

    def _abort_acceptance_jobs(self, message: str) -> None:
        processes = tuple(self._acceptance_processes)
        self._acceptance_processes.clear()
        for process in processes:
            if process.state() != QtCore.QProcess.ProcessState.NotRunning:
                process.kill()
        self._acceptance_plan = None
        self._set_acceptance_running(False)
        self.acceptance_log.appendPlainText(message)

    def _finish_acceptance_plan(self) -> None:
        try:
            outcome = finalize_acceptance_plan(self._acceptance_plan)
        except Exception as exc:
            self._set_acceptance_running(False)
            self._show_acceptance_error(exc)
            return
        for metric in outcome.metrics:
            value = "N/A" if metric.value is None else f"{metric.value:.6g} {metric.unit}"
            status = (
                "已记录" if metric.passed is None
                else "通过" if metric.passed else "未通过"
            )
            item = QtWidgets.QTreeWidgetItem(
                (metric.metric, value, metric.threshold, status)
            )
            if metric.passed is False:
                item.setForeground(3, QtGui.QBrush(QtGui.QColor("#b00020")))
            elif metric.passed is True:
                item.setForeground(3, QtGui.QBrush(QtGui.QColor("#137333")))
            self.acceptance_results.addTopLevelItem(item)
        self.acceptance_results.resizeColumnToContents(0)
        self.acceptance_progress.setValue(100)
        outcome_status = {
            "RECORDED": "已记录",
            "PASS": "通过",
            "FAIL": "未通过",
            "PENDING_MANUAL_CHECK": "等待人工检查",
        }.get(outcome.status, outcome.status)
        self.acceptance_log.appendPlainText(
            f"\n运行完成：{outcome_status}\n{outcome.note}\n"
            f"汇总文件：{outcome.summary_path}"
        )
        self._set_acceptance_running(False)
        if self._acceptance_plan.recording_path is not None:
            self._open_replay(Path(self._acceptance_plan.recording_path))

    def _stop_acceptance_test(self) -> None:
        self._abort_acceptance_jobs("测试已由用户停止。")

    def _set_acceptance_running(self, running: bool) -> None:
        self.acceptance_run_button.setEnabled(not running)
        self.acceptance_stop_button.setEnabled(running)
        self.acceptance_test.setEnabled(not running)
        self.acceptance_mode.setEnabled(not running)
        self.acceptance_single_seed.setEnabled(
            not running and self._acceptance_uses_selected_seed()
        )
        self.acceptance_output.setEnabled(not running)
        if running:
            self.acceptance_default_r06_window.setEnabled(False)
            self.acceptance_dropout_start_s.setEnabled(False)
            self.acceptance_dropout_duration_s.setEnabled(False)
        else:
            self._refresh_acceptance_window_controls()
        self.acceptance_run_button.setText(
            "运行中..." if running else "运行所选测试"
        )

    def _open_acceptance_output(self) -> None:
        path = Path(self.acceptance_output.text().strip())
        if not path.exists():
            self._show_acceptance_error(FileNotFoundError(path))
            return
        QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(str(path.resolve()))
        )

    def _show_acceptance_error(self, exc: Exception) -> None:
        self.acceptance_log.appendPlainText(
            f"错误：{type(exc).__name__}: {exc}"
        )

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
                    "enable automatic timestamped directories."
                ))
                return
            selected_output = next_available_recording_path(requested_output)
            config = replace(config, output=str(selected_output))
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
            "bilateral_frontend_started": "Generating bilateral raw sensor frontend",
            "bilateral_frontend_complete": "Bilateral raw frontend complete",
            "bilateral_tracking_complete": "Bilateral target tracking complete",
            "bilateral_cann_complete": "Absolute-position CANN display complete",
        }
        self.progress_bar.setValue(percent)
        self.status.setPlainText(
            f"{labels.get(stage, stage)}\nProgress: {percent}% | elapsed: {elapsed:.1f} s"
        )

    def _simulation_completed(self, output: str) -> None:
        config = self._worker.config
        configuration_path = Path(output) / "launch_config.json"
        try:
            save_visualization_simulation_config(config, configuration_path)
            configuration_note = f"\nConfiguration: {configuration_path}"
        except Exception as exc:
            configuration_note = (
                f"\nConfiguration snapshot warning: "
                f"{type(exc).__name__}: {exc}"
            )
        self._set_running(False)
        self.progress_bar.setValue(100)
        self.status.setPlainText(
            f"Simulation completed: {output}{configuration_note}"
        )
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


def run_simulation_launcher(*, auto_close_ms: int | None = None) -> int:
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = VisualizationSimulationLauncher()
    window.show()
    if auto_close_ms is not None:
        QtCore.QTimer.singleShot(max(0, int(auto_close_ms)), application.quit)
    return application.exec()
