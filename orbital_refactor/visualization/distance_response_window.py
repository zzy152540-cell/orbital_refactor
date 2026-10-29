"""Interactive review window for the three-modality distance-response demo."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6 import QtCore, QtWidgets


_MODALITIES = (
    ("OPTICAL", "Optical", "gray", "optical_detector_frame", "optical"),
    ("INFRARED", "Infrared", "inferno", "infrared_detector_frame", "infrared"),
    ("RADAR", "Radar", "viridis", "radar_range_doppler", "radar_range_doppler"),
)


class DistanceResponseWindow(QtWidgets.QMainWindow):
    """Show generated raw observations at one range and across all ranges."""

    def __init__(self, result_directory: str | Path) -> None:
        super().__init__()
        self.result_directory = Path(result_directory)
        manifest_path = self.result_directory / "manifest.json"
        records_path = self.result_directory / "distance_measurements.json"
        if not manifest_path.is_file() or not records_path.is_file():
            raise FileNotFoundError(
                "The selected directory is not an F11 distance-response result."
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.records = json.loads(records_path.read_text(encoding="utf-8"))
        self.epoch_directories = tuple(
            self.result_directory / name
            for name in self.manifest["epochDirectories"]
        )
        if len(self.epoch_directories) != len(self.records):
            raise ValueError("F11 result has inconsistent epoch and record counts.")
        self._raw_limits = self._shared_raw_limits()

        self.setWindowTitle("三模态量测随距离变化展示")
        self.resize(1380, 900)
        self._build_ui()
        self._set_index(0)

    def _build_ui(self) -> None:
        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        browser = QtWidgets.QWidget()
        browser_layout = QtWidgets.QVBoxLayout(browser)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("距离点"))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, max(0, len(self.records) - 1))
        self.slider.valueChanged.connect(self._set_index)
        controls.addWidget(self.slider, 1)
        self.distance_label = QtWidgets.QLabel()
        self.distance_label.setMinimumWidth(210)
        controls.addWidget(self.distance_label)
        browser_layout.addLayout(controls)

        self.detail_figure = Figure(figsize=(13.2, 7.6), constrained_layout=True)
        self.detail_canvas = FigureCanvasQTAgg(self.detail_figure)
        browser_layout.addWidget(self.detail_canvas, 1)
        self.measurement_label = QtWidgets.QLabel()
        self.measurement_label.setWordWrap(True)
        browser_layout.addWidget(self.measurement_label)
        tabs.addTab(browser, "逐距离查看")

        overview = QtWidgets.QWidget()
        overview_layout = QtWidgets.QVBoxLayout(overview)
        overview_layout.addWidget(QtWidgets.QLabel(
            "各列为由近到远的距离点；三行分别为光学、红外和雷达原始量测数据。"
        ))
        self.overview_figure = Figure(figsize=(14.0, 7.2), constrained_layout=True)
        self.overview_canvas = FigureCanvasQTAgg(self.overview_figure)
        self.overview_canvas.setMinimumSize(980, 620)
        overview_layout.addWidget(self.overview_canvas, 1)
        tabs.addTab(overview, "距离变化并列总览")
        self._draw_overview()

    def _load_epoch(self, index: int):
        directory = self.epoch_directories[index]
        raw = np.load(directory / "sensor_raw.npz", allow_pickle=False)
        enhanced = np.load(
            directory / "presentation_enhanced.npz", allow_pickle=False
        )
        measurements = json.loads(
            (directory / "extracted_measurements.json").read_text(
                encoding="utf-8"
            )
        )["measurements"]
        return raw, enhanced, measurements

    def _shared_raw_limits(self):
        values = {name: [] for name, *_ in _MODALITIES}
        for directory in self.epoch_directories:
            with np.load(directory / "sensor_raw.npz", allow_pickle=False) as raw:
                for name, _label, _cmap, raw_key, _enhanced_key in _MODALITIES:
                    array = np.asarray(raw[raw_key], dtype=float)
                    values[name].append(array[np.isfinite(array)])
        limits = {}
        for name, arrays in values.items():
            finite = np.concatenate([item for item in arrays if item.size])
            limits[name] = (
                float(np.percentile(finite, 1.0)),
                float(np.percentile(finite, 99.9)),
            )
        return limits

    def _set_index(self, index: int) -> None:
        index = int(index)
        raw, enhanced, measurements = self._load_epoch(index)
        record = self.records[index]
        distance_m = float(record["distance_m"])
        self.distance_label.setText(
            f"{index + 1}/{len(self.records)}  ·  {_format_distance(distance_m)}"
        )
        self.detail_figure.clear()
        axes = self.detail_figure.subplots(2, 3, squeeze=False)
        for column, (name, label, cmap, raw_key, enhanced_key) in enumerate(
            _MODALITIES
        ):
            preview = np.asarray(enhanced[enhanced_key], dtype=float)
            sensor = np.asarray(raw[raw_key], dtype=float)
            origin = "lower" if name == "RADAR" else "upper"
            axes[0, column].imshow(
                preview, cmap=cmap, origin=origin, aspect="equal"
            )
            axes[0, column].set_title(f"{label} display preview")
            vmin, vmax = self._raw_limits[name]
            axes[1, column].imshow(
                sensor, cmap=cmap, origin=origin, aspect="equal",
                vmin=vmin, vmax=vmax,
            )
            axes[1, column].set_title(
                f"{label} raw observation · filter input"
            )
            for axis in axes[:, column]:
                axis.set_xticks([])
                axis.set_yticks([])
        self.detail_figure.suptitle(
            f"Three-modality observations  ·  range={_format_distance(distance_m)}",
            fontsize=14,
        )
        self.detail_canvas.draw_idle()
        self.measurement_label.setText(_measurement_text(measurements))
        raw.close()
        enhanced.close()

    def _draw_overview(self) -> None:
        self.overview_figure.clear()
        axes = self.overview_figure.subplots(
            3, len(self.records), squeeze=False
        )
        for column, (directory, record) in enumerate(
            zip(self.epoch_directories, self.records)
        ):
            with np.load(directory / "sensor_raw.npz", allow_pickle=False) as raw:
                for row, (name, label, cmap, raw_key, _enhanced_key) in enumerate(
                    _MODALITIES
                ):
                    origin = "lower" if name == "RADAR" else "upper"
                    vmin, vmax = self._raw_limits[name]
                    axes[row, column].imshow(
                        raw[raw_key], cmap=cmap, origin=origin, aspect="equal",
                        vmin=vmin, vmax=vmax,
                    )
                    axes[row, column].set_xticks([])
                    axes[row, column].set_yticks([])
                    if column == 0:
                        axes[row, column].set_ylabel(label)
            axes[0, column].set_title(
                _format_distance(float(record["distance_m"])), fontsize=9
            )
        self.overview_figure.suptitle(
            "Three-modality raw observations versus distance · shared scale per modality",
            fontsize=14,
        )
        self.overview_canvas.draw_idle()


def _format_distance(distance_m: float) -> str:
    if distance_m >= 1_000_000.0:
        return f"{distance_m / 1000.0:.4g} km"
    if distance_m >= 1000.0:
        return f"{distance_m / 1000.0:.3g} km"
    return f"{distance_m:.3g} m"


def _measurement_text(measurements) -> str:
    by_modality = {item["modality"]: item for item in measurements}
    radar = by_modality["RADAR"]["measurement"]
    infrared = by_modality["INFRARED"]["measurement"]
    optical = by_modality["OPTICAL"]["measurement"]
    return (
        f"雷达：距离 {radar[0]:.6g} m，距离率 {radar[1]:.6g} m/s    |    "
        f"红外：方位角 {infrared[0]:.6g} rad，俯仰角 {infrared[1]:.6g} rad    |    "
        f"光学：归一化像平面坐标 ({optical[0]:.6g}, {optical[1]:.6g})"
    )
