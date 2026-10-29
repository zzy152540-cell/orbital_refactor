"""Generate and display a lightweight Tango-like three-modality data set."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adapters.tango_like_physical_frontend import (
    PhysicalFrontendConfig,
    build_tango_like_model,
    simulate_tango_like_physical_epoch,
)
from orbital_core.attitude import small_angle_quaternion_wxyz


def run_demo(*, output: Path, frame_count: int = 4, distance_m: float = 8.0,
             maximum_distance_m: float | None = None,
             focal_length_pixels: float = 1600.0,
             unresolved_extent_pixels: float = 0.25,
             resolved_extent_pixels: float = 16.0,
             maximum_observation_range_km: float = 6000.0,
             display_minimum_extent_pixels: float = 2.0,
             display_maximum_extent_pixels: float = 32.0,
             seed: int = 0, show: bool = True) -> Path:
    if frame_count < 1:
        raise ValueError("frame_count must be positive.")
    if distance_m <= 1.0:
        raise ValueError("distance_m must exceed 1 m for this demonstration.")
    if maximum_distance_m is not None and maximum_distance_m < distance_m:
        raise ValueError("maximum_distance_m must not be smaller than distance_m.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = build_tango_like_model()
    default_config = PhysicalFrontendConfig()
    frontend_config = replace(
        default_config,
        optical=replace(
            default_config.optical,
            focal_length_x_pixels=float(focal_length_pixels),
            focal_length_y_pixels=float(focal_length_pixels),
        ),
        infrared=replace(
            default_config.infrared,
            focal_length_x_pixels=float(focal_length_pixels),
            focal_length_y_pixels=float(focal_length_pixels),
        ),
        unresolved_extent_pixels=float(unresolved_extent_pixels),
        resolved_extent_pixels=float(resolved_extent_pixels),
        maximum_observation_range_m=float(maximum_observation_range_km) * 1000.0,
        display_minimum_extent_pixels=float(display_minimum_extent_pixels),
        display_maximum_extent_pixels=float(display_maximum_extent_pixels),
    )
    observer = np.zeros(6)
    sensor_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    epochs = []
    for index in range(frame_count):
        phase = 0.0 if frame_count == 1 else index / (frame_count - 1)
        timestamp = 20.0 * phase
        current_distance = (
            float(np.geomspace(distance_m, maximum_distance_m, frame_count)[index])
            if maximum_distance_m is not None and frame_count > 1
            else distance_m - 0.5 * phase
        )
        target = np.array([
            current_distance,
            0.12 * np.sin(2.0 * np.pi * phase),
            0.08 * np.cos(2.0 * np.pi * phase),
            -0.025, 0.035 * np.cos(2.0 * np.pi * phase), 0.0,
        ])
        target_quaternion = small_angle_quaternion_wxyz(np.array([
            0.35 * phase, 0.65 * phase, 1.1 * phase,
        ]))
        epochs.append(simulate_tango_like_physical_epoch(
            timestamp=timestamp, observer_id="friendly_demo",
            target_id="tango_like_target", observer_state=observer,
            target_state=target,
            sensor_quaternion_i2b_wxyz=sensor_quaternion,
            target_quaternion_i2b_wxyz=target_quaternion,
            target_angular_velocity_body_radps=np.array([0.015, 0.025, 0.04]),
            model=model, config=frontend_config, random_seed=seed + index,
        ))

    preview_files = []
    epoch_directories = []
    distance_records = []
    for index, epoch in enumerate(epochs):
        epoch_directory = output / f"epoch_{index:03d}"
        epoch_directory.mkdir(parents=True, exist_ok=True)
        epoch_directories.append(epoch_directory.name)
        optical = epoch.raw_frames["OPTICAL"].data
        infrared = epoch.raw_frames["INFRARED"].data
        radar = epoch.raw_frames["RADAR"].data
        radar_frame = epoch.raw_frames["RADAR"]
        sources = epoch.source_products
        np.savez_compressed(
            epoch_directory / "physical_sources.npz",
            optical_relative_reflected_radiance=(
                sources.optical_relative_reflected_radiance
            ),
            infrared_temperature_k=sources.infrared_temperature_k,
            infrared_spectral_radiance_w_m3_sr=(
                sources.infrared_spectral_radiance_w_m3_sr
            ),
            radar_scatterer_table=sources.radar_scatterer_table,
            radar_iq_real=sources.radar_iq.real,
            radar_iq_imag=sources.radar_iq.imag,
        )
        np.savez_compressed(
            epoch_directory / "sensor_raw.npz",
            optical_detector_frame=optical,
            infrared_detector_frame=infrared,
            radar_iq=sources.radar_iq,
            radar_range_doppler=radar,
            radar_range_axis_m=radar_frame.axes["range_m"],
            radar_range_rate_axis_mps=radar_frame.axes["range_rate_mps"],
        )
        np.savez_compressed(
            epoch_directory / "presentation_enhanced.npz",
            optical=epoch.presentation_frames["OPTICAL"],
            infrared=epoch.presentation_frames["INFRARED"],
            radar_range_doppler=epoch.presentation_frames["RADAR"],
        )
        source_state = {
            "schemaVersion": "tango-like-source-state.v1",
            "modelId": model.model_id,
            "modelProvenance": model.provenance,
            "geometry": _jsonable(sources.geometry),
            "materials": [
                {
                    "name": item.name,
                    "opticalAlbedo": item.optical_albedo,
                    "opticalSpecular": item.optical_specular,
                    "infraredEmissivity": item.infrared_emissivity,
                    "baseTemperatureK": item.base_temperature_k,
                    "sunTemperatureDeltaK": item.sun_temperature_delta_k,
                }
                for item in model.materials
            ],
            "radarScattererColumns": [
                "range_m", "range_rate_mps", "amplitude", "rd_row",
                "rd_column", "position_eci_x_m", "position_eci_y_m",
                "position_eci_z_m",
            ],
            "radarScattererLabels": list(sources.radar_scatterer_labels),
        }
        (epoch_directory / "source_state.json").write_text(
            json.dumps(source_state, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        measurements = {
            "schemaVersion": "tango-like-extracted-measurements.v1",
            "measurements": [
                {
                    "modality": message.modality,
                    "measurement": message.measurement.tolist(),
                    "covariance": message.covariance.tolist(),
                    "valid": bool(message.valid_flag),
                    "measurementType": message.metadata.get("measurement_type"),
                    "sourceProduct": message.metadata.get("raw_source_type"),
                }
                for message in epoch.messages
            ],
        }
        (epoch_directory / "extracted_measurements.json").write_text(
            json.dumps(measurements, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        distance_records.append(_distance_record(index, epoch))
        preview = epoch_directory / "source_to_sensor_preview.png"
        _save_trace_preview(epoch, preview)
        preview_files.append(str(preview.relative_to(output)))

    transition_summary = output / "distance_transition_summary.png"
    _save_distance_transition_summary(epochs, transition_summary)
    trend_plot = output / "measurement_distance_trends.png"
    _save_measurement_distance_trends(distance_records, trend_plot)
    records_json = output / "distance_measurements.json"
    records_json.write_text(
        json.dumps(distance_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    records_csv = output / "distance_measurements.csv"
    with records_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(distance_records[0]))
        writer.writeheader()
        writer.writerows(distance_records)
    checks = _distance_response_generation_checks(
        distance_records, epoch_directories, output
    )
    test_summary = {
        "schemaVersion": "multimodal-distance-response-test.v1",
        "testName": "F11 three-modality measurement distance response",
        "overallPassed": bool(all(item["passed"] for item in checks)),
        "passedCount": int(sum(item["passed"] for item in checks)),
        "checkCount": len(checks),
        "sampleCount": len(distance_records),
        "minimumDistanceM": float(distance_records[0]["distance_m"]),
        "maximumDistanceM": float(distance_records[-1]["distance_m"]),
        "modalities": ["RADAR", "INFRARED", "OPTICAL"],
        "checks": checks,
        "recordsJson": records_json.name,
        "recordsCsv": records_csv.name,
        "trendPlot": trend_plot.name,
        "transitionPreview": transition_summary.name,
        "manualVisualReviewRequired": True,
        "evaluationScope": "FUNCTIONAL_DATA_GENERATION_AND_VISUAL_REVIEW_ONLY",
        "accuracyThresholdsApplied": False,
    }
    (output / "test_summary.json").write_text(
        json.dumps(test_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (output / "README.txt").write_text(
        "Tango-like three-modality trace bundle\n\n"
        "Each epoch directory contains:\n"
        "  source_state.json          geometry, materials and provenance\n"
        "  physical_sources.npz       reflected radiance, temperature/radiance, scatterers and IQ\n"
        "  sensor_raw.npz             optical/IR detector frames and radar IQ/range-Doppler\n"
        "  presentation_enhanced.npz  display-only products; never supplied to the estimator\n"
        "  extracted_measurements.json measurements supplied to the estimator\n"
        "  source_to_sensor_preview.png fixed 2x3 source-to-sensor explanation\n\n"
        "The root directory additionally contains distance_measurements.csv,\n"
        "distance_measurements.json, measurement_distance_trends.png and\n"
        "test_summary.json for the distance-response display test.\n\n"
        "The upper row is an explanatory physical-source view, not a detector image.\n"
        "The lower row contains the sensor products used for measurement extraction.\n\n"
        "Presentation-enhanced products are display-only logical magnifications.\n"
        "They never replace sensor_raw.npz and never enter measurement extraction.\n\n"
        "Detector rendering changes continuously with projected target extent:\n"
        "  >= 16 px       resolved extended target\n"
        "  0.25 to 16 px  smooth extended-target / PSF transition\n"
        "  <= 0.25 px     unresolved point source\n",
        encoding="utf-8",
    )
    manifest = {
        "schemaVersion": "tango-like-physical-frontend-demo.v2",
        "modelId": model.model_id,
        "modelProvenance": model.provenance,
        "physicalFidelity": "SIMPLIFIED_FUNCTIONAL_DEMONSTRATION",
        "frameCount": frame_count,
        "initialDistanceM": distance_m,
        "maximumDistanceM": maximum_distance_m,
        "detectorTransition": {
            "focalLengthPixels": focal_length_pixels,
            "unresolvedExtentPixels": unresolved_extent_pixels,
            "resolvedExtentPixels": resolved_extent_pixels,
            "maximumObservationRangeKm": maximum_observation_range_km,
            "displayMinimumExtentPixels": display_minimum_extent_pixels,
            "displayMaximumExtentPixels": display_maximum_extent_pixels,
        },
        "seed": seed,
        "processingChain": {
            "optical": "surface reflection -> detector PSF/noise -> centroid -> normalized image coordinates",
            "infrared": "surface temperature/emissivity -> 10 um radiance -> detector PSF/noise -> centroid -> azimuth/elevation",
            "radar": "aspect-dependent scatterers -> complex baseband IQ -> 2-D FFT range-Doppler -> centroid -> range/range-rate",
        },
        "epochDirectories": epoch_directories,
        "previews": preview_files,
        "distanceTransitionSummary": transition_summary.name,
        "measurementDistanceTrends": trend_plot.name,
        "distanceMeasurementRecords": records_csv.name,
        "testSummary": "test_summary.json",
        "readme": "README.txt",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    if show:
        image = plt.imread(
            transition_summary
            if maximum_distance_m is not None else output / preview_files[0]
        )
        figure, axis = plt.subplots(figsize=(12.5, 7.5), constrained_layout=True)
        axis.imshow(image)
        axis.axis("off")
        plt.show()
    return output


def _distance_record(index, epoch):
    messages = {item.modality: item for item in epoch.messages}
    geometry = epoch.source_products.geometry
    relative = np.asarray(geometry["relative_position_eci_m"], dtype=float)
    normalized = relative[1:] / relative[0]
    ideal_infrared = np.array([
        np.arctan(normalized[0]),
        np.arctan2(normalized[1], np.sqrt(1.0 + normalized[0] ** 2)),
    ])
    radar_truth = np.array([
        geometry["range_m"], geometry["range_rate_mps"],
    ])
    optical = messages["OPTICAL"]
    infrared = messages["INFRARED"]
    radar = messages["RADAR"]
    return {
        "sample_index": int(index),
        "distance_m": float(geometry["range_m"]),
        "range_rate_truth_mps": float(geometry["range_rate_mps"]),
        "within_observation_range": bool(
            geometry["within_observation_range"]
        ),
        "optical_valid": bool(optical.valid_flag),
        "optical_u": float(optical.measurement[0]),
        "optical_v": float(optical.measurement[1]),
        "optical_angle_error_rad": float(
            np.linalg.norm(optical.measurement - normalized)
        ),
        "optical_raw_contrast": _raw_contrast(
            epoch.raw_frames["OPTICAL"].data
        ),
        "optical_projected_extent_px": float(
            geometry["optical_projected_extent_pixels"]
        ),
        "infrared_valid": bool(infrared.valid_flag),
        "infrared_azimuth_rad": float(infrared.measurement[0]),
        "infrared_elevation_rad": float(infrared.measurement[1]),
        "infrared_angle_error_rad": float(
            np.linalg.norm(infrared.measurement - ideal_infrared)
        ),
        "infrared_raw_contrast": _raw_contrast(
            epoch.raw_frames["INFRARED"].data
        ),
        "infrared_projected_extent_px": float(
            geometry["infrared_projected_extent_pixels"]
        ),
        "radar_valid": bool(radar.valid_flag),
        "radar_range_m": float(radar.measurement[0]),
        "radar_range_rate_mps": float(radar.measurement[1]),
        "radar_range_error_m": float(radar.measurement[0] - radar_truth[0]),
        "radar_range_rate_error_mps": float(
            radar.measurement[1] - radar_truth[1]
        ),
        "radar_raw_contrast": _raw_contrast(
            epoch.raw_frames["RADAR"].data
        ),
        "display_extent_px": float(
            geometry["presentation_display_extent_pixels"]
        ),
        "resolution_regime": str(geometry["optical_resolution_regime"]),
    }


def _raw_contrast(values):
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite) - np.median(finite))


def _distance_response_generation_checks(records, epoch_directories, output):
    distances = np.asarray([item["distance_m"] for item in records])
    all_valid = all(
        item[f"{modality}_valid"]
        for item in records
        for modality in ("radar", "infrared", "optical")
        if item["within_observation_range"]
    )
    trace_files = (
        "physical_sources.npz", "sensor_raw.npz",
        "presentation_enhanced.npz", "extracted_measurements.json",
    )
    traces_recorded = all(
        all((Path(output) / name / filename).is_file() for filename in trace_files)
        for name in epoch_directories
    )
    return [
        {
            "name": "DISTANCE_SEQUENCE_STRICTLY_INCREASING",
            "passed": bool(np.all(np.diff(distances) > 0.0)),
        },
        {
            "name": "THREE_MODALITIES_GENERATED_AT_EACH_RANGE",
            "passed": bool(all_valid),
        },
        {
            "name": "SOURCE_RAW_MEASUREMENT_TRACE_FILES_RECORDED",
            "passed": bool(traces_recorded),
        },
    ]


def _save_measurement_distance_trends(records, path):
    distance_km = np.asarray([item["distance_m"] for item in records]) / 1000.0
    figure, axes = plt.subplots(4, 1, figsize=(9.5, 12.0), constrained_layout=True)
    for modality, color in (
        ("optical", "#9e9e9e"),
        ("infrared", "#ff7f0e"),
        ("radar", "#27b7d8"),
    ):
        contrast = np.asarray([
            item[f"{modality}_raw_contrast"] for item in records
        ])
        reference = max(float(contrast[0]), 1.0e-12)
        axes[0].plot(
            distance_km, contrast / reference, "o-",
            color=color, label=modality.upper(),
        )
    axes[0].set_ylabel("normalized raw contrast")
    axes[0].set_title("Three-modality raw response versus distance")
    axes[0].legend()
    axes[1].plot(
        distance_km,
        [item["optical_u"] for item in records],
        "o-", color="#9e9e9e", label="image u",
    )
    axes[1].plot(
        distance_km,
        [item["optical_v"] for item in records],
        "s-", color="#d0d0d0", label="image v",
    )
    axes[1].set_ylabel("normalized image coordinate")
    axes[1].set_title("Optical extracted measurement versus distance")
    axes[1].legend()
    axes[2].plot(
        distance_km,
        [item["infrared_azimuth_rad"] for item in records],
        "o-", color="#ff7f0e", label="azimuth",
    )
    axes[2].plot(
        distance_km,
        [item["infrared_elevation_rad"] for item in records],
        "s-", color="#ffb060", label="elevation",
    )
    axes[2].set_ylabel("angle (rad)")
    axes[2].set_title("Infrared extracted measurement versus distance")
    axes[2].legend()
    axes[3].plot(
        distance_km,
        np.asarray([item["radar_range_m"] for item in records]) / 1000.0,
        "o-", color="#27b7d8", label="range",
    )
    axes[3].set_ylabel("measured range (km)")
    radar_rate_axis = axes[3].twinx()
    radar_rate_axis.plot(
        distance_km,
        [item["radar_range_rate_mps"] for item in records],
        "s--", color="#80d8e8", label="range rate",
    )
    radar_rate_axis.set_ylabel("measured range rate (m/s)")
    axes[3].set_xlabel("distance (km, logarithmic scale)")
    axes[3].set_title("Radar extracted measurement versus distance")
    axes[3].legend(loc="upper left")
    radar_rate_axis.legend(loc="lower right")
    for axis in axes:
        axis.set_xscale("log")
        axis.grid(True, which="both", alpha=0.25)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _save_distance_transition_summary(epochs, path):
    column_count = len(epochs)
    figure, axes = plt.subplots(
        4, column_count, figsize=(3.0 * column_count, 11.2),
        squeeze=False, constrained_layout=True,
    )
    optical_frames = [item.raw_frames["OPTICAL"].data for item in epochs]
    infrared_frames = [item.raw_frames["INFRARED"].data for item in epochs]
    optical_enhanced = [item.presentation_frames["OPTICAL"] for item in epochs]
    infrared_enhanced = [item.presentation_frames["INFRARED"] for item in epochs]
    optical_min = float(min(np.percentile(item, 1.0) for item in optical_frames))
    optical_max = float(max(np.max(item) for item in optical_frames))
    infrared_min = float(min(np.percentile(item, 1.0) for item in infrared_frames))
    infrared_max = float(max(np.max(item) for item in infrared_frames))
    for index, epoch in enumerate(epochs):
        geometry = epoch.source_products.geometry
        axes[0, index].imshow(
            optical_frames[index], cmap="gray", origin="upper",
            vmin=optical_min, vmax=optical_max,
        )
        axes[1, index].imshow(
            infrared_frames[index], cmap="inferno", origin="upper",
            vmin=infrared_min, vmax=infrared_max,
        )
        axes[2, index].imshow(
            optical_enhanced[index], cmap="gray", origin="upper",
            vmin=optical_min, vmax=optical_max,
        )
        axes[3, index].imshow(
            infrared_enhanced[index], cmap="inferno", origin="upper",
            vmin=infrared_min, vmax=infrared_max,
        )
        axes[0, index].set_title(
            f"range={geometry['range_m']:.3g} m\n"
            f"extent={geometry['optical_projected_extent_pixels']:.3g} px"
        )
        for axis in axes[:, index]:
            axis.set_xticks([])
            axis.set_yticks([])
    axes[0, 0].set_ylabel("Optical detector")
    axes[1, 0].set_ylabel("Infrared detector")
    axes[2, 0].set_ylabel("Optical display enhanced")
    axes[3, 0].set_ylabel("Infrared display enhanced")
    figure.suptitle(
        "Distance-dependent resolved-to-point-source transition (shared scale)",
        fontsize=14,
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _save_trace_preview(epoch, path):
    sources = epoch.source_products
    figure, axes = plt.subplots(3, 3, figsize=(13.0, 10.4), constrained_layout=True)
    optical_source = sources.optical_relative_reflected_radiance
    axes[0, 0].imshow(optical_source, cmap="gray", origin="upper")
    axes[0, 0].set_title("Optical source: relative reflected radiance")
    temperature = np.ma.masked_invalid(sources.infrared_temperature_k)
    axes[0, 1].imshow(temperature, cmap="inferno", origin="upper")
    axes[0, 1].set_title("Infrared physical source: surface temperature (K)")
    table = sources.radar_scatterer_table
    center_range = float(epoch.concrete_frames["RADAR"].acquisition_center[0])
    center_rate = float(epoch.concrete_frames["RADAR"].acquisition_center[1])
    axes[0, 2].scatter(
        1000.0 * (table[:, 0] - center_range),
        1000.0 * (table[:, 1] - center_rate),
        s=25.0 + 90.0 * table[:, 2] / max(float(np.max(table[:, 2])), 1.0e-12),
        c=table[:, 2], cmap="viridis", edgecolors="white", linewidths=0.5,
    )
    axes[0, 2].set_title("Radar physical source: aspect-dependent scatterers")
    axes[0, 2].set_xlabel("range offset (mm)")
    axes[0, 2].set_ylabel("range-rate offset (mm/s)")

    axes[1, 0].imshow(epoch.raw_frames["OPTICAL"].data, cmap="gray", origin="upper")
    axes[1, 0].set_title("Optical detector raw frame")
    axes[1, 1].imshow(epoch.raw_frames["INFRARED"].data, cmap="inferno", origin="upper")
    axes[1, 1].set_title("Infrared detector raw frame")
    radar_frame = epoch.raw_frames["RADAR"]
    axes[1, 2].imshow(
        radar_frame.data, cmap="gray", origin="lower", aspect="auto",
        extent=[
            radar_frame.axes["range_m"][0], radar_frame.axes["range_m"][-1],
            radar_frame.axes["range_rate_mps"][0],
            radar_frame.axes["range_rate_mps"][-1],
        ],
    )
    axes[1, 2].set_title("Radar sensor product: range-Doppler power")
    axes[1, 2].set_xlabel("range (m)")
    axes[1, 2].set_ylabel("range rate (m/s)")
    axes[2, 0].imshow(
        epoch.presentation_frames["OPTICAL"], cmap="gray", origin="upper",
    )
    axes[2, 0].set_title("Display-enhanced optical (not used by filter)")
    axes[2, 1].imshow(
        epoch.presentation_frames["INFRARED"], cmap="inferno", origin="upper",
    )
    axes[2, 1].set_title("Display-enhanced infrared (not used by filter)")
    axes[2, 2].imshow(
        epoch.presentation_frames["RADAR"], cmap="gray", origin="lower",
        aspect="auto",
    )
    axes[2, 2].set_title("Display-enhanced radar contrast (not used by filter)")
    for axis in (
        axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1],
    ):
        axis.set_xlabel("pixel x")
        axis.set_ylabel("pixel y")
    figure.suptitle(
        "Physical source -> sensor raw -> display-only enhancement\n"
        f"range={sources.geometry['range_m']:.3g} m | "
        f"optical extent={sources.geometry['optical_projected_extent_pixels']:.3g} px | "
        f"{sources.geometry['optical_resolution_regime']}",
        fontsize=13,
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/physical_frontend/tango_like_demo"),
    )
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--distance-m", type=float, default=8.0)
    parser.add_argument(
        "--maximum-distance-m", type=float, default=None,
        help="If supplied, use logarithmically spaced ranges from --distance-m.",
    )
    parser.add_argument("--focal-length-pixels", type=float, default=1600.0)
    parser.add_argument("--unresolved-extent-pixels", type=float, default=0.25)
    parser.add_argument("--resolved-extent-pixels", type=float, default=16.0)
    parser.add_argument(
        "--maximum-observation-range-km", type=float, default=6000.0,
    )
    parser.add_argument(
        "--display-minimum-extent-pixels", type=float, default=2.0,
    )
    parser.add_argument(
        "--display-maximum-extent-pixels", type=float, default=32.0,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args(argv)
    output = run_demo(
        output=args.output, frame_count=args.frames,
        distance_m=args.distance_m, maximum_distance_m=args.maximum_distance_m,
        focal_length_pixels=args.focal_length_pixels,
        unresolved_extent_pixels=args.unresolved_extent_pixels,
        resolved_extent_pixels=args.resolved_extent_pixels,
        maximum_observation_range_km=args.maximum_observation_range_km,
        display_minimum_extent_pixels=args.display_minimum_extent_pixels,
        display_maximum_extent_pixels=args.display_maximum_extent_pixels,
        seed=args.seed, show=not args.no_show,
    )
    print(f"Tango-like physical front-end demo written to {output}")


if __name__ == "__main__":
    main()
