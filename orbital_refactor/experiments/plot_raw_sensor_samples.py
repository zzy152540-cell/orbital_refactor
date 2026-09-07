from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adapters.infrared_image_adapter import (
    InfraredCameraConfig,
    infrared_frame_to_observation_message,
    render_infrared_point_source_frame,
)
from adapters.optical_image_adapter import (
    OpticalCameraConfig,
    optical_frame_to_observation_message,
    render_optical_point_source_frame,
)
from adapters.radar_range_doppler_adapter import (
    RadarRangeDopplerConfig,
    radar_frame_to_observation_message,
    render_radar_range_doppler_frame,
)
from experiments.inter_satellite_observation_factory import (
    target_pointing_quaternion,
)
from experiments.walker_raw_sensor_comparison import _walker_case
from orbital_core.measurements import (
    measure_relative_range,
    measure_relative_range_rate,
)


def plot_raw_sensor_samples(
    output_directory="results/raw_sensor_samples", *, show_markers=False,
):
    """Render one representative matrix from each simulated sensor front end."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    case = _walker_case(seed=0, duration=20.0, dt=2.0)

    optical_source = _first_message(case, "OPTICAL")
    infrared_source = _first_message(case, "INFRARED")
    radar_source = _first_message(case, "RADAR")

    optical_config = OpticalCameraConfig(
        width=128, height=128,
        focal_length_x_pixels=100.0, focal_length_y_pixels=100.0,
        source_peak=100.0, background=10.0, read_noise_sigma=10.0,
        reported_centroid_sigma_pixels=0.25,
    )
    infrared_config = InfraredCameraConfig(
        width=128, height=128,
        focal_length_x_pixels=300.0, focal_length_y_pixels=300.0,
        source_peak=100.0, background=10.0, read_noise_sigma=10.0,
        reported_centroid_sigma_pixels=0.26,
    )
    radar_config = RadarRangeDopplerConfig(
        width=129, height=129, range_bin_size_m=8.0,
        range_rate_bin_size_mps=0.2,
        source_peak=100.0, background=10.0, read_noise_sigma=10.0,
        reported_centroid_sigma_bins=0.19,
    )

    observer, target = _states_for_message(case, optical_source)
    optical_frame = render_optical_point_source_frame(
        timestamp=optical_source.timestamp,
        observer_id=optical_source.observer_id,
        target_id=optical_source.target_id,
        observer_state=observer, target_state=target,
        quaternion_i2b_wxyz=np.asarray(
            optical_source.metadata["quaternion_i2b_wxyz"], dtype=float,
        ),
        config=optical_config, rng=np.random.default_rng(11),
    )
    optical_message = optical_frame_to_observation_message(
        optical_frame, config=optical_config,
    )

    observer, target = _states_for_message(case, infrared_source)
    infrared_frame = render_infrared_point_source_frame(
        timestamp=infrared_source.timestamp,
        observer_id=infrared_source.observer_id,
        target_id=infrared_source.target_id,
        observer_state=observer, target_state=target,
        quaternion_i2b_wxyz=target_pointing_quaternion(observer, target),
        config=infrared_config, rng=np.random.default_rng(12),
    )
    infrared_message = infrared_frame_to_observation_message(
        infrared_frame, config=infrared_config,
    )

    observer, target = _states_for_message(case, radar_source)
    ideal_radar = np.array([
        measure_relative_range(observer, target),
        measure_relative_range_rate(observer, target),
    ])
    radar_frame = render_radar_range_doppler_frame(
        timestamp=radar_source.timestamp,
        observer_id=radar_source.observer_id,
        target_id=radar_source.target_id,
        observer_state=observer, target_state=target,
        acquisition_range_m=ideal_radar[0] + 80.0,
        acquisition_range_rate_mps=ideal_radar[1] + 1.0,
        config=radar_config, rng=np.random.default_rng(13),
    )
    radar_message = radar_frame_to_observation_message(
        radar_frame, config=radar_config,
    )

    samples = (
        (
            "Optical focal-plane image", optical_frame.image,
            optical_frame.ideal_pixel_xy,
            optical_message.metadata["centroid_pixel_xy"],
            "x pixel", "y pixel", "magma",
        ),
        (
            "Infrared focal-plane image", infrared_frame.image,
            infrared_frame.ideal_pixel_xy,
            infrared_message.metadata["centroid_pixel_xy"],
            "x pixel", "y pixel", "inferno",
        ),
        (
            "Radar range-Doppler power map", radar_frame.power,
            radar_frame.ideal_bin_xy,
            radar_message.metadata["centroid_bin_xy"],
            "range bin", "range-rate bin", "viridis",
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    for axis, sample in zip(axes, samples):
        _draw_sample(axis, *sample, show_markers=show_markers)
    figure.suptitle(
        "Simulated raw sensor matrices - Walker 20-satellite case",
        fontsize=15,
    )
    overview_path = output / "raw_sensor_matrix_overview.png"
    figure.savefig(overview_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    individual_paths = []
    for file_stem, sample in zip(("optical", "infrared", "radar"), samples):
        figure, axis = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
        _draw_sample(axis, *sample, show_markers=show_markers)
        path = output / f"raw_{file_stem}_sample.png"
        figure.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        individual_paths.append(path)
    return overview_path, tuple(individual_paths)


def _draw_sample(
    axis, title, values, ideal, centroid, xlabel, ylabel, cmap,
    *, show_markers,
):
    image = axis.imshow(values, origin="lower", cmap=cmap, interpolation="nearest")
    if show_markers:
        axis.scatter(*ideal, marker="x", s=90, linewidths=2.0,
                     color="cyan", label="ideal location")
        axis.scatter(*centroid, marker="+", s=110, linewidths=2.0,
                     color="lime", label="extracted centroid")
        axis.legend(loc="upper right", fontsize=8)
    axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04,
                         label="simulated intensity / power")


def _first_message(case, modality):
    return next(
        item for item in case["observations"]
        if item.modality.upper() == modality
    )


def _states_for_message(case, message):
    index = int(np.flatnonzero(np.isclose(
        case["timestamps"], message.timestamp,
    ))[0])
    return (
        case["truth"][message.observer_id][index],
        case["truth"][message.target_id][index],
    )


if __name__ == "__main__":
    overview, individuals = plot_raw_sensor_samples()
    print(overview)
    for path in individuals:
        print(path)
