from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PointSourceImageConfig:
    """Shared detector settings for optical and infrared point sources."""

    width: int = 512
    height: int = 512
    principal_x_pixels: float | None = None
    principal_y_pixels: float | None = None
    psf_sigma_pixels: float = 1.2
    source_peak: float = 1000.0
    background: float = 10.0
    read_noise_sigma: float = 1.0
    detection_sigma: float = 5.0
    centroid_edge_margin_sigma: float = 4.0

    def validate_image_settings(self):
        if self.width < 3 or self.height < 3:
            raise ValueError("Point-source images must be at least 3 pixels wide.")
        positive = (self.psf_sigma_pixels, self.source_peak)
        if np.any(~np.isfinite(positive)) or min(positive) <= 0.0:
            raise ValueError("Point-source scales must be finite and positive.")
        nonnegative = (
            self.background, self.read_noise_sigma, self.detection_sigma,
            self.centroid_edge_margin_sigma,
        )
        if np.any(~np.isfinite(nonnegative)) or min(nonnegative) < 0.0:
            raise ValueError("Point-source noise must be finite and nonnegative.")
        if not 0.0 <= self.principal_x <= self.width - 1:
            raise ValueError("Principal x must lie inside the image.")
        if not 0.0 <= self.principal_y <= self.height - 1:
            raise ValueError("Principal y must lie inside the image.")

    @property
    def principal_x(self):
        return (
            0.5 * (self.width - 1)
            if self.principal_x_pixels is None
            else float(self.principal_x_pixels)
        )

    @property
    def principal_y(self):
        return (
            0.5 * (self.height - 1)
            if self.principal_y_pixels is None
            else float(self.principal_y_pixels)
        )


def render_gaussian_point_source(pixel_xy, *, config, rng=None):
    """Render a Gaussian point source and return image plus in-frame flag."""
    config.validate_image_settings()
    pixel = np.asarray(pixel_xy, dtype=float).reshape(2)
    margin = config.centroid_edge_margin_sigma * config.psf_sigma_pixels
    in_frame = bool(
        np.all(np.isfinite(pixel))
        and margin <= pixel[0] <= config.width - 1 - margin
        and margin <= pixel[1] <= config.height - 1 - margin
    )
    rows, columns = np.indices((config.height, config.width), dtype=float)
    image = np.full(
        (config.height, config.width), config.background, dtype=float,
    )
    if np.all(np.isfinite(pixel)):
        squared_distance = (
            (columns - pixel[0]) ** 2 + (rows - pixel[1]) ** 2
        )
        image += config.source_peak * np.exp(
            -0.5 * squared_distance / config.psf_sigma_pixels**2
        )
    if config.read_noise_sigma > 0.0:
        generator = np.random.default_rng() if rng is None else rng
        image += generator.normal(0.0, config.read_noise_sigma, image.shape)
    return image, in_frame


def extract_point_source_centroid(image, *, config, target_in_frame=True):
    """Extract a local-window, background-subtracted subpixel centroid."""
    values = np.asarray(image, dtype=float)
    if values.shape != (config.height, config.width):
        raise ValueError("Point-source frame shape does not match its config.")
    if not target_in_frame or np.any(~np.isfinite(values)):
        return np.array([np.nan, np.nan]), False
    peak_row, peak_column = np.unravel_index(np.argmax(values), values.shape)
    peak_signal = float(values[peak_row, peak_column] - config.background)
    if peak_signal <= config.detection_sigma * config.read_noise_sigma:
        return np.array([np.nan, np.nan]), False
    radius = max(2, int(np.ceil(4.0 * config.psf_sigma_pixels)))
    row_slice = slice(
        max(0, peak_row - radius), min(config.height, peak_row + radius + 1),
    )
    column_slice = slice(
        max(0, peak_column - radius), min(config.width, peak_column + radius + 1),
    )
    patch = np.maximum(
        values[row_slice, column_slice] - config.background, 0.0,
    )
    total = float(np.sum(patch))
    if total <= 0.0:
        return np.array([np.nan, np.nan]), False
    rows, columns = np.indices(patch.shape, dtype=float)
    rows += row_slice.start
    columns += column_slice.start
    return np.array([
        float(np.sum(columns * patch) / total),
        float(np.sum(rows * patch) / total),
    ]), True
