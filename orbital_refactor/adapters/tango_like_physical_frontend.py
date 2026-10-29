"""Lightweight physically motivated three-modality Tango-like front end.

The model is intentionally approximate.  It uses public principal dimensions
of the PRISMA Tango target and simple material regions, not the unavailable
official SPEED/SPEED+ CAD model.  Its purpose is to provide traceable raw
optical, thermal-infrared, and radar products without changing the downstream
``ObservationMessage`` contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.ndimage import gaussian_filter

from adapters.infrared_image_adapter import (
    InfraredCameraConfig,
    InfraredImageFrame,
    infrared_pixel_to_az_el,
)
from adapters.optical_image_adapter import OpticalCameraConfig, OpticalImageFrame
from adapters.radar_range_doppler_adapter import (
    RadarRangeDopplerConfig,
    RadarRangeDopplerFrame,
    radar_frame_to_observation_message,
)
from adapters.raw_sensor_frame_adapter import (
    raw_sensor_frame_from_infrared,
    raw_sensor_frame_from_optical,
    raw_sensor_frame_from_radar,
)
from interfaces.data_objects import ObservationMessage, RawSensorFrame
from orbital_core.attitude import quat_to_dcm_i2b
from orbital_core.measurements import (
    measure_relative_range,
    measure_relative_range_rate,
)


@dataclass(frozen=True)
class TangoLikeMaterial:
    name: str
    optical_albedo: float
    optical_specular: float
    infrared_emissivity: float
    base_temperature_k: float
    sun_temperature_delta_k: float


@dataclass(frozen=True)
class RadarScatterer:
    position_body_m: np.ndarray
    normal_body: np.ndarray
    strength: float
    label: str


@dataclass(frozen=True)
class TangoLikeModel:
    vertices_body_m: np.ndarray
    faces: np.ndarray
    face_material_indices: np.ndarray
    materials: tuple[TangoLikeMaterial, ...]
    radar_scatterers: tuple[RadarScatterer, ...]
    model_id: str = "tango-like-public-dimensions-v1"
    provenance: str = (
        "Approximate model: 0.56 x 0.55 x 0.30 m main body, "
        "0.56 x 0.75 m body-mounted panel, 0.204 m appendages."
    )


@dataclass(frozen=True)
class PhysicalFrontendConfig:
    optical: OpticalCameraConfig = field(
        default_factory=lambda: OpticalCameraConfig(
            width=256, height=256,
            focal_length_x_pixels=1600.0, focal_length_y_pixels=1600.0,
            source_peak=220.0, background=4.0, read_noise_sigma=1.0,
            psf_sigma_pixels=1.0, reported_centroid_sigma_pixels=0.35,
        )
    )
    infrared: InfraredCameraConfig = field(
        default_factory=lambda: InfraredCameraConfig(
            width=256, height=256,
            focal_length_x_pixels=1600.0, focal_length_y_pixels=1600.0,
            source_peak=220.0, background=5.0, read_noise_sigma=1.2,
            psf_sigma_pixels=1.2, reported_centroid_sigma_pixels=0.45,
        )
    )
    radar: RadarRangeDopplerConfig = field(
        default_factory=lambda: RadarRangeDopplerConfig(
            width=129, height=129,
            range_bin_size_m=0.015, range_rate_bin_size_mps=0.01,
            source_peak=150.0, background=2.0, read_noise_sigma=0.8,
            psf_sigma_pixels=1.0, reported_centroid_sigma_bins=0.35,
        )
    )
    sun_direction_eci: tuple[float, float, float] = (0.4, -0.3, 0.866)
    optical_ambient_fraction: float = 0.08
    thermal_reference_temperature_k: float = 330.0
    thermal_wavelength_um: float = 10.0
    unresolved_extent_pixels: float = 0.25
    resolved_extent_pixels: float = 16.0
    source_display_extent_pixels: float = 100.0
    radiometric_reference_distance_m: float = 3.0e6
    maximum_observation_range_m: float = 6.0e6
    display_minimum_extent_pixels: float = 2.0
    display_maximum_extent_pixels: float = 32.0
    display_range_exponent: float = 1.5

    def __post_init__(self):
        sun = np.asarray(self.sun_direction_eci, dtype=float).reshape(3)
        if not np.all(np.isfinite(sun)) or np.linalg.norm(sun) <= 0.0:
            raise ValueError("sun_direction_eci must be finite and nonzero.")
        if not 0.0 <= self.optical_ambient_fraction <= 1.0:
            raise ValueError("optical_ambient_fraction must lie in [0, 1].")
        if self.thermal_reference_temperature_k <= 0.0:
            raise ValueError("thermal_reference_temperature_k must be positive.")
        if self.thermal_wavelength_um <= 0.0:
            raise ValueError("thermal_wavelength_um must be positive.")
        if not 0.0 < self.unresolved_extent_pixels < self.resolved_extent_pixels:
            raise ValueError("Resolution-transition pixel extents are invalid.")
        if self.source_display_extent_pixels <= 0.0:
            raise ValueError("source_display_extent_pixels must be positive.")
        if self.radiometric_reference_distance_m <= 0.0:
            raise ValueError("radiometric_reference_distance_m must be positive.")
        if self.maximum_observation_range_m <= 0.0:
            raise ValueError("maximum_observation_range_m must be positive.")
        if not 0.0 < self.display_minimum_extent_pixels <= (
            self.display_maximum_extent_pixels
        ):
            raise ValueError("Display-enhancement extents are invalid.")
        if self.display_range_exponent <= 0.0:
            raise ValueError("display_range_exponent must be positive.")


@dataclass(frozen=True)
class PhysicalMultimodalEpoch:
    messages: tuple[ObservationMessage, ...]
    raw_frames: dict[str, RawSensorFrame]
    concrete_frames: dict[str, object]
    source_products: "PhysicalSourceProducts"
    presentation_frames: dict[str, np.ndarray]
    model_id: str


@dataclass(frozen=True)
class PhysicalSourceProducts:
    """Products upstream of the sensor detector/receiver models.

    These arrays are deliberately kept separate from ``raw_frames``.  They
    explain where each sensor product came from without pretending that a
    magnified surface-property view is a scale-correct detector image.
    """

    optical_relative_reflected_radiance: np.ndarray
    infrared_temperature_k: np.ndarray
    infrared_spectral_radiance_w_m3_sr: np.ndarray
    radar_scatterer_table: np.ndarray
    radar_scatterer_labels: tuple[str, ...]
    radar_iq: np.ndarray
    geometry: dict[str, object]


def build_tango_like_model() -> TangoLikeModel:
    """Build a primitive-only, redistributable approximation of Tango."""
    materials = (
        TangoLikeMaterial("main_body", 0.72, 0.18, 0.82, 270.0, 25.0),
        TangoLikeMaterial("solar_panel", 0.18, 0.08, 0.88, 250.0, 18.0),
        TangoLikeMaterial("metal_appendage", 0.55, 0.55, 0.28, 245.0, 12.0),
    )
    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    material_indices: list[int] = []
    _append_box(
        vertices, faces, material_indices,
        center=(0.0, 0.0, 0.0), size=(0.30, 0.56, 0.55), material=0,
    )
    # A shallow body-mounted panel on the +X face.  The dimensions follow the
    # published simplified reconstruction, while placement is approximate.
    _append_box(
        vertices, faces, material_indices,
        center=(0.157, 0.0, 0.0), size=(0.014, 0.75, 0.56), material=1,
    )
    antenna_centers = (
        (-0.252, -0.22, -0.20), (-0.252, 0.22, -0.20),
        (-0.252, -0.22, 0.20), (-0.252, 0.22, 0.20),
    )
    for center in antenna_centers:
        _append_box(
            vertices, faces, material_indices,
            center=center, size=(0.204, 0.014, 0.014), material=2,
        )

    body_half = np.array([0.15, 0.28, 0.275])
    corner_signs = np.array([
        [sx, sy, sz]
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ])
    scatterers = [
        RadarScatterer(np.zeros(3), np.array([1.0, 0.0, 0.0]), 3.0, "body"),
    ]
    for index, sign in enumerate(corner_signs):
        position = sign * body_half
        scatterers.append(RadarScatterer(
            position, sign / np.linalg.norm(sign), 0.65, f"body_corner_{index}",
        ))
    for index, position in enumerate((
        (0.164, -0.375, -0.28), (0.164, -0.375, 0.28),
        (0.164, 0.375, -0.28), (0.164, 0.375, 0.28),
    )):
        scatterers.append(RadarScatterer(
            np.asarray(position, dtype=float), np.array([1.0, 0.0, 0.0]),
            0.9, f"panel_corner_{index}",
        ))
    for index, center in enumerate(antenna_centers):
        tip = np.asarray(center, dtype=float) + np.array([-0.102, 0.0, 0.0])
        scatterers.append(RadarScatterer(
            tip, np.array([-1.0, 0.0, 0.0]), 0.35, f"antenna_tip_{index}",
        ))
    return TangoLikeModel(
        vertices_body_m=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=int),
        face_material_indices=np.asarray(material_indices, dtype=int),
        materials=materials, radar_scatterers=tuple(scatterers),
    )


def simulate_tango_like_physical_epoch(
    *, timestamp: float, observer_id: str, target_id: str,
    observer_state, target_state, sensor_quaternion_i2b_wxyz,
    target_quaternion_i2b_wxyz, target_angular_velocity_body_radps=None,
    config: PhysicalFrontendConfig | None = None,
    model: TangoLikeModel | None = None, random_seed: int = 0,
) -> PhysicalMultimodalEpoch:
    """Generate paired raw frames and filter-compatible messages."""
    selected = config or PhysicalFrontendConfig()
    spacecraft = model or build_tango_like_model()
    observer = np.asarray(observer_state, dtype=float).reshape(6)
    target = np.asarray(target_state, dtype=float).reshape(6)
    sensor_quaternion = np.asarray(sensor_quaternion_i2b_wxyz, dtype=float).reshape(4)
    target_quaternion = np.asarray(target_quaternion_i2b_wxyz, dtype=float).reshape(4)
    angular_velocity = np.zeros(3) if target_angular_velocity_body_radps is None else (
        np.asarray(target_angular_velocity_body_radps, dtype=float).reshape(3)
    )
    generators = {
        name: np.random.default_rng(int(random_seed) + offset)
        for name, offset in (("OPTICAL", 101), ("INFRARED", 202), ("RADAR", 303))
    }
    optical, optical_source = render_tango_like_optical_frame(
        timestamp=timestamp, observer_id=observer_id, target_id=target_id,
        observer_state=observer, target_state=target,
        sensor_quaternion_i2b_wxyz=sensor_quaternion,
        target_quaternion_i2b_wxyz=target_quaternion,
        model=spacecraft, config=selected, rng=generators["OPTICAL"],
        return_source=True,
    )
    infrared, infrared_source = render_tango_like_infrared_frame(
        timestamp=timestamp, observer_id=observer_id, target_id=target_id,
        observer_state=observer, target_state=target,
        sensor_quaternion_i2b_wxyz=sensor_quaternion,
        target_quaternion_i2b_wxyz=target_quaternion,
        model=spacecraft, config=selected, rng=generators["INFRARED"],
        return_source=True,
    )
    radar, radar_source = render_tango_like_radar_frame(
        timestamp=timestamp, observer_id=observer_id, target_id=target_id,
        observer_state=observer, target_state=target,
        target_quaternion_i2b_wxyz=target_quaternion,
        target_angular_velocity_body_radps=angular_velocity,
        acquisition_range_m=measure_relative_range(observer, target),
        acquisition_range_rate_mps=measure_relative_range_rate(observer, target),
        model=spacecraft, config=selected, rng=generators["RADAR"],
        return_source=True,
    )
    range_m = float(np.linalg.norm(target[:3] - observer[:3]))
    within_observation_range = bool(
        range_m <= selected.maximum_observation_range_m + 1.0e-6
    )
    messages = (
        _optical_message(optical, selected.optical, spacecraft),
        _infrared_message(infrared, selected.infrared, spacecraft),
        _radar_message(radar, selected.radar, spacecraft),
    )
    messages = tuple(
        replace(
            message,
            valid_flag=bool(message.valid_flag and within_observation_range),
            metadata={
                **message.metadata,
                "maximum_observation_range_m": float(
                    selected.maximum_observation_range_m
                ),
                "within_observation_range": within_observation_range,
            },
        )
        for message in messages
    )
    raw_frames = {
        "OPTICAL": _annotate_raw(
            raw_sensor_frame_from_optical(optical, config=selected.optical), spacecraft,
        ),
        "INFRARED": _annotate_raw(
            raw_sensor_frame_from_infrared(infrared, config=selected.infrared), spacecraft,
        ),
        "RADAR": _annotate_raw(
            raw_sensor_frame_from_radar(radar, config=selected.radar), spacecraft,
        ),
    }
    relative_position = target[:3] - observer[:3]
    source_products = PhysicalSourceProducts(
        optical_relative_reflected_radiance=optical_source[
            "relative_reflected_radiance"
        ],
        infrared_temperature_k=infrared_source["temperature_k"],
        infrared_spectral_radiance_w_m3_sr=infrared_source[
            "spectral_radiance_w_m3_sr"
        ],
        radar_scatterer_table=radar_source["scatterer_table"],
        radar_scatterer_labels=radar_source["scatterer_labels"],
        radar_iq=radar_source["iq"],
        geometry={
            "timestamp_s": float(timestamp),
            "observer_id": str(observer_id),
            "target_id": str(target_id),
            "relative_position_eci_m": relative_position.copy(),
            "relative_velocity_eci_mps": (target[3:] - observer[3:]).copy(),
            "range_m": range_m,
            "range_rate_mps": float(measure_relative_range_rate(observer, target)),
            "sun_direction_eci": _unit(selected.sun_direction_eci),
            "target_quaternion_i2b_wxyz": target_quaternion.copy(),
            "optical_projected_extent_pixels": float(
                optical.error_diagnostics["projected_extent_pixels"]
            ),
            "optical_resolved_blend_fraction": float(
                optical.error_diagnostics["resolved_blend_fraction"]
            ),
            "optical_resolution_regime": optical.error_diagnostics[
                "resolution_regime"
            ],
            "infrared_projected_extent_pixels": float(
                infrared.error_diagnostics["projected_extent_pixels"]
            ),
            "infrared_resolved_blend_fraction": float(
                infrared.error_diagnostics["resolved_blend_fraction"]
            ),
            "infrared_resolution_regime": infrared.error_diagnostics[
                "resolution_regime"
            ],
            "maximum_observation_range_m": float(
                selected.maximum_observation_range_m
            ),
            "within_observation_range": within_observation_range,
            "presentation_display_extent_pixels": float(
                optical_source["presentation_display_extent_pixels"]
            ),
            "presentation_scale_factor": float(
                optical_source["presentation_scale_factor"]
            ),
        },
    )
    return PhysicalMultimodalEpoch(
        messages=messages, raw_frames=raw_frames,
        concrete_frames={"OPTICAL": optical, "INFRARED": infrared, "RADAR": radar},
        source_products=source_products,
        presentation_frames={
            "OPTICAL": optical_source["presentation_enhanced"],
            "INFRARED": infrared_source["presentation_enhanced"],
            "RADAR": radar_source["presentation_enhanced"],
        },
        model_id=spacecraft.model_id,
    )


def render_tango_like_optical_frame(
    *, timestamp, observer_id, target_id, observer_state, target_state,
    sensor_quaternion_i2b_wxyz, target_quaternion_i2b_wxyz,
    model, config, rng, return_source=False,
) -> OpticalImageFrame | tuple[OpticalImageFrame, dict[str, np.ndarray]]:
    camera = config.optical
    geometry = _scene_geometry(
        observer_state, target_state, sensor_quaternion_i2b_wxyz,
        target_quaternion_i2b_wxyz, model,
    )
    sun = _unit(config.sun_direction_eci)
    view = _unit(np.asarray(observer_state)[:3] - np.asarray(target_state)[:3])
    relative_values = []
    for normal, material_index in zip(
        geometry["normals_eci"], model.face_material_indices,
    ):
        material = model.materials[int(material_index)]
        incident = max(float(normal @ sun), 0.0)
        half_vector = _unit(sun + view, fallback=view)
        specular = max(float(normal @ half_vector), 0.0) ** 24
        fraction = (
            config.optical_ambient_fraction
            + material.optical_albedo * incident
            + material.optical_specular * specular
        )
        relative_values.append(np.clip(fraction, 0.0, 1.25))
    detector_values = camera.source_peak * np.asarray(relative_values, dtype=float)
    resolved_signal, visible = _rasterize_mesh(
        geometry["vertices_sensor"], model.faces, detector_values,
        width=camera.width, height=camera.height,
        focal_xy=(camera.focal_length_x_pixels, camera.focal_length_y_pixels),
        principal_xy=(camera.principal_x, camera.principal_y),
        background=0.0,
    )
    normalized_uv, pixel_xy, center_visible = _center_projection(
        observer_state, target_state, sensor_quaternion_i2b_wxyz,
        camera.focal_length_x_pixels, camera.focal_length_y_pixels,
        camera.principal_x, camera.principal_y, camera.width, camera.height,
    )
    projected_extent = _projected_mesh_extent_pixels(
        geometry["vertices_sensor"], camera,
    )
    resolved_fraction = _resolution_blend_fraction(projected_extent, config)
    distance_m = float(np.linalg.norm(
        np.asarray(target_state, dtype=float)[:3]
        - np.asarray(observer_state, dtype=float)[:3]
    ))
    radiometric_scale = min(
        1.0, (config.radiometric_reference_distance_m / distance_m) ** 2,
    )
    point_peak = camera.source_peak * float(np.mean(relative_values)) * radiometric_scale
    point_signal = _render_point_signal(
        pixel_xy, shape=(camera.height, camera.width),
        peak=point_peak,
        sigma=_effective_transition_sigma(
            projected_extent, camera.psf_sigma_pixels,
        ),
    )
    resolved_signal = gaussian_filter(
        resolved_signal, sigma=camera.psf_sigma_pixels, mode="nearest",
    )
    image = (
        resolved_fraction * resolved_signal
        + (1.0 - resolved_fraction) * point_signal
        + camera.background
    )
    if camera.read_noise_sigma > 0.0:
        image += rng.normal(0.0, camera.read_noise_sigma, image.shape)
    regime = _resolution_regime(resolved_fraction)
    frame = OpticalImageFrame(
        timestamp=float(timestamp), observer_id=str(observer_id), target_id=str(target_id),
        image=image, quaternion_i2b_wxyz=np.asarray(sensor_quaternion_i2b_wxyz).copy(),
        ideal_normalized_uv=normalized_uv, ideal_pixel_xy=pixel_xy,
        target_in_frame=bool(center_visible and (np.any(visible) or point_peak > 0.0)),
        error_diagnostics={
            "rendered_face_pixels": float(np.count_nonzero(visible)),
            "projected_extent_pixels": float(projected_extent),
            "resolved_blend_fraction": float(resolved_fraction),
            "resolution_regime": regime,
            "radiometric_scale": float(radiometric_scale),
            "unresolved_point_peak": float(point_peak),
        },
    )
    if return_source:
        display_geometry = _magnified_source_vertices(
            geometry["vertices_sensor"], geometry["center_sensor"], camera,
            config.source_display_extent_pixels,
        )
        relative_image, _ = _rasterize_mesh(
            display_geometry, model.faces,
            np.asarray(relative_values, dtype=float),
            width=camera.width, height=camera.height,
            focal_xy=(camera.focal_length_x_pixels, camera.focal_length_y_pixels),
            principal_xy=(camera.principal_x, camera.principal_y), background=0.0,
        )
        presentation_extent = _presentation_display_extent(
            distance_m, projected_extent, config,
        )
        presentation_vertices = _magnified_source_vertices(
            geometry["vertices_sensor"], geometry["center_sensor"], camera,
            presentation_extent,
        )
        presentation_signal, presentation_visible = _rasterize_mesh(
            presentation_vertices, model.faces, detector_values * radiometric_scale,
            width=camera.width, height=camera.height,
            focal_xy=(camera.focal_length_x_pixels, camera.focal_length_y_pixels),
            principal_xy=(camera.principal_x, camera.principal_y), background=0.0,
        )
        if presentation_extent > 0.0 and not np.any(presentation_visible):
            presentation_signal = _render_point_signal(
                pixel_xy, shape=(camera.height, camera.width),
                peak=point_peak,
                sigma=_effective_transition_sigma(
                    presentation_extent, camera.psf_sigma_pixels,
                ),
            )
        presentation_image = gaussian_filter(
            presentation_signal, sigma=camera.psf_sigma_pixels, mode="nearest",
        ) + camera.background
        return frame, {
            "relative_reflected_radiance": relative_image,
            "face_relative_reflected_radiance": np.asarray(
                relative_values, dtype=float,
            ),
            "presentation_enhanced": presentation_image,
            "presentation_display_extent_pixels": presentation_extent,
            "presentation_scale_factor": (
                presentation_extent / max(projected_extent, 1.0e-15)
            ),
        }
    return frame


def render_tango_like_infrared_frame(
    *, timestamp, observer_id, target_id, observer_state, target_state,
    sensor_quaternion_i2b_wxyz, target_quaternion_i2b_wxyz,
    model, config, rng, return_source=False,
) -> InfraredImageFrame | tuple[InfraredImageFrame, dict[str, np.ndarray]]:
    camera = config.infrared
    geometry = _scene_geometry(
        observer_state, target_state, sensor_quaternion_i2b_wxyz,
        target_quaternion_i2b_wxyz, model,
    )
    sun = _unit(config.sun_direction_eci)
    reference = _planck_radiance(
        config.thermal_reference_temperature_k, config.thermal_wavelength_um,
    )
    detector_values = []
    spectral_radiances = []
    temperatures = []
    for normal, material_index in zip(
        geometry["normals_eci"], model.face_material_indices,
    ):
        material = model.materials[int(material_index)]
        incident = max(float(normal @ sun), 0.0)
        temperature = material.base_temperature_k + material.sun_temperature_delta_k * incident
        radiance = material.infrared_emissivity * _planck_radiance(
            temperature, config.thermal_wavelength_um,
        )
        detector_values.append(camera.source_peak * radiance / reference)
        spectral_radiances.append(radiance)
        temperatures.append(temperature)
    resolved_signal, visible = _rasterize_mesh(
        geometry["vertices_sensor"], model.faces, np.asarray(detector_values),
        width=camera.width, height=camera.height,
        focal_xy=(camera.focal_length_x_pixels, camera.focal_length_y_pixels),
        principal_xy=(camera.principal_x, camera.principal_y),
        background=0.0,
    )
    normalized, pixel_xy, center_visible = _center_projection(
        observer_state, target_state, sensor_quaternion_i2b_wxyz,
        camera.focal_length_x_pixels, camera.focal_length_y_pixels,
        camera.principal_x, camera.principal_y, camera.width, camera.height,
    )
    projected_extent = _projected_mesh_extent_pixels(
        geometry["vertices_sensor"], camera,
    )
    resolved_fraction = _resolution_blend_fraction(projected_extent, config)
    distance_m = float(np.linalg.norm(
        np.asarray(target_state, dtype=float)[:3]
        - np.asarray(observer_state, dtype=float)[:3]
    ))
    radiometric_scale = min(
        1.0, (config.radiometric_reference_distance_m / distance_m) ** 2,
    )
    point_peak = camera.source_peak * float(np.mean(detector_values)) / max(
        camera.source_peak, np.finfo(float).eps,
    ) * radiometric_scale
    point_signal = _render_point_signal(
        pixel_xy, shape=(camera.height, camera.width),
        peak=point_peak,
        sigma=_effective_transition_sigma(
            projected_extent, camera.psf_sigma_pixels,
        ),
    )
    resolved_signal = gaussian_filter(
        resolved_signal, sigma=camera.psf_sigma_pixels, mode="nearest",
    )
    image = (
        resolved_fraction * resolved_signal
        + (1.0 - resolved_fraction) * point_signal
        + camera.background
    )
    display_geometry = _magnified_source_vertices(
        geometry["vertices_sensor"], geometry["center_sensor"], camera,
        config.source_display_extent_pixels,
    )
    temperature_image, _ = _rasterize_mesh(
        display_geometry, model.faces, np.asarray(temperatures),
        width=camera.width, height=camera.height,
        focal_xy=(camera.focal_length_x_pixels, camera.focal_length_y_pixels),
        principal_xy=(camera.principal_x, camera.principal_y),
        background=np.nan,
    )
    spectral_radiance_image, _ = _rasterize_mesh(
        display_geometry, model.faces, np.asarray(spectral_radiances),
        width=camera.width, height=camera.height,
        focal_xy=(camera.focal_length_x_pixels, camera.focal_length_y_pixels),
        principal_xy=(camera.principal_x, camera.principal_y), background=np.nan,
    )
    if camera.read_noise_sigma > 0.0:
        image += rng.normal(0.0, camera.read_noise_sigma, image.shape)
    ideal_az_el = (
        infrared_pixel_to_az_el(pixel_xy, camera)
        if np.all(np.isfinite(pixel_xy)) else np.array([np.nan, np.nan])
    )
    frame = InfraredImageFrame(
        timestamp=float(timestamp), observer_id=str(observer_id), target_id=str(target_id),
        image=image, quaternion_i2b_wxyz=np.asarray(sensor_quaternion_i2b_wxyz).copy(),
        ideal_az_el=ideal_az_el, ideal_pixel_xy=pixel_xy,
        target_in_frame=bool(center_visible and (np.any(visible) or point_peak > 0.0)),
        error_diagnostics={
            "minimum_surface_temperature_k": float(min(temperatures)),
            "maximum_surface_temperature_k": float(max(temperatures)),
            "projected_extent_pixels": float(projected_extent),
            "resolved_blend_fraction": float(resolved_fraction),
            "resolution_regime": _resolution_regime(resolved_fraction),
            "radiometric_scale": float(radiometric_scale),
            "unresolved_point_peak": float(point_peak),
        },
    )
    if return_source:
        presentation_extent = _presentation_display_extent(
            distance_m, projected_extent, config,
        )
        presentation_vertices = _magnified_source_vertices(
            geometry["vertices_sensor"], geometry["center_sensor"], camera,
            presentation_extent,
        )
        presentation_signal, presentation_visible = _rasterize_mesh(
            presentation_vertices, model.faces,
            np.asarray(detector_values) * radiometric_scale,
            width=camera.width, height=camera.height,
            focal_xy=(camera.focal_length_x_pixels, camera.focal_length_y_pixels),
            principal_xy=(camera.principal_x, camera.principal_y), background=0.0,
        )
        if presentation_extent > 0.0 and not np.any(presentation_visible):
            presentation_signal = _render_point_signal(
                pixel_xy, shape=(camera.height, camera.width),
                peak=point_peak,
                sigma=_effective_transition_sigma(
                    presentation_extent, camera.psf_sigma_pixels,
                ),
            )
        presentation_image = gaussian_filter(
            presentation_signal, sigma=camera.psf_sigma_pixels, mode="nearest",
        ) + camera.background
        return frame, {
            "temperature_k": temperature_image,
            "spectral_radiance_w_m3_sr": spectral_radiance_image,
            "face_temperature_k": np.asarray(temperatures, dtype=float),
            "face_spectral_radiance_w_m3_sr": np.asarray(
                spectral_radiances, dtype=float,
            ),
            "presentation_enhanced": presentation_image,
            "presentation_display_extent_pixels": presentation_extent,
            "presentation_scale_factor": (
                presentation_extent / max(projected_extent, 1.0e-15)
            ),
        }
    return frame


def render_tango_like_radar_frame(
    *, timestamp, observer_id, target_id, observer_state, target_state,
    target_quaternion_i2b_wxyz, target_angular_velocity_body_radps,
    acquisition_range_m, acquisition_range_rate_mps, model, config, rng,
    return_source=False,
) -> RadarRangeDopplerFrame | tuple[RadarRangeDopplerFrame, dict[str, object]]:
    radar = config.radar
    observer = np.asarray(observer_state, dtype=float).reshape(6)
    target = np.asarray(target_state, dtype=float).reshape(6)
    dcm_b2i = quat_to_dcm_i2b(target_quaternion_i2b_wxyz).T
    omega_eci = dcm_b2i @ np.asarray(target_angular_velocity_body_radps, dtype=float)
    center = np.array([acquisition_range_m, acquisition_range_rate_mps], dtype=float)
    any_in_window = False
    scatterer_rows = []
    iq = np.zeros((radar.height, radar.width), dtype=np.complex128)
    slow_index = np.arange(radar.height, dtype=float)[:, None]
    fast_index = np.arange(radar.width, dtype=float)[None, :]
    for scatterer_index, scatterer in enumerate(model.radar_scatterers):
        offset_eci = dcm_b2i @ np.asarray(scatterer.position_body_m, dtype=float)
        position = target[:3] + offset_eci
        velocity = target[3:] + np.cross(omega_eci, offset_eci)
        relative = position - observer[:3]
        scatter_range = float(np.linalg.norm(relative))
        los = relative / max(scatter_range, np.finfo(float).eps)
        scatter_rate = float(los @ (velocity - observer[3:]))
        normal_eci = dcm_b2i @ np.asarray(scatterer.normal_body, dtype=float)
        aspect_gain = 0.2 + 0.8 * abs(float(normal_eci @ (-los)))
        column = radar.principal_x + (
            scatter_range - center[0]
        ) / radar.range_bin_size_m
        row = radar.principal_y + (
            scatter_rate - center[1]
        ) / radar.range_rate_bin_size_mps
        amplitude = radar.source_peak * scatterer.strength * aspect_gain
        scatterer_rows.append((
            scatter_range, scatter_rate, amplitude, row, column,
            float(position[0]), float(position[1]), float(position[2]),
        ))
        if 0.0 <= column < radar.width and 0.0 <= row < radar.height:
            any_in_window = True
        range_frequency = (column - radar.principal_x) / radar.width
        doppler_frequency = (row - radar.principal_y) / radar.height
        phase = 2.0 * np.pi * ((scatterer_index * 0.61803398875) % 1.0)
        iq += np.sqrt(max(amplitude, 0.0)) * np.exp(
            -2.0j * np.pi * (
                doppler_frequency * slow_index + range_frequency * fast_index
            ) + 1.0j * phase
        )
    spectrum = np.fft.fftshift(np.fft.fft2(iq))
    signal_power = np.abs(spectrum) ** 2
    signal_power = gaussian_filter(
        signal_power, sigma=radar.psf_sigma_pixels, mode="nearest",
    )
    radar_radiometric_scale = min(
        1.0,
        (
            config.radiometric_reference_distance_m
            / max(float(measure_relative_range(observer, target)), 1.0)
        ) ** 4,
    )
    if np.max(signal_power) > 0.0:
        signal_power *= (
            radar.source_peak * radar_radiometric_scale / np.max(signal_power)
        )
    power = signal_power + radar.background
    if radar.read_noise_sigma > 0.0:
        power += rng.normal(0.0, radar.read_noise_sigma, power.shape)
    ideal = np.array([
        measure_relative_range(observer, target),
        measure_relative_range_rate(observer, target),
    ])
    ideal_bin = np.array([
        radar.principal_x + (ideal[0] - center[0]) / radar.range_bin_size_m,
        radar.principal_y + (ideal[1] - center[1]) / radar.range_rate_bin_size_mps,
    ])
    frame = RadarRangeDopplerFrame(
        timestamp=float(timestamp), observer_id=str(observer_id), target_id=str(target_id),
        power=power, acquisition_center=center,
        ideal_range_range_rate=ideal, ideal_bin_xy=ideal_bin,
        target_in_window=bool(any_in_window),
        error_diagnostics={
            "scatterer_count": float(len(model.radar_scatterers)),
            "source_product": "COMPLEX_BASEBAND_IQ_TO_RANGE_DOPPLER_FFT",
            "radiometric_scale": float(radar_radiometric_scale),
        },
    )
    if return_source:
        radar_enhanced = np.log1p(np.maximum(power - radar.background, 0.0))
        if ideal[0] > config.maximum_observation_range_m + 1.0e-6:
            radar_enhanced = np.zeros_like(radar_enhanced)
        if np.max(radar_enhanced) > 0.0:
            radar_enhanced /= np.max(radar_enhanced)
        return frame, {
            "scatterer_table": np.asarray(scatterer_rows, dtype=float),
            "scatterer_labels": tuple(item.label for item in model.radar_scatterers),
            "iq": iq,
            "presentation_enhanced": radar_enhanced,
        }
    return frame


def _append_box(vertices, faces, material_indices, *, center, size, material):
    center = np.asarray(center, dtype=float)
    half = 0.5 * np.asarray(size, dtype=float)
    local = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ], dtype=float) * half + center
    offset = len(vertices)
    vertices.extend(local)
    quads = (
        (0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4),
        (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5),
    )
    for a, b, c, d in quads:
        faces.extend(((offset + a, offset + b, offset + c),
                      (offset + a, offset + c, offset + d)))
        material_indices.extend((material, material))


def _scene_geometry(observer_state, target_state, sensor_quaternion, target_quaternion, model):
    observer = np.asarray(observer_state, dtype=float).reshape(6)
    target = np.asarray(target_state, dtype=float).reshape(6)
    dcm_i2sensor = quat_to_dcm_i2b(sensor_quaternion)
    dcm_b2i = quat_to_dcm_i2b(target_quaternion).T
    vertices_eci = target[:3] + (dcm_b2i @ model.vertices_body_m.T).T
    vertices_sensor = (dcm_i2sensor @ (vertices_eci - observer[:3]).T).T
    normals_body = []
    for face in model.faces:
        triangle = model.vertices_body_m[face]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normals_body.append(_unit(normal))
    normals_eci = (dcm_b2i @ np.asarray(normals_body).T).T
    center_sensor = dcm_i2sensor @ (target[:3] - observer[:3])
    return {
        "vertices_sensor": vertices_sensor,
        "normals_eci": normals_eci,
        "center_sensor": center_sensor,
    }


def _projected_mesh_extent_pixels(vertices_sensor, camera):
    vertices = np.asarray(vertices_sensor, dtype=float)
    valid = vertices[:, 0] > 1.0e-9
    if not np.any(valid):
        return 0.0
    projected = np.column_stack((
        camera.focal_length_x_pixels * vertices[valid, 1] / vertices[valid, 0],
        camera.focal_length_y_pixels * vertices[valid, 2] / vertices[valid, 0],
    ))
    extent = np.ptp(projected, axis=0)
    return float(np.max(extent))


def _resolution_blend_fraction(projected_extent, config):
    fraction = (
        float(projected_extent) - config.unresolved_extent_pixels
    ) / (config.resolved_extent_pixels - config.unresolved_extent_pixels)
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return fraction * fraction * (3.0 - 2.0 * fraction)


def _resolution_regime(resolved_fraction):
    if resolved_fraction <= 1.0e-6:
        return "UNRESOLVED_POINT_SOURCE"
    if resolved_fraction >= 1.0 - 1.0e-6:
        return "RESOLVED_EXTENDED_TARGET"
    return "PARTIALLY_RESOLVED_TRANSITION"


def _render_point_signal(pixel_xy, *, shape, peak, sigma):
    pixel = np.asarray(pixel_xy, dtype=float).reshape(2)
    if not np.all(np.isfinite(pixel)) or peak <= 0.0:
        return np.zeros(shape, dtype=float)
    rows, columns = np.indices(shape, dtype=float)
    squared = (columns - pixel[0]) ** 2 + (rows - pixel[1]) ** 2
    return float(peak) * np.exp(-0.5 * squared / float(sigma) ** 2)


def _effective_transition_sigma(projected_extent, detector_psf_sigma):
    """Approximate convolution of a finite target footprint and detector PSF."""
    footprint_sigma = max(float(projected_extent), 0.0) / 2.355
    return float(np.hypot(float(detector_psf_sigma), footprint_sigma))


def _presentation_display_extent(distance_m, physical_extent_pixels, config):
    if distance_m > config.maximum_observation_range_m + 1.0e-6:
        return 0.0
    normalized_range = float(np.clip(
        distance_m / config.maximum_observation_range_m, 0.0, 1.0,
    ))
    logical_extent = (
        config.display_minimum_extent_pixels
        + (
            config.display_maximum_extent_pixels
            - config.display_minimum_extent_pixels
        ) * (1.0 - normalized_range) ** config.display_range_exponent
    )
    return float(max(float(physical_extent_pixels), logical_extent))


def _magnified_source_vertices(vertices_sensor, center_sensor, camera, target_extent):
    vertices = np.asarray(vertices_sensor, dtype=float)
    center = np.asarray(center_sensor, dtype=float).reshape(3)
    extent = _projected_mesh_extent_pixels(vertices, camera)
    if extent <= 1.0e-12:
        return vertices.copy()
    scale = float(target_extent) / extent
    return center + scale * (vertices - center)


def _rasterize_mesh(
    vertices_sensor, faces, face_values, *, width, height,
    focal_xy, principal_xy, background,
):
    image = np.full((height, width), float(background), dtype=float)
    depth_buffer = np.full((height, width), np.inf, dtype=float)
    painted = np.zeros((height, width), dtype=bool)
    fx, fy = map(float, focal_xy)
    cx, cy = map(float, principal_xy)
    for face, value in zip(faces, face_values):
        triangle = np.asarray(vertices_sensor[face], dtype=float)
        if np.any(triangle[:, 0] <= 1.0e-6):
            continue
        projected = np.column_stack((
            cx + fx * triangle[:, 1] / triangle[:, 0],
            cy + fy * triangle[:, 2] / triangle[:, 0],
        ))
        minimum = np.floor(np.min(projected, axis=0)).astype(int)
        maximum = np.ceil(np.max(projected, axis=0)).astype(int)
        x0, y0 = max(0, minimum[0]), max(0, minimum[1])
        x1, y1 = min(width - 1, maximum[0]), min(height - 1, maximum[1])
        if x0 > x1 or y0 > y1:
            continue
        yy, xx = np.indices((y1 - y0 + 1, x1 - x0 + 1), dtype=float)
        xx += x0
        yy += y0
        a, b, c = projected
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (
            c[0] - b[0]
        ) * (a[1] - c[1])
        if abs(denominator) < 1.0e-12:
            continue
        w0 = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / denominator
        w1 = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / denominator
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1.0e-8) & (w1 >= -1.0e-8) & (w2 >= -1.0e-8)
        depth = w0 * triangle[0, 0] + w1 * triangle[1, 0] + w2 * triangle[2, 0]
        local_depth = depth_buffer[y0:y1 + 1, x0:x1 + 1]
        update = inside & (depth < local_depth)
        if not np.any(update):
            continue
        local_image = image[y0:y1 + 1, x0:x1 + 1]
        local_painted = painted[y0:y1 + 1, x0:x1 + 1]
        local_image[update] = float(value)
        local_depth[update] = depth[update]
        local_painted[update] = True
    return image, painted


def _center_projection(
    observer_state, target_state, sensor_quaternion,
    focal_x, focal_y, principal_x, principal_y, width, height,
):
    relative = quat_to_dcm_i2b(sensor_quaternion) @ (
        np.asarray(target_state, dtype=float)[:3]
        - np.asarray(observer_state, dtype=float)[:3]
    )
    if relative[0] <= 0.0:
        nan = np.array([np.nan, np.nan])
        return nan, nan.copy(), False
    normalized = relative[1:] / relative[0]
    pixel = normalized * np.array([focal_x, focal_y]) + np.array([
        principal_x, principal_y,
    ])
    visible = bool(0 <= pixel[0] < width and 0 <= pixel[1] < height)
    return normalized, pixel, visible


def _extended_centroid(image, *, background, noise_sigma, detection_sigma):
    values = np.asarray(image, dtype=float)
    threshold = float(background + max(detection_sigma * noise_sigma, 1.0))
    signal = np.maximum(values - threshold, 0.0)
    total = float(np.sum(signal))
    if total <= 0.0:
        return np.array([np.nan, np.nan]), False
    rows, columns = np.indices(values.shape, dtype=float)
    return np.array([
        float(np.sum(columns * signal) / total),
        float(np.sum(rows * signal) / total),
    ]), True


def _optical_message(frame, config, model):
    pixel, detected = _extended_centroid(
        frame.image, background=config.background,
        noise_sigma=config.read_noise_sigma, detection_sigma=config.detection_sigma,
    )
    measurement = (
        (pixel - np.array([config.principal_x, config.principal_y]))
        / np.array([config.focal_length_x_pixels, config.focal_length_y_pixels])
        if detected else np.zeros(2)
    )
    sigma = config.reported_centroid_sigma_pixels
    covariance = np.diag([
        (sigma / config.focal_length_x_pixels) ** 2,
        (sigma / config.focal_length_y_pixels) ** 2,
    ])
    return _camera_message(
        frame, "OPTICAL", "NORMALIZED_IMAGE_COORDINATES", measurement,
        covariance, pixel, bool(frame.target_in_frame and detected), model,
    )


def _infrared_message(frame, config, model):
    pixel, detected = _extended_centroid(
        frame.image, background=config.background,
        noise_sigma=config.read_noise_sigma, detection_sigma=config.detection_sigma,
    )
    measurement = infrared_pixel_to_az_el(pixel, config) if detected else np.zeros(2)
    epsilon = 1.0e-3
    jacobian = np.empty((2, 2))
    if detected:
        for axis in range(2):
            offset = np.zeros(2)
            offset[axis] = epsilon
            jacobian[:, axis] = (
                infrared_pixel_to_az_el(pixel + offset, config)
                - infrared_pixel_to_az_el(pixel - offset, config)
            ) / (2.0 * epsilon)
        covariance = jacobian @ (
            np.eye(2) * config.reported_centroid_sigma_pixels**2
        ) @ jacobian.T
    else:
        covariance = np.eye(2)
    return _camera_message(
        frame, "INFRARED", "AZIMUTH_ELEVATION", measurement,
        covariance, pixel, bool(frame.target_in_frame and detected), model,
    )


def _camera_message(frame, modality, measurement_type, measurement, covariance,
                    pixel, valid, model):
    information_id = (
        f"{frame.observer_id}->{frame.target_id}:"
        f"tango_like_{modality.lower()}:{frame.timestamp:g}"
    )
    return ObservationMessage(
        message_id=information_id, physical_observation_id=information_id,
        observer_id=frame.observer_id, target_id=frame.target_id,
        timestamp=frame.timestamp, modality=modality, frame="BODY",
        measurement=np.asarray(measurement), covariance=np.asarray(covariance),
        valid_flag=valid, source_timestamp=frame.timestamp,
        arrival_timestamp=frame.timestamp,
        metadata={
            "sensor_modality": modality,
            "measurement_type": measurement_type,
            "measurement_component": modality,
            "raw_source_type": "TANGO_LIKE_SIMPLIFIED_PHYSICAL_RENDER",
            "centroid_pixel_xy": np.asarray(pixel),
            "physical_frontend": True,
            "physical_fidelity": "SIMPLIFIED_FUNCTIONAL_DEMONSTRATION",
            "spacecraft_model_id": model.model_id,
            "spacecraft_model_provenance": model.provenance,
        },
    )


def _radar_message(frame, config, model):
    message = radar_frame_to_observation_message(frame, config=config)
    message.metadata.update({
        "raw_source_type": "TANGO_LIKE_SCATTERING_CENTER_RANGE_DOPPLER",
        "physical_frontend": True,
        "physical_fidelity": "SIMPLIFIED_FUNCTIONAL_DEMONSTRATION",
        "spacecraft_model_id": model.model_id,
        "spacecraft_model_provenance": model.provenance,
        "radar_scatterer_count": len(model.radar_scatterers),
    })
    return message


def _annotate_raw(frame, model):
    return replace(frame, metadata={
        **frame.metadata,
        "physical_frontend": True,
        "physical_fidelity": "SIMPLIFIED_FUNCTIONAL_DEMONSTRATION",
        "spacecraft_model_id": model.model_id,
        "spacecraft_model_provenance": model.provenance,
    })


def _planck_radiance(temperature_k, wavelength_um):
    wavelength_m = float(wavelength_um) * 1.0e-6
    h = 6.62607015e-34
    c = 299792458.0
    k = 1.380649e-23
    exponent = h * c / (wavelength_m * k * float(temperature_k))
    return 2.0 * h * c**2 / wavelength_m**5 / np.expm1(exponent)


def _unit(vector, *, fallback=None):
    values = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(values))
    if norm > 1.0e-15:
        return values / norm
    if fallback is None:
        raise ValueError("Cannot normalize a zero vector.")
    return _unit(fallback)
