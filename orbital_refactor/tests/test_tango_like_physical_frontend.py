import numpy as np
import json

from adapters.tango_like_physical_frontend import (
    build_tango_like_model,
    simulate_tango_like_physical_epoch,
)
from orbital_core.attitude import small_angle_quaternion_wxyz
from experiments.run_tango_like_physical_frontend_demo import run_demo


def _epoch(*, angle=0.0, seed=4):
    return simulate_tango_like_physical_epoch(
        timestamp=2.0, observer_id="observer", target_id="target",
        observer_state=np.zeros(6),
        target_state=np.array([8.0, 0.05, -0.03, -0.02, 0.01, 0.0]),
        sensor_quaternion_i2b_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        target_quaternion_i2b_wxyz=small_angle_quaternion_wxyz(
            np.array([0.2 * angle, 0.5 * angle, angle]),
        ),
        target_angular_velocity_body_radps=np.array([0.01, 0.02, 0.03]),
        random_seed=seed,
    )


def _epoch_at_distance(distance_m, *, seed=7):
    return simulate_tango_like_physical_epoch(
        timestamp=2.0, observer_id="observer", target_id="target",
        observer_state=np.zeros(6),
        target_state=np.array([distance_m, 0.05, -0.03, -0.02, 0.01, 0.0]),
        sensor_quaternion_i2b_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        target_quaternion_i2b_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        random_seed=seed,
    )


def test_tango_like_model_uses_public_simplified_scale_and_multiple_materials():
    model = build_tango_like_model()
    extent = np.ptp(model.vertices_body_m, axis=0)

    assert model.model_id.startswith("tango-like")
    assert extent[0] >= 0.50  # body plus 0.204 m rear appendages
    assert extent[1] >= 0.75  # published body-mounted panel width
    assert extent[2] >= 0.56  # published body-mounted panel height
    assert {item.name for item in model.materials} == {
        "main_body", "solar_panel", "metal_appendage",
    }
    assert len(model.radar_scatterers) >= 12


def test_physical_epoch_is_deterministic_and_filter_compatible():
    first = _epoch()
    second = _epoch()

    assert {item.modality for item in first.messages} == {
        "OPTICAL", "INFRARED", "RADAR",
    }
    assert set(first.raw_frames) == {"OPTICAL", "INFRARED", "RADAR"}
    assert set(first.presentation_frames) == {"OPTICAL", "INFRARED", "RADAR"}
    assert all(item.valid_flag for item in first.messages)
    assert all(item.metadata["physical_frontend"] for item in first.messages)
    for modality in first.raw_frames:
        assert np.array_equal(
            first.raw_frames[modality].data,
            second.raw_frames[modality].data,
        )


def test_modalities_are_visually_distinct_and_attitude_dependent():
    reference = _epoch(angle=0.0)
    rotated = _epoch(angle=0.9)
    optical = reference.raw_frames["OPTICAL"].data
    infrared = reference.raw_frames["INFRARED"].data
    radar = reference.raw_frames["RADAR"].data

    assert optical.shape == infrared.shape == (256, 256)
    assert radar.shape == (129, 129)
    assert not np.allclose(optical, infrared)
    assert not np.allclose(optical, rotated.raw_frames["OPTICAL"].data)
    assert not np.allclose(infrared, rotated.raw_frames["INFRARED"].data)
    assert not np.allclose(radar, rotated.raw_frames["RADAR"].data)


def test_radar_map_contains_distributed_scattering_energy():
    epoch = _epoch()
    radar = epoch.raw_frames["RADAR"].data
    threshold = np.median(radar) + 8.0 * np.std(radar[:12, :12])

    assert np.count_nonzero(radar > threshold) > 8
    radar_message = next(item for item in epoch.messages if item.modality == "RADAR")
    truth = epoch.concrete_frames["RADAR"].ideal_range_range_rate
    assert abs(radar_message.measurement[0] - truth[0]) < 0.08
    assert abs(radar_message.measurement[1] - truth[1]) < 0.06


def test_physical_source_products_are_separate_and_traceable():
    epoch = _epoch()
    products = epoch.source_products

    assert products.optical_relative_reflected_radiance.shape == (256, 256)
    assert products.infrared_temperature_k.shape == (256, 256)
    assert products.infrared_spectral_radiance_w_m3_sr.shape == (256, 256)
    assert np.iscomplexobj(products.radar_iq)
    assert products.radar_iq.shape == (129, 129)
    assert products.radar_scatterer_table.shape[1] == 8
    assert products.radar_scatterer_table.shape[0] == len(
        products.radar_scatterer_labels
    )
    assert products.geometry["range_m"] > 0.0
    assert not np.array_equal(
        products.optical_relative_reflected_radiance,
        epoch.raw_frames["OPTICAL"].data,
    )


def test_detector_rendering_transitions_continuously_to_a_far_point_source():
    near = _epoch_at_distance(8.0)
    transition = _epoch_at_distance(120.0)
    far = _epoch_at_distance(6000000.0)
    geometries = [item.source_products.geometry for item in (near, transition, far)]

    assert geometries[0]["optical_projected_extent_pixels"] > 16.0
    assert 0.25 < geometries[1]["optical_projected_extent_pixels"] < 16.0
    assert geometries[2]["optical_projected_extent_pixels"] < 0.25
    assert [item["optical_resolution_regime"] for item in geometries] == [
        "RESOLVED_EXTENDED_TARGET",
        "PARTIALLY_RESOLVED_TRANSITION",
        "UNRESOLVED_POINT_SOURCE",
    ]
    assert all(message.valid_flag for message in far.messages)
    assert np.nanmax(far.source_products.infrared_temperature_k) > 250.0


def test_display_enhancement_follows_visibility_range_without_entering_raw_data():
    boundary = _epoch_at_distance(6000000.0)
    outside = _epoch_at_distance(6100000.0)
    geometry = boundary.source_products.geometry

    assert geometry["within_observation_range"] is True
    assert geometry["presentation_display_extent_pixels"] >= 2.0
    assert geometry["presentation_scale_factor"] > 1000.0
    assert not np.array_equal(
        boundary.presentation_frames["OPTICAL"],
        boundary.raw_frames["OPTICAL"].data,
    )
    assert np.std(boundary.presentation_frames["OPTICAL"]) > 0.0
    assert all(not message.valid_flag for message in outside.messages)
    assert outside.source_products.geometry["within_observation_range"] is False
    assert np.std(outside.presentation_frames["OPTICAL"]) == 0.0


def test_distance_response_demo_writes_reviewable_three_modality_bundle(tmp_path):
    output = run_demo(
        output=tmp_path / "distance-response",
        frame_count=4,
        distance_m=10.0,
        maximum_distance_m=6_000_000.0,
        seed=0,
        show=False,
    )
    summary = json.loads(
        (output / "test_summary.json").read_text(encoding="utf-8")
    )

    assert summary["overallPassed"] is True
    assert summary["accuracyThresholdsApplied"] is False
    assert summary["checkCount"] == 3
    assert summary["sampleCount"] == 4
    assert summary["modalities"] == ["RADAR", "INFRARED", "OPTICAL"]
    assert (output / "distance_measurements.csv").is_file()
    assert (output / "distance_measurements.json").is_file()
    assert (output / "measurement_distance_trends.png").is_file()
    assert all(
        (output / f"epoch_{index:03d}" / "sensor_raw.npz").is_file()
        for index in range(4)
    )
