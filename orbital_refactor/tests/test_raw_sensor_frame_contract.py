import numpy as np
import pytest

from adapters.infrared_image_adapter import (
    InfraredCameraConfig, render_infrared_point_source_frame,
)
from adapters.optical_image_adapter import (
    OpticalCameraConfig, render_optical_point_source_frame,
)
from adapters.radar_range_doppler_adapter import (
    RadarRangeDopplerConfig, render_radar_range_doppler_frame,
)
from adapters.raw_sensor_frame_adapter import (
    infrared_frame_from_raw, optical_frame_from_raw, radar_frame_from_raw,
    raw_sensor_frame_from_infrared, raw_sensor_frame_from_optical,
    raw_sensor_frame_from_radar,
)
from interfaces.data_objects import RawSensorFrame
from interfaces.interface_contracts import InterfaceValidationError, validate_raw_sensor_frame


OBSERVER = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
TARGET = np.array([7.001e6, 100.0, 20.0, 0.2, 7500.1, 0.0])
QUATERNION = np.array([1.0, 0.0, 0.0, 0.0])


def test_radar_unified_frame_is_lossless_and_has_physical_axes():
    config = RadarRangeDopplerConfig(width=33, height=31)
    concrete = render_radar_range_doppler_frame(
        timestamp=1.0, observer_id="a", target_id="b",
        observer_state=OBSERVER, target_state=TARGET,
        acquisition_range_m=1005.0, acquisition_range_rate_mps=0.2,
        config=config, rng=np.random.default_rng(1),
    )
    raw = raw_sensor_frame_from_radar(concrete, config=config)
    restored = radar_frame_from_raw(raw)
    assert raw.data_kind == "RANGE_DOPPLER_POWER_MAP"
    assert raw.axes["range_m"].shape == (33,)
    assert raw.axes["range_rate_mps"].shape == (31,)
    np.testing.assert_array_equal(restored.power, concrete.power)
    assert not raw.data.flags.writeable


@pytest.mark.parametrize("modality", ["INFRARED", "OPTICAL"])
def test_camera_unified_frames_round_trip(modality):
    config_type, render, wrap, unwrap = (
        (InfraredCameraConfig, render_infrared_point_source_frame,
         raw_sensor_frame_from_infrared, infrared_frame_from_raw)
        if modality == "INFRARED" else
        (OpticalCameraConfig, render_optical_point_source_frame,
         raw_sensor_frame_from_optical, optical_frame_from_raw)
    )
    config = config_type(width=32, height=24)
    concrete = render(
        timestamp=2.0, observer_id="a", target_id="b",
        observer_state=OBSERVER, target_state=TARGET,
        quaternion_i2b_wxyz=QUATERNION, config=config,
        rng=np.random.default_rng(2),
    )
    raw = wrap(concrete, config=config)
    restored = unwrap(raw)
    assert raw.axes["pixel_x"].shape == (32,)
    assert raw.axes["pixel_y"].shape == (24,)
    np.testing.assert_array_equal(restored.image, concrete.image)


def test_raw_contract_rejects_kind_and_axis_mismatch():
    invalid = RawSensorFrame(
        timestamp=0.0, observer_id="a", target_id="b", modality="RADAR",
        data=np.ones((4, 5)), data_kind="OPTICAL_POINT_SOURCE_IMAGE",
        axes={"pixel_x": np.arange(5), "pixel_y": np.arange(4)},
    )
    with pytest.raises(InterfaceValidationError):
        validate_raw_sensor_frame(invalid)
