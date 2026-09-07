import numpy as np

from adapters.optical_image_adapter import OpticalCameraConfig
from experiments.inter_satellite_observation_factory import (
    target_pointing_quaternion,
)
from experiments.walker_optical_image_comparison import (
    replace_optical_messages_with_images,
)
from interfaces.data_objects import ObservationMessage


def test_image_replacement_changes_only_optical_payload_and_keeps_lineage():
    timestamps = np.array([0.0])
    observer = np.array([7e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    target = observer + np.array([1000.0, 20.0, -10.0, 0.0, 0.0, 0.0])
    quaternion = target_pointing_quaternion(observer, target)
    optical = ObservationMessage(
        message_id="optical-message", physical_observation_id="physical-optical",
        observer_id="a", target_id="b", timestamp=0.0,
        modality="OPTICAL", measurement=np.array([9.0, 9.0]),
        covariance=np.eye(2), frame="BODY",
        metadata={"quaternion_i2b_wxyz": quaternion},
    )
    radar = ObservationMessage(
        message_id="radar-message", observer_id="a", target_id="b",
        timestamp=0.0, modality="RADAR", measurement=np.array([1.0, 2.0]),
        covariance=np.eye(2),
    )
    config = OpticalCameraConfig(
        width=32, height=32, focal_length_x_pixels=20.0,
        focal_length_y_pixels=20.0, read_noise_sigma=0.0,
    )
    result, frames = replace_optical_messages_with_images(
        [optical, radar], timestamps=timestamps,
        truth_state_history_by_node={
            "a": observer.reshape(1, 6), "b": target.reshape(1, 6),
        },
        config=config, rng=np.random.default_rng(0), retain_raw_frames=True,
    )

    by_id = {message.message_id: message for message in result}
    assert by_id["radar-message"] is radar
    assert by_id["optical-message"].information_id == "physical-optical"
    assert by_id["optical-message"].metadata["raw_source_type"] == (
        "POINT_SOURCE_IMAGE"
    )
    assert frames["physical-optical"].image.shape == (32, 32)
