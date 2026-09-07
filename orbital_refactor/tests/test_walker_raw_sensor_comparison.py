import numpy as np
from dataclasses import replace
from types import SimpleNamespace

from adapters.infrared_image_adapter import InfraredCameraConfig
from adapters.radar_range_doppler_adapter import RadarRangeDopplerConfig
from experiments.inter_satellite_observation_factory import (
    target_pointing_quaternion,
)
from experiments.walker_raw_sensor_comparison import (
    build_lagged_radar_acquisition_centers,
    replace_infrared_messages_with_body_pair,
    replace_radar_messages_with_power_maps,
    replace_radar_messages_with_reacquisition,
)
from interfaces.data_objects import ObservationMessage


def _case():
    times = np.array([0.0])
    observer = np.array([7e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    target = observer + np.array([1000.0, 20.0, -10.0, 1.0, 0.0, 0.0])
    truth = {"a": observer.reshape(1, 6), "b": target.reshape(1, 6)}
    infrared = ObservationMessage(
        message_id="ir", observer_id="a", target_id="b", timestamp=0.0,
        modality="INFRARED", measurement=np.zeros(2),
        covariance=np.eye(2) * 1e-6,
    )
    radar = ObservationMessage(
        message_id="radar", observer_id="a", target_id="b", timestamp=0.0,
        modality="RADAR", measurement=np.zeros(2), covariance=np.eye(2),
    )
    return times, truth, infrared, radar


def test_infrared_pair_preserves_identity_and_uses_body_frame():
    times, truth, infrared, radar = _case()
    analytic, image = replace_infrared_messages_with_body_pair(
        [infrared, radar], timestamps=times,
        truth_state_history_by_node=truth,
        config=InfraredCameraConfig(read_noise_sigma=0.0),
        analytic_rng=np.random.default_rng(0),
        image_rng=np.random.default_rng(1),
    )
    assert analytic[0].message_id == image[0].message_id == "ir"
    assert analytic[0].frame == image[0].frame == "BODY"
    assert "quaternion_i2b_wxyz" in image[0].metadata
    assert analytic[1] is image[1] is radar


def test_radar_map_replacement_preserves_message_identity():
    times, truth, infrared, radar = _case()
    result = replace_radar_messages_with_power_maps(
        [infrared, radar], timestamps=times,
        truth_state_history_by_node=truth,
        config=RadarRangeDopplerConfig(read_noise_sigma=0.0),
        rng=np.random.default_rng(0),
    )
    assert result[0] is infrared
    assert result[1].message_id == "radar"
    assert result[1].metadata["raw_source_type"] == (
        "RANGE_DOPPLER_POWER_MAP"
    )


def test_lagged_radar_centers_do_not_use_current_epoch_posterior():
    times, truth, _, radar = _case()
    times = np.array([0.0, 2.0])
    radar_later = replace(radar, message_id="radar:2", timestamp=2.0)
    initial = {node: values[0].copy() for node, values in truth.items()}
    previous_observer = initial["a"] + np.array([4.0, 0, 0, 0, 0, 0])
    previous_target = initial["b"] + np.array([-3.0, 0, 0, 0, 0, 0])

    def history(current_offset):
        return SimpleNamespace(
            active_state_history_by_node={
                "a": np.vstack([previous_observer,
                                previous_observer + current_offset]),
                "b": np.vstack([previous_target,
                                previous_target - current_offset]),
            },
            neighbor_state_history_by_node={
                "a": {"b": np.vstack([
                    previous_target, previous_target + current_offset,
                ])},
            },
        )

    case = {
        "timestamps": times,
        "observations": [radar, radar_later],
        "initial_states": initial,
    }
    first = build_lagged_radar_acquisition_centers(case, history(1e3))
    second = build_lagged_radar_acquisition_centers(case, history(1e6))
    np.testing.assert_allclose(first["radar:2"], second["radar:2"])


def test_radar_reacquisition_expands_then_returns_to_tracking():
    _, one_step_truth, _, radar = _case()
    times = np.array([0.0, 2.0, 4.0])
    messages = [
        replace(radar, message_id=f"radar:{time:g}", timestamp=time)
        for time in times
    ]
    truth = {
        node: np.repeat(values, times.size, axis=0)
        for node, values in one_step_truth.items()
    }
    ideal = np.array([
        np.linalg.norm(truth["b"][0, :3] - truth["a"][0, :3]),
        0.0,
    ])
    centers = {message.message_id: ideal.copy() for message in messages}
    tracking = RadarRangeDopplerConfig(
        width=65, height=65, range_bin_size_m=5.0,
        range_rate_bin_size_mps=0.1, read_noise_sigma=0.0,
    )
    output = replace_radar_messages_with_reacquisition(
        messages, timestamps=times, truth_state_history_by_node=truth,
        acquisition_center_by_message_id=centers,
        initial_offset=np.array([150.0, 0.0]),
        tracking_config=tracking,
        reacquisition_config=replace(
            tracking, range_bin_size_m=10.0,
            range_rate_bin_size_mps=0.2,
        ),
        rng=np.random.default_rng(0),
    )
    assert [item.valid_flag for item in output] == [False, True, True]
    assert [
        item.metadata["acquisition_search_mode"] for item in output
    ] == ["tracking", "reacquisition", "tracking"]


def test_radar_reacquisition_escalates_after_consecutive_failures():
    _, one_step_truth, _, radar = _case()
    times = np.array([0.0, 2.0, 4.0, 6.0])
    messages = [
        replace(radar, message_id=f"radar:{time:g}", timestamp=time)
        for time in times
    ]
    truth = {
        node: np.repeat(values, times.size, axis=0)
        for node, values in one_step_truth.items()
    }
    ideal = np.array([
        np.linalg.norm(truth["b"][0, :3] - truth["a"][0, :3]), 0.0,
    ])
    centers = {message.message_id: ideal.copy() for message in messages}
    tracking = RadarRangeDopplerConfig(
        width=65, height=65, range_bin_size_m=5.0,
        range_rate_bin_size_mps=0.1, read_noise_sigma=0.0,
    )
    output = replace_radar_messages_with_reacquisition(
        messages, timestamps=times, truth_state_history_by_node=truth,
        acquisition_center_by_message_id=centers,
        initial_offset=np.array([400.0, 0.0]),
        tracking_config=tracking,
        reacquisition_config=replace(
            tracking, range_bin_size_m=10.0,
            range_rate_bin_size_mps=0.2,
        ),
        maximum_reacquisition_scale=4.0,
        rng=np.random.default_rng(0),
    )
    assert [item.valid_flag for item in output] == [False, False, True, True]
    assert [
        item.metadata["acquisition_window_scale"] for item in output
    ] == [1.0, 2.0, 4.0, 1.0]
