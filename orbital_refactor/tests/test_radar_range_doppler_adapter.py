import numpy as np

from adapters.radar_range_doppler_adapter import (
    RadarRangeDopplerConfig,
    radar_frame_to_observation_message,
    render_radar_range_doppler_frame,
)
from cooperative.network_schmidt_runner import run_network_schmidt_filter
from cooperative.topology import chain_topology
from orbital_core.measurements import (
    measure_relative_range,
    measure_relative_range_rate,
)


def _states():
    observer = np.array([7e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    target = observer + np.array([1000.0, 100.0, -50.0, 1.0, -0.2, 0.1])
    return observer, target


def test_noiseless_range_doppler_centroid_recovers_measurement():
    observer, target = _states()
    expected = np.array([
        measure_relative_range(observer, target),
        measure_relative_range_rate(observer, target),
    ])
    config = RadarRangeDopplerConfig(
        width=65, height=65, range_bin_size_m=5.0,
        range_rate_bin_size_mps=0.02, read_noise_sigma=0.0,
    )
    frame = render_radar_range_doppler_frame(
        timestamp=0.0, observer_id="a", target_id="b",
        observer_state=observer, target_state=target,
        acquisition_range_m=expected[0] - 1.3,
        acquisition_range_rate_mps=expected[1] + 0.007,
        config=config,
    )
    message = radar_frame_to_observation_message(frame, config=config)

    assert frame.power.shape == (65, 65)
    assert message.valid_flag
    assert 0.0 <= message.metadata["frontend_quality_score"] <= 1.0
    assert message.metadata["peak_snr"] > 0.0
    assert abs(message.measurement[0] - expected[0]) < 1e-3
    assert abs(message.measurement[1] - expected[1]) < 1e-5


def test_target_outside_range_doppler_window_is_invalid():
    observer, target = _states()
    frame = render_radar_range_doppler_frame(
        timestamp=0.0, observer_id="a", target_id="b",
        observer_state=observer, target_state=target,
        acquisition_range_m=0.0, acquisition_range_rate_mps=0.0,
    )
    message = radar_frame_to_observation_message(frame)
    assert not frame.target_in_window
    assert not message.valid_flag


def test_range_doppler_message_runs_through_network_schmidt():
    observer, target = _states()
    expected_range = measure_relative_range(observer, target)
    expected_rate = measure_relative_range_rate(observer, target)
    config = RadarRangeDopplerConfig(read_noise_sigma=0.0)
    frame = render_radar_range_doppler_frame(
        timestamp=0.0, observer_id="a", target_id="b",
        observer_state=observer, target_state=target,
        acquisition_range_m=expected_range,
        acquisition_range_rate_mps=expected_rate, config=config,
    )
    message = radar_frame_to_observation_message(frame, config=config)
    history = run_network_schmidt_filter(
        timestamps=np.array([0.0, 2.0]),
        initial_state_by_node={"a": observer, "b": target},
        initial_covariance_by_node={"a": np.eye(6), "b": np.eye(6)},
        topology=chain_topology(["a", "b"]),
        observation_messages=[message], process_noise_acceleration=0.0,
    )
    assert np.isfinite(history.active_state_history_by_node["a"]).all()
    assert set(history.nis_history_by_node["a"][0]) == {
        message.information_id,
    }
