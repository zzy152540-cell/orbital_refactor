import numpy as np

from experiments.run_bilateral_visualization_recording import (
    generate_bilateral_visualization_recording,
)
from visualization.recording import VisualizationRecordingReader


def test_bilateral_recording_contains_roles_edges_observations_and_estimates(tmp_path):
    output = generate_bilateral_visualization_recording(
        tmp_path / "bilateral",
        duration=4.0,
        dt=1.0,
        seed=0,
        include_cann=False,
        retain_raw_frames=True,
    )
    frames = tuple(VisualizationRecordingReader(output))

    assert len(frames) == 5
    frame = frames[-1]
    assert len(frame.nodes) == 30
    assert sum(node.status.startswith("FRIENDLY") for node in frame.nodes) == 15
    assert sum(node.status.startswith("HOSTILE") for node in frame.nodes) == 15
    assert {item.modality for item in frame.observations} == {
        "RADAR", "INFRARED", "OPTICAL",
    }
    assert any(item.raw_sensor_data is not None for item in frame.observations)
    assert any(item.edge_type == "ACTIVE_TOPOLOGY" for item in frame.edges)
    assert any(item.edge_type == "OBSERVATION_LINK" for item in frame.edges)
    hostile = tuple(
        node for node in frame.nodes if node.status.startswith("HOSTILE")
    )
    assert all(node.truth_state is not None for node in hostile)
    assert frame.metadata["friendly_observer_count"] == 15
    assert frame.metadata["hostile_target_count"] == 15
    assert np.isfinite(frame.metadata["fleet_position_rmse_m"])


def test_bilateral_recording_cann_exposes_absolute_eci_position(tmp_path):
    output = generate_bilateral_visualization_recording(
        tmp_path / "bilateral-cann",
        duration=4.0,
        dt=1.0,
        seed=0,
        include_cann=True,
        cann_node_limit=1,
        retain_raw_frames=False,
    )
    frame = tuple(VisualizationRecordingReader(output))[-1]
    absolute = next(
        item for item in frame.cann
        if item.representation == "ABSOLUTE_POSITION_ECI"
    )
    assert absolute.node_id.startswith("hostile_")
    assert absolute.decoded_value.shape == (3,)
    assert np.all(np.isfinite(absolute.decoded_value))
    assert absolute.metrics["source"] == "DIRECTION_RING_PLUS_RT_LINE_PAIR"
    assert absolute.metrics["input_state_source"] == "TARGET_FILTER_POSTERIOR"
    assert absolute.metrics["units"] == "m"
    assert absolute.metrics["position_error_to_posterior_m"] < 100.0

    frames = tuple(VisualizationRecordingReader(output))
    assert not frames[0].cann
    assert not frames[1].cann
    assert frames[0].navigation[-1].status == "TARGET_CANN_NOT_RECORDED"
    selected = next(
        item for item in frames[0].navigation
        if item.node_id == absolute.node_id
    )
    assert selected.status == "TARGET_FILTER_UNINITIALIZED"
