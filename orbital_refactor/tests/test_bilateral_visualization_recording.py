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
    assert frames[0].metadata["fleet_position_rmse_m"] is None
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
    assert frame.metadata["radial_backup_profile"] == "none"


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


def test_acceptance_recording_applies_dropout_and_node_failure(tmp_path):
    dropout_output = generate_bilateral_visualization_recording(
        tmp_path / "acceptance-dropout",
        duration=8.0,
        dt=1.0,
        seed=0,
        include_cann=False,
        enable_consensus=True,
        radial_backup_profile="legacy",
        dropout_modality="RADAR",
        dropout_start=3.0,
        dropout_end=5.0,
        retain_raw_frames=False,
    )
    dropout_frame = tuple(VisualizationRecordingReader(dropout_output))[4]

    assert dropout_frame.metadata["acceptance_dropout_modality"] == "RADAR"
    assert dropout_frame.metadata["radial_backup_profile"] == "legacy"
    assert not any(
        observation.modality == "RADAR"
        for observation in dropout_frame.observations
    )

    failure_output = generate_bilateral_visualization_recording(
        tmp_path / "acceptance-failure",
        duration=8.0,
        dt=1.0,
        seed=0,
        include_cann=False,
        enable_consensus=True,
        radial_backup_profile="none",
        failed_observer_indices=(0, 1, 2),
        failure_start=3.0,
        failure_end=5.0,
        retain_raw_frames=False,
    )
    failure_frame = tuple(VisualizationRecordingReader(failure_output))[4]

    assert len(failure_frame.metadata["failed_observer_ids"]) == 3
    assert sum(
        node.status == "FRIENDLY_NODE_FAILED" for node in failure_frame.nodes
    ) == 3


def test_single_observer_recording_restricts_estimation_after_initialization(
    tmp_path,
):
    output = generate_bilateral_visualization_recording(
        tmp_path / "single-observer",
        duration=8.0,
        dt=1.0,
        seed=0,
        include_cann=False,
        enable_consensus=False,
        tracking_observer_indices=(0,),
        show_communication_topology=False,
        radial_backup_profile="none",
        retain_raw_frames=False,
    )
    frames = tuple(VisualizationRecordingReader(output))
    final = frames[-1]

    assert final.metadata["estimation_observer_scope"] == (
        "SELECTED_OBSERVERS_AFTER_COMMON_INITIALIZATION"
    )
    assert tuple(final.metadata["estimation_observer_ids"]) == (
        "friendly_p01_s01",
    )
    assert not any(edge.edge_type == "ACTIVE_TOPOLOGY" for edge in final.edges)
    assert {
        observation.observer_id for observation in final.observations
    } == {"friendly_p01_s01"}


def test_configurable_bilateral_walker_population_is_recorded(tmp_path):
    output = generate_bilateral_visualization_recording(
        tmp_path / "bilateral-6x4",
        duration=8.0,
        dt=1.0,
        seed=0,
        include_cann=False,
        radial_backup_profile="none",
        friendly_total_satellites=6,
        friendly_plane_count=2,
        friendly_phasing=1,
        friendly_altitude_m=720_000.0,
        hostile_total_satellites=4,
        hostile_plane_count=2,
        hostile_phasing=1,
        hostile_altitude_m=760_000.0,
        retain_raw_frames=False,
    )
    frame = tuple(VisualizationRecordingReader(output))[-1]

    assert frame.scenario_id == "bilateral-6-friendly-4-hostile"
    assert frame.metadata["friendly_observer_count"] == 6
    assert frame.metadata["hostile_target_count"] == 4
    assert len(frame.nodes) == 10
