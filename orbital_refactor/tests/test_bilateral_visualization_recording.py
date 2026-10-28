import sys

import numpy as np

from experiments.run_bilateral_dropout_prescan import main as dropout_main
from experiments.run_bilateral_node_failure_prescan import main as failure_main
from experiments.run_bilateral_visualization_recording import (
    generate_bilateral_visualization_recording,
)
from visualization.recording import VisualizationRecordingReader
from visualization.acceptance_test_workflow import _functional_recording_audit


def test_robustness_runners_write_seed_zero_recording_in_process(
    tmp_path, monkeypatch,
):
    dropout_result = tmp_path / "dropout.json"
    dropout_recording = tmp_path / "dropout-recording"
    monkeypatch.setattr(sys, "argv", [
        "run_bilateral_dropout_prescan",
        "--duration", "3", "--dropout-start", "1", "--dropout-end", "2",
        "--seed", "0", "--modalities", "OPTICAL",
        "--acceptance-cann-only", "--local-unified-multimodal-update",
        "--electro-optical-radial-failover",
        "--output", str(dropout_result),
        "--recording-output", str(dropout_recording),
    ])
    dropout_main()

    dropout_frame = VisualizationRecordingReader(dropout_recording)[2]
    assert dropout_frame.metadata["recording_source"] == (
        "IN_PROCESS_ESTIMATOR_CAPTURE"
    )
    assert dropout_frame.metadata["acceptance_result_path"] == str(
        dropout_result
    )
    assert not any(
        item.modality == "OPTICAL" for item in dropout_frame.observations
    )
    assert dropout_frame.metadata["radial_backup_profile"] == "none"
    assert dropout_frame.metadata["electro_optical_radial_failover"]
    assert dropout_frame.metadata[
        "electro_optical_radial_failover_policy"
    ] == "radar_primary_failover"
    assert dropout_frame.metadata[
        "electro_optical_range_proxy_sigma_m"
    ] == 2.5
    assert dropout_frame.metadata[
        "electro_optical_range_rate_proxy_sigma_mps"
    ] == 0.010

    failure_result = tmp_path / "failure.json"
    failure_recording = tmp_path / "failure-recording"
    monkeypatch.setattr(sys, "argv", [
        "run_bilateral_node_failure_prescan",
        "--duration", "3", "--failure-start", "1", "--failure-end", "2",
        "--seed", "0", "--failed-observer-indices", "0", "5", "10",
        "--acceptance-cann-only", "--local-unified-multimodal-update",
        "--output", str(failure_result),
        "--recording-output", str(failure_recording),
    ])
    failure_main()

    failure_frame = VisualizationRecordingReader(failure_recording)[2]
    assert failure_frame.metadata["consensus_enabled"]
    assert len(failure_frame.metadata["failed_observer_ids"]) == 3


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
        duration=8.0,
        dt=1.0,
        seed=0,
        include_cann=True,
        enable_consensus=True,
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
    diagnostics = frame.metadata["estimation_diagnostics"]
    assert diagnostics["all_states_finite"] is True
    assert diagnostics["cann_preprocessed_observation_count"] > 0
    assert diagnostics["consensus_received_message_count"] > 0
    assert frame.metadata["estimation_architecture"] == "MULTI_COOPERATIVE"
    assert _functional_recording_audit(
        output,
        expected_architecture="MULTI_COOPERATIVE",
        require_consensus=True,
    )["passed"]

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
        include_cann=True,
        enable_consensus=False,
        tracking_observer_indices=(0,),
        tracking_target_indices=(0,),
        show_communication_topology=False,
        radial_backup_profile="none",
        retain_raw_frames=False,
    )
    frames = tuple(VisualizationRecordingReader(output))
    final = frames[-1]

    assert final.metadata["estimation_observer_scope"] == (
        "SELECTED_OBSERVERS_AFTER_COMMON_INITIALIZATION"
    )
    assert final.metadata["estimation_architecture"] == "SINGLE_OBSERVER"
    assert frames[0].metadata["estimation_phase"] == (
        "COMMON_MULTIOBSERVER_INITIALIZATION"
    )
    assert final.metadata["estimation_phase"] == "SELECTED_OBSERVER_TRACKING"
    assert final.metadata["topology_version"] == "not-active-for-estimation"
    assert tuple(final.metadata["estimation_observer_ids"]) == (
        "friendly_p01_s01",
    )
    assert tuple(final.metadata["estimation_target_ids"]) == (
        "hostile_p01_s01",
    )
    assert final.metadata["estimation_target_scope"] == (
        "SELECTED_HOSTILE_TARGETS_ONLY"
    )
    assert not any(edge.edge_type == "ACTIVE_TOPOLOGY" for edge in final.edges)
    assert {
        observation.observer_id for observation in final.observations
    } == {"friendly_p01_s01"}
    assert {
        observation.target_id
        for frame in frames for observation in frame.observations
    } == {"hostile_p01_s01"}
    assert {
        node.node_id
        for frame in frames for node in frame.nodes
        if node.status == "HOSTILE_TARGET_TRACKED"
    } <= {"hostile_p01_s01"}
    assert _functional_recording_audit(
        output,
        expected_architecture="SINGLE_OBSERVER",
        require_consensus=False,
    )["passed"]


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
        maximum_observation_range_m=7_000_000.0,
        process_noise_acceleration=2e-4,
        local_velocity_refinement_alpha=0.3,
        tracking_modalities=("OPTICAL",),
        dropout_modality="OPTICAL",
        dropout_start=4.0,
        dropout_end=np.nextafter(6.0, 4.0),
        configured_dropout_duration_s=2.0,
        retain_raw_frames=False,
    )
    frame = tuple(VisualizationRecordingReader(output))[-1]

    assert frame.scenario_id == "bilateral-6-friendly-4-hostile"
    assert frame.metadata["friendly_observer_count"] == 6
    assert frame.metadata["hostile_target_count"] == 4
    assert frame.metadata["maximum_observation_range_m"] == 7_000_000.0
    assert frame.metadata["process_noise_acceleration"] == 2e-4
    assert frame.metadata["tracking_modalities"] == ("OPTICAL",)
    assert frame.metadata["exported_measurement_modalities"] == (
        "OPTICAL",
    )
    assert frame.metadata["maneuver_profile"] == "NONE"
    assert frame.metadata["velocity_refinement"] == {
        "method": "causal-two-body-j2-residual-blend",
        "local_alpha": 0.3,
        "consensus_alpha": None,
        "maneuver_reset_threshold_mps": 0.5,
    }
    assert {
        observation.modality
        for recorded_frame in VisualizationRecordingReader(output)
        for observation in recorded_frame.observations
    } <= {"OPTICAL"}
    assert frame.metadata["estimation_architecture"] == "MULTI_INDEPENDENT"
    assert frame.metadata["tracking_modality_scope"] == (
        "SELECTED_MODALITIES_AFTER_COMMON_MULTIMODAL_INITIALIZATION"
    )
    assert frame.metadata["acceptance_dropout_modality"] == "OPTICAL"
    assert frame.metadata["dropout_start_seconds"] == 4.0
    assert frame.metadata["dropout_end_seconds"] < 6.0
    assert frame.metadata["configured_dropout_duration_s"] == 2.0
    assert len(frame.nodes) == 10
