import numpy as np
import pytest

from interfaces.data_objects import InitialState, ModuleInput, Observation
from interfaces.state_awareness_module import StateAwarenessModule
from orbital_core.dynamics import make_process_noise
from tracking import (
    TargetInitialState,
    TrackLifecycle,
    run_cooperative_target_fusion,
    run_known_target_batch,
    run_known_target_sequence,
    run_local_target_filter,
    run_local_target_history,
)


def _module_input(observer_id="observer_a", target_id="target_01", offset=0.0):
    timestamps = np.array([0.0, 1.0, 2.0])
    chief = np.array([
        [7.0e6 + offset, 0.0, 0.0, 0.0, 7500.0, 0.0],
        [7.0e6 + offset, 7500.0, 0.0, -8.0, 7500.0, 0.0],
        [7.0e6 - 8.0 + offset, 15000.0, 0.0, -16.0, 7499.99, 0.0],
    ])
    relative = np.array([100.0, 50.0, 500.0, 0.1, -0.05, 0.0])
    observations = []
    for index, timestamp in enumerate(timestamps):
        observations.extend([
            Observation(
                timestamp=float(timestamp), observer_id=observer_id,
                target_id=target_id, modality="OPTICAL", source_type="LEARNING",
                measurement=relative[:3] + np.array([0.1 * index, 0.0, 0.0]),
                covariance=np.diag([9.0, 9.0, 16.0]), confidence=0.9,
                frame="ECI", valid_flag=True,
                metadata={"measurement_id": f"{observer_id}:opt:{index}"},
            ),
            Observation(
                timestamp=float(timestamp), observer_id=observer_id,
                target_id=target_id, modality="RADAR", source_type="TRADITIONAL",
                measurement=np.array([512.35, 0.02]),
                covariance=np.diag([25.0, 0.04]), confidence=1.0,
                frame="SPRI", valid_flag=True,
                metadata={"measurement_id": f"{observer_id}:rad:{index}"},
            ),
        ])
    return ModuleInput(
        initial_state=InitialState(
            target_id=target_id, timestamp=0.0, state_estimate=relative,
            covariance=np.diag([100.0] * 3 + [0.1] * 3),
        ),
        sensor_measurements=observations,
        config={
            "runtime": {
                "timestamps": timestamps,
                "chief_state_history_eci": chief,
                "q_eci2pri_history": np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
                "node_id": observer_id,
            },
            "filter": {
                "process_noise": make_process_noise(1.0, 1e-4),
                "reset_feedback": True,
                "ci_grid_points": 11,
            },
            "modalities": {
                "nn": {"nn_meas_frame": "eci"},
                "radar": {},
            },
        },
    )


def test_local_filter_converts_relative_posterior_to_absolute_j2000():
    module_input = _module_input()
    history = StateAwarenessModule().run_history(module_input)
    relative = history.final_fusion_result(
        node_id="observer_a", target_id="target_01",
    ).state_estimate

    report = run_local_target_filter(module_input)

    np.testing.assert_allclose(
        report.state_eci,
        module_input.config["runtime"]["chief_state_history_eci"][-1] + relative,
    )
    assert report.observer_id == "observer_a"
    assert report.target_id == "target_01"
    assert report.track_id == "target_01"
    assert set(report.modality_weights) == {"nn", "rad"}
    assert report.used_measurement_ids == (
        "observer_a:opt:2", "observer_a:rad:2",
    )


def test_local_filter_rejects_mixed_observers_and_node_mismatch():
    mixed = _module_input()
    mixed.sensor_measurements[-1].observer_id = "observer_b"
    with pytest.raises(ValueError, match="exactly one observer_id"):
        run_local_target_filter(mixed)

    mismatch = _module_input()
    mismatch.config["runtime"]["node_id"] = "observer_b"
    with pytest.raises(ValueError, match="node_id"):
        run_local_target_filter(mismatch)


def test_three_existing_local_filters_feed_one_target_level_ci():
    reports = [
        run_local_target_filter(_module_input(observer_id, offset=offset))
        for observer_id, offset in (
            ("observer_a", 0.0),
            ("observer_b", 20.0),
            ("observer_c", -20.0),
        )
    ]

    output = run_cooperative_target_fusion(
        scene_id="scene-local-filters",
        timestamp=2.0,
        expected_target_ids=("target_01",),
        reports=reports,
    )

    estimate = output.estimates_by_target["target_01"]
    assert estimate.contributing_observer_ids == (
        "observer_a", "observer_b", "observer_c",
    )
    assert np.isfinite(estimate.state_eci).all()
    assert np.linalg.eigvalsh(estimate.covariance_eci).min() >= -1e-10


def test_known_target_batch_runs_three_observers_against_two_targets():
    tasks = []
    for target_id, target_offset in (("target_01", 0.0), ("target_02", 1000.0)):
        for observer_id, observer_offset in (
            ("observer_a", 0.0),
            ("observer_b", 20.0),
            ("observer_c", -20.0),
        ):
            tasks.append(_module_input(
                observer_id=observer_id,
                target_id=target_id,
                offset=target_offset + observer_offset,
            ))

    output = run_known_target_batch(
        scene_id="scene-known-targets",
        module_inputs=tasks,
        track_id_by_target={
            "target_01": "track-001",
            "target_02": "track-002",
        },
        expected_target_ids=("target_01", "target_02"),
    )

    assert set(output.estimates_by_target) == {"target_01", "target_02"}
    assert output.estimates_by_target["target_01"].track_id == "track-001"
    assert output.estimates_by_target["target_02"].track_id == "track-002"
    for estimate in output.estimates_by_target.values():
        assert estimate.contributing_observer_ids == (
            "observer_a", "observer_b", "observer_c",
        )


def test_known_target_batch_rejects_duplicate_local_task():
    task = _module_input()
    with pytest.raises(ValueError, match="one local ModuleInput"):
        run_known_target_batch(
            scene_id="scene",
            module_inputs=[task, task],
        )


def test_local_history_marks_prediction_only_epoch_invalid_for_node_fusion():
    module_input = _module_input()
    for observation in module_input.sensor_measurements:
        if observation.timestamp == 1.0:
            observation.valid_flag = False

    reports = run_local_target_history(module_input)

    assert [report.valid_flag for report in reports] == [True, False, True]
    assert reports[1].modality_weights == {}


def test_known_target_sequence_runs_filter_ci_and_lifecycle_per_epoch():
    task_a = _module_input("observer_a")
    task_b = _module_input("observer_b", offset=20.0)
    for task in (task_a, task_b):
        for observation in task.sensor_measurements:
            if observation.timestamp == 1.0:
                observation.valid_flag = False
    initial = TargetInitialState(
        target_id="target_01", track_id="track-001", timestamp=0.0,
        state_eci=(
            task_a.config["runtime"]["chief_state_history_eci"][0]
            + task_a.initial_state.state_estimate
        ),
        covariance_eci=task_a.initial_state.covariance,
    )

    history = run_known_target_sequence(
        scene_id="scene-sequence",
        module_inputs=[task_a, task_b],
        initial_target_states={"target_01": initial},
        max_coast_epochs=2,
    )

    assert history.timestamps.tolist() == [0.0, 1.0, 2.0]
    assert [
        track.lifecycle for track in history.track_history_by_target["target_01"]
    ] == [
        TrackLifecycle.TRACKING,
        TrackLifecycle.COASTING,
        TrackLifecycle.TRACKING,
    ]
    assert set(history.output_by_epoch[0].estimates_by_target) == {"target_01"}
    assert history.output_by_epoch[1].estimates_by_target == {}
    assert set(history.local_report_history_by_link) == {
        ("observer_a", "target_01"), ("observer_b", "target_01"),
    }
