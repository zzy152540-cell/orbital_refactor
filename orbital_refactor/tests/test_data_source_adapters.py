from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from adapters.nn_prediction_adapter import (
    build_pseudo_velocity,
    create_nn_observations,
    load_aligned_nn_positions,
)
from adapters.shirt_data_adapter import build_shirt_module_input, load_shirt_orbit_dataset
from adapters.synthetic_measurement_adapter import (
    apply_dropout_windows,
    create_infrared_observations,
    create_radar_observations,
)
from interfaces.state_awareness_module import StateAwarenessModule


def _write_synthetic_shirt(tmp_path: Path, sample_count: int = 5) -> tuple[Path, Path, list[str]]:
    filenames = [f"frame_{index:04d}.png" for index in range(sample_count)]
    chief = np.tile(np.array([7.0e6, 0.0, 0.0, 0.0, 7.5e3, 0.0]), (sample_count, 1))
    relative = np.column_stack(
        (
            100.0 + np.arange(sample_count),
            20.0 + 0.5 * np.arange(sample_count),
            np.full(sample_count, 5.0),
            np.ones(sample_count),
            np.full(sample_count, 0.5),
            np.zeros(sample_count),
        )
    )
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (sample_count, 1))
    metadata = {
        "pSim": {"cam_step": 1.0},
        "sAbsState": {
            "rv_eci2com_eci": chief.tolist(),
            "q_eci2pri": quaternions.tolist(),
            "w_pri": np.zeros((sample_count, 3)).tolist(),
        },
        "tRelState": {
            "rv_scom2tcom_spri": relative.tolist(),
            "q_spri2tpri": quaternions.tolist(),
            "w_tpri2spri_tpri": np.zeros((sample_count, 3)).tolist(),
        },
    }
    roe = [
        {
            "filename": filename,
            "q_vbs2tango_true": [1.0, 0.0, 0.0, 0.0],
            "r_Vo2To_vbs_true": relative[index, :3].tolist(),
        }
        for index, filename in enumerate(filenames)
    ]
    metadata_path = tmp_path / "metadata.json"
    roe_path = tmp_path / "roe1.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    roe_path.write_text(json.dumps(roe), encoding="utf-8")
    return metadata_path, roe_path, filenames


def test_shirt_loader_preserves_legacy_identity_rotation(tmp_path: Path):
    metadata_path, roe_path, _ = _write_synthetic_shirt(tmp_path)
    dataset = load_shirt_orbit_dataset(metadata_path, roe_path, "roe1")

    np.testing.assert_allclose(
        dataset.relative_state_eci[:, :3], dataset.relative_position_spri
    )
    np.testing.assert_allclose(
        dataset.target_state_eci, dataset.chief_state_eci + dataset.relative_state_eci
    )
    assert dataset.preprocess_summary["quaternion_order"] == "wxyz"


def test_nn_alignment_and_pseudo_velocity(tmp_path: Path):
    _, _, filenames = _write_synthetic_shirt(tmp_path)
    np.savez(
        tmp_path / "predictions.npz",
        image_path=np.array([f"/images/{filenames[2]}", f"/images/{filenames[0]}"]),
        t_pred=np.array([[3.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    positions, valid = load_aligned_nn_positions(tmp_path / "predictions.npz", filenames)
    assert valid.tolist() == [True, False, True, False, False]
    np.testing.assert_allclose(positions[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(positions[2], [3.0, 0.0, 0.0])

    complete_positions = np.column_stack((np.arange(5.0), np.zeros(5), np.zeros(5)))
    velocity, velocity_valid = build_pseudo_velocity(
        complete_positions, np.ones(5, dtype=bool), np.arange(5.0)
    )
    np.testing.assert_allclose(velocity[:, 0], 1.0)
    assert velocity_valid.all()


def test_measurement_factories_and_dropout(tmp_path: Path):
    metadata_path, roe_path, _ = _write_synthetic_shirt(tmp_path)
    dataset = load_shirt_orbit_dataset(metadata_path, roe_path, "roe1")
    valid = apply_dropout_windows(
        np.ones(len(dataset.timestamps), dtype=bool), dataset.timestamps, [(1.0, 2.0)]
    )
    assert valid.tolist() == [True, False, False, True, True]

    rng = np.random.default_rng(1)
    ir = create_infrared_observations(
        timestamps=dataset.timestamps,
        relative_position_spri=dataset.relative_position_spri,
        covariance=np.diag([1e-6, 1e-6]),
        observer_id="sat_01",
        target_id="target_01",
        rng=rng,
        valid_flags=valid,
    )
    radar = create_radar_observations(
        timestamps=dataset.timestamps,
        relative_position_spri=dataset.relative_position_spri,
        relative_velocity_spri=dataset.relative_velocity_spri,
        covariance=np.diag([1.0, 1e-4]),
        observer_id="sat_01",
        target_id="target_01",
        rng=rng,
    )
    assert len(ir) == len(radar) == len(dataset.timestamps)
    assert ir[1].valid_flag is False
    assert radar[0].metadata["measurement_type"] == "RANGE_RANGE_RATE"


def test_real_data_adapter_to_standard_module_output(tmp_path: Path):
    metadata_path, roe_path, filenames = _write_synthetic_shirt(tmp_path, sample_count=6)
    dataset = load_shirt_orbit_dataset(metadata_path, roe_path, "roe1")
    np.savez(
        tmp_path / "predictions.npz",
        image_path=np.array(filenames),
        t_pred=dataset.relative_position_spri.copy(),
    )
    nn_positions, nn_valid = load_aligned_nn_positions(
        tmp_path / "predictions.npz", dataset.filenames
    )
    nn_observations = create_nn_observations(
        timestamps=dataset.timestamps,
        positions=nn_positions,
        valid_positions=nn_valid,
        covariance_position=np.diag([0.1, 0.1, 0.1]) ** 2,
        covariance_velocity=np.diag([0.1, 0.1, 0.1]) ** 2,
        observer_id="sat_01",
        target_id="target_01",
        frame="SPRI",
        use_pseudo_velocity=True,
    )
    rng = np.random.default_rng(2)
    ir_observations = create_infrared_observations(
        timestamps=dataset.timestamps,
        relative_position_spri=dataset.relative_position_spri,
        covariance=np.diag(np.deg2rad([1.5, 1.5])) ** 2,
        observer_id="sat_01",
        target_id="target_01",
        rng=rng,
    )
    radar_observations = create_radar_observations(
        timestamps=dataset.timestamps,
        relative_position_spri=dataset.relative_position_spri,
        relative_velocity_spri=dataset.relative_velocity_spri,
        covariance=np.diag([1.0, 0.02]) ** 2,
        observer_id="sat_01",
        target_id="target_01",
        rng=rng,
    )
    module_input = build_shirt_module_input(
        dataset,
        node_id="sat_01",
        target_id="target_01",
        process_noise_acceleration=1e-4,
        initial_position_std=10.0,
        initial_velocity_std=0.05,
        observations=nn_observations + ir_observations + radar_observations,
        filter_config={
            "reset_feedback": True,
            "ci_objective": "trace",
            "ci_grid_points": 21,
            "gate_enable": False,
        },
        modalities_config={
            "nn": {"nn_meas_frame": "spri", "nn_use_pseudo_velocity": True}
        },
    )
    output = StateAwarenessModule().run(module_input)
    assert output.state_output.position_estimate.shape == (3,)
    assert output.state_output.velocity_estimate.shape == (3,)
    assert output.state_output.acceleration_estimate.shape == (3,)
    assert set(output.fusion_status.modality_weights) == {"nn", "ir", "rad"}
