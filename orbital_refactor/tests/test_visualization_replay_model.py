import shutil
from pathlib import Path

import numpy as np

from visualization.data_contract import (
    VisualCANNSnapshot,
    VisualNodeState,
    VisualizationFrame,
)
from visualization.recording import VisualizationRecordingWriter
from visualization.replay_model import VisualizationReplayModel


def test_replay_model_exposes_timeline_and_node_series():
    target = Path("results/test_visualization_model")
    if target.exists():
        shutil.rmtree(target)
    try:
        with VisualizationRecordingWriter(target) as writer:
            for index in range(3):
                writer.append(VisualizationFrame(
                    scenario_id="test", run_id="seed-0",
                    timestamp=2.0 * index, epoch_index=index,
                    nodes=tuple(VisualNodeState(
                        node_id=f"sat_{node + 1:02d}",
                        estimate_state=np.arange(6.0) + index + node,
                        truth_state=np.arange(6.0) + node,
                        covariance_diagonal=np.ones(6),
                    ) for node in range(2)),
                    metadata={
                        "fleet_position_rmse_m": (
                            None if index == 0 else float(index)
                        )
                    },
                ))
        model = VisualizationReplayModel(target)
        assert len(model) == 3
        assert len(model.node_ids) == 2
        assert model.timestamps.tolist() == [0.0, 2.0, 4.0]
        truth, estimate = model.node_position_history(model.node_ids[0])
        assert truth.shape == estimate.shape == (3, 3)
        assert model.node_position_error_history(model.node_ids[0]).shape == (3,)
        assert (
            model.node_position_error_history(model.node_ids[0])
            is model.node_position_error_history(model.node_ids[0])
        )
        np.testing.assert_allclose(
            model.node_velocity_error_history(model.node_ids[0]),
            np.sqrt(3.0) * np.arange(3),
        )
        acceleration_error = model.node_acceleration_error_history(
            model.node_ids[0]
        )
        assert np.isnan(acceleration_error[0])
        np.testing.assert_allclose(
            acceleration_error[1:], np.full(2, np.sqrt(3.0) / 2.0),
        )
        assert model.fleet_velocity_rmse_history.shape == (3,)
        assert model.fleet_acceleration_rmse_history.shape == (3,)
        assert np.isfinite(model.fleet_position_rmse_history[0])
    finally:
        if target.exists():
            shutil.rmtree(target)


def test_replay_model_preserves_late_cann_initialization_as_nan_history():
    target = Path("results/test_visualization_model_late_cann")
    if target.exists():
        shutil.rmtree(target)
    try:
        with VisualizationRecordingWriter(target) as writer:
            for index in range(3):
                writer.append(VisualizationFrame(
                    scenario_id="test", run_id="seed-0",
                    timestamp=float(index), epoch_index=index,
                    nodes=(VisualNodeState(
                        node_id="target_01",
                        estimate_state=np.arange(6.0) + index,
                        truth_state=np.arange(6.0),
                        covariance_diagonal=np.ones(6),
                        valid=index > 0,
                    ),),
                    cann=(() if index == 0 else (VisualCANNSnapshot(
                        node_id="target_01",
                        representation="ABSOLUTE_POSITION_ECI",
                        available=True,
                        decoded_value=np.full(3, float(index)),
                    ),)),
                ))

        model = VisualizationReplayModel(target)
        _, position = model.node_position_history("target_01")
        decoded = model.cann_decoded_history(
            "target_01", "ABSOLUTE_POSITION_ECI",
        )
        assert np.isnan(position[0]).all()
        assert decoded is not None
        assert decoded.shape == (3, 3)
        assert np.isnan(decoded[0]).all()
        np.testing.assert_allclose(decoded[1:], [[1.0] * 3, [2.0] * 3])
    finally:
        if target.exists():
            shutil.rmtree(target)
