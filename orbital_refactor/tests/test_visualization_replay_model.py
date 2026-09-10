import shutil
from pathlib import Path

import numpy as np

from visualization.data_contract import VisualNodeState, VisualizationFrame
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
                    metadata={"fleet_position_rmse_m": float(index)},
                ))
        model = VisualizationReplayModel(target)
        assert len(model) == 3
        assert len(model.node_ids) == 2
        assert model.timestamps.tolist() == [0.0, 2.0, 4.0]
        truth, estimate = model.node_position_history(model.node_ids[0])
        assert truth.shape == estimate.shape == (3, 3)
        assert model.node_position_error_history(model.node_ids[0]).shape == (3,)
    finally:
        if target.exists():
            shutil.rmtree(target)
