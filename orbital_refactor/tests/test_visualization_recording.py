import shutil
from pathlib import Path

import numpy as np
import pytest

from visualization.data_contract import (
    VisualCANNSnapshot,
    VisualNodeState,
    VisualizationFrame,
)
from visualization.recording import (
    VisualizationRecordingReader,
    VisualizationRecordingWriter,
)


def _frame(index):
    return VisualizationFrame(
        scenario_id="walker5", run_id="seed-0",
        timestamp=float(index * 2), epoch_index=index,
        nodes=(VisualNodeState(
            node_id="sat_01", estimate_state=np.arange(6.0) + index,
            truth_state=np.arange(6.0), covariance_diagonal=np.ones(6),
        ),),
        cann=(VisualCANNSnapshot(
            node_id="sat_01", representation="DIRECTION_RING",
            available=True, activity=np.linspace(0.0, 1.0, 32),
            decoded_value=np.array([0.1 * index]),
        ),),
    )


def test_recording_round_trip_and_random_access():
    target = Path("results/test_visualization_recording")
    if target.exists():
        shutil.rmtree(target)
    try:
        with VisualizationRecordingWriter(target) as writer:
            writer.append(_frame(0))
            writer.append(_frame(1))

        reader = VisualizationRecordingReader(target)
        assert len(reader) == 2
        assert reader.manifest.start_timestamp == 0.0
        assert reader[-1].epoch_index == 1
        np.testing.assert_array_equal(
            reader[1].nodes[0].estimate_state, np.arange(6.0) + 1.0,
        )
        assert not reader[0].cann[0].activity.flags.writeable
        array_files = tuple((target / "arrays").iterdir())
        assert len(array_files) == 2
        assert all(path.suffix == ".npz" for path in array_files)
    finally:
        if target.exists():
            shutil.rmtree(target)


def test_recording_rejects_noncontiguous_epochs():
    target = Path("results/test_visualization_bad_recording")
    if target.exists():
        shutil.rmtree(target)
    try:
        writer = VisualizationRecordingWriter(target)
        with pytest.raises(ValueError, match="contiguous"):
            writer.append(_frame(1))
    finally:
        if target.exists():
            shutil.rmtree(target)
