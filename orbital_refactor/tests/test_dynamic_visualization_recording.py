import shutil
from pathlib import Path

from experiments.run_v15_dynamic_visualization_recording import (
    generate_dynamic_walker_recording,
)
from visualization.recording import VisualizationRecordingReader


def test_walker_25_5_1_uses_bounded_audit_and_records_without_failure():
    target = Path("results/test_walker_25_5_1_disconnected_recording")
    if target.exists():
        shutil.rmtree(target)
    try:
        generate_dynamic_walker_recording(
            target, duration=2.0, dt=2.0, seed=0, include_cann=False,
            total_satellites=25, plane_count=5, phasing=1,
            enable_link_suspension=False,
            enable_absolute_navigation_dropout=False,
        )
        reader = VisualizationRecordingReader(target)
        frame = reader[0]

        assert len(frame.nodes) == 25
        assert frame.metadata["topology_audit_duration_s"] == 2.0
        assert "persistent_isolated_node_count" in frame.metadata
        assert "persistent_connected" in frame.metadata
        processed_observations = tuple(
            observation for observation in frame.observations
            if observation.nis is not None
        )
        assert processed_observations
        assert all(
            observation.processing_status != "STATUS_UNAVAILABLE"
            for observation in processed_observations
        )
        if not frame.metadata["persistent_connected"]:
            assert any(
                event.event_type == "PERSISTENT_TOPOLOGY_DISCONNECTED"
                for event in frame.events
            )
    finally:
        if target.exists():
            shutil.rmtree(target)
