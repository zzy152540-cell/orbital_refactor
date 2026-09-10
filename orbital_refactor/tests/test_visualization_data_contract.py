import json

import numpy as np
import pytest

from visualization.data_contract import (
    VisualCANNSnapshot,
    VisualEdge,
    VisualNavigationState,
    VisualNodeState,
    VisualObservation,
    VisualizationFrame,
    visualization_frame_from_dict,
    visualization_frame_to_dict,
)


def _node(node_id, offset=0.0):
    return VisualNodeState(
        node_id=node_id,
        truth_state=np.arange(6.0) + offset,
        estimate_state=np.arange(6.0) + offset + 0.1,
        covariance_diagonal=np.full(6, 0.25),
    )


def test_visualization_frame_collects_synchronized_multimodal_state():
    frame = VisualizationFrame(
        scenario_id="walker20",
        run_id="seed-0",
        timestamp=2.0,
        epoch_index=1,
        nodes=(_node("sat_01"), _node("sat_02", 1.0)),
        edges=(VisualEdge(
            "sat_01", "sat_02", "MEASUREMENT", modality="OPTICAL",
            metrics={"delay_s": 0.5},
        ),),
        observations=(VisualObservation(
            observer_id="sat_01", target_id="sat_02", modality="OPTICAL",
            measurement=np.array([0.1, -0.2]),
            covariance_diagonal=np.array([1e-4, 1e-4]), visible=True,
            valid=True, processing_status="ACCEPTED", frame="CAMERA",
            nis=1.2, raw_sensor_data=np.ones((8, 8)),
            raw_data_kind="GRAYSCALE_IMAGE",
        ),),
        navigation=(VisualNavigationState(
            node_id="sat_01", absolute_navigation_available=True,
            last_absolute_navigation_timestamp=2.0, anchor_age=0.0,
            orbital_phase=0.2, radial_along_track=np.array([1.0, 2.0]),
        ),),
        cann=(VisualCANNSnapshot(
            node_id="sat_01", representation="DIRECTION_RING",
            available=True, activity=np.linspace(0.0, 1.0, 16),
            decoded_value=np.array([0.2]), concentration=0.9,
        ),),
    )

    assert frame.schema_version == "v1.0"
    assert frame.observations[0].raw_sensor_data.shape == (8, 8)
    assert frame.cann[0].representation == "DIRECTION_RING"

    portable = visualization_frame_to_dict(frame)
    restored = visualization_frame_from_dict(json.loads(json.dumps(portable)))
    assert restored.timestamp == frame.timestamp
    assert restored.edges[0].metrics["delay_s"] == 0.5
    np.testing.assert_array_equal(
        restored.observations[0].raw_sensor_data,
        frame.observations[0].raw_sensor_data,
    )
    assert not restored.observations[0].raw_sensor_data.flags.writeable


def test_visualization_arrays_are_defensive_copies_and_read_only():
    estimate = np.arange(6.0)
    node = VisualNodeState(
        node_id="sat_01", estimate_state=estimate,
        covariance_diagonal=np.ones(6),
    )
    estimate[0] = 99.0

    assert node.estimate_state[0] == 0.0
    with pytest.raises(ValueError):
        node.estimate_state[0] = 1.0


def test_full_covariance_must_match_the_display_diagonal():
    with pytest.raises(ValueError, match="inconsistent"):
        VisualNodeState(
            node_id="sat_01", estimate_state=np.arange(6.0),
            covariance_diagonal=np.ones(6), covariance=np.eye(6) * 2.0,
        )


def test_unavailable_cann_cannot_expose_stale_activity():
    with pytest.raises(ValueError, match="Unavailable CANN"):
        VisualCANNSnapshot(
            node_id="sat_01", representation="DIRECTION_RING",
            available=False, activity=np.ones(8),
        )


def test_frame_rejects_cross_epoch_unknown_nodes():
    with pytest.raises(ValueError, match="unknown node"):
        VisualizationFrame(
            scenario_id="walker5", run_id="seed-0", timestamp=0.0,
            epoch_index=0, nodes=(_node("sat_01"),),
            edges=(VisualEdge("sat_01", "sat_02", "COMMUNICATION"),),
        )
