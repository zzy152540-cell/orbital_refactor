import numpy as np

from experiments.v15_visualization_raw_frontend import (
    replace_selected_link_with_raw_frontends,
)
from interfaces.data_objects import ObservationMessage


def _message(modality, measurement, covariance):
    information_id = f"sat_01->sat_02:{modality.lower()}:0"
    return ObservationMessage(
        message_id=information_id, physical_observation_id=information_id,
        observer_id="sat_01", target_id="sat_02", timestamp=0.0,
        modality=modality, measurement=np.asarray(measurement, dtype=float),
        covariance=np.asarray(covariance, dtype=float), frame="ECI",
        metadata={},
    )


def test_selected_link_raw_frontends_are_same_source_as_returned_messages():
    observer = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    target = np.array([7.001e6, 100.0, 20.0, 0.2, 7500.1, 0.0])
    messages = (
        _message("RADAR", [1005.0, 0.2], np.diag([4.0, 0.0025])),
        _message("INFRARED", [0.0, 0.0], np.eye(2) * 1e-6),
        _message("OPTICAL", [0.0, 0.0], np.eye(2) * 1e-6),
    )
    replaced, raw = replace_selected_link_with_raw_frontends(
        messages, timestamps=np.array([0.0]),
        truth_state_history_by_node={
            "sat_01": observer[None, :], "sat_02": target[None, :],
        }, capture_link=("sat_01", "sat_02"), seed=3,
    )

    assert {item.modality for item in replaced} == {
        "RADAR", "INFRARED", "OPTICAL",
    }
    assert set(raw) == {item.information_id for item in replaced}
    shapes = {item.modality: raw[item.information_id][0].shape for item in replaced}
    assert shapes == {
        "RADAR": (65, 65), "INFRARED": (64, 64), "OPTICAL": (64, 64),
    }
    assert all(item.metadata["visualization_raw_frontend"] for item in replaced)
