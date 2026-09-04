import numpy as np
import pytest

from experiments.network_cann_fault_comparison import (
    inject_network_faulty_recovery,
    inject_network_gradual_drift,
    inject_network_impulse_faults,
)
from interfaces.data_objects import ObservationMessage


def _message(modality, observer="a", target="b", timestamp=8.0):
    dimensions = 2
    return ObservationMessage(
        message_id=f"{observer}-{target}-{modality}-{timestamp:g}",
        observer_id=observer, target_id=target, timestamp=timestamp,
        modality=modality, measurement=np.zeros(dimensions),
        covariance=np.eye(dimensions),
    )


def test_impulse_injection_is_limited_by_modality_edge_and_time():
    messages = [
        _message(modality, observer, timestamp=timestamp)
        for observer in ("a", "c")
        for modality in ("RADAR", "INFRARED", "OPTICAL")
        for timestamp in (6.0, 8.0)
    ]
    result, edges = inject_network_impulse_faults(
        messages, fault_time=8.0, fault_modalities=("RADAR",),
        affected_edge_count=1,
    )
    injected = [
        item for item in result
        if item.metadata.get("injected_network_fault", False)
    ]
    assert len(edges) == 1
    assert len(injected) == 1
    assert injected[0].modality == "RADAR"
    assert injected[0].timestamp == 8.0
    assert np.array_equal(injected[0].measurement, [3000.0, 5.0])


def test_impulse_injection_rejects_unknown_modality():
    with pytest.raises(ValueError, match="Unsupported fault modalities"):
        inject_network_impulse_faults(
            [_message("RADAR")], fault_time=8.0,
            fault_modalities=("UNKNOWN",),
        )


def test_gradual_drift_grows_after_the_selected_start_time():
    messages = [_message("OPTICAL", timestamp=value) for value in (6.0, 8.0, 10.0)]
    result, _ = inject_network_gradual_drift(
        messages, start_time=6.0, dt=2.0, step_fraction=0.1,
        fault_modalities=("OPTICAL",), affected_edge_count=1,
    )
    np.testing.assert_array_equal(result[0].measurement, [0.0, 0.0])
    np.testing.assert_allclose(result[1].measurement, [0.02, 0.04])
    np.testing.assert_allclose(result[2].measurement, [0.04, 0.08])


def test_faulty_recovery_removes_outage_and_corrupts_first_two_returns():
    messages = [
        _message("INFRARED", timestamp=value)
        for value in (6.0, 8.0, 10.0, 12.0, 14.0)
    ]
    result, _ = inject_network_faulty_recovery(
        messages, outage_start=8.0, outage_end=10.0, dt=2.0,
        fault_modalities=("INFRARED",), affected_edge_count=1,
    )
    assert [item.timestamp for item in result] == [6.0, 12.0, 14.0]
    np.testing.assert_allclose(result[1].measurement, np.deg2rad([5.0, 5.0]))
    np.testing.assert_allclose(result[2].measurement, -np.deg2rad([5.0, 5.0]))
