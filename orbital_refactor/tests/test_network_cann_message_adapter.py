import numpy as np
import pytest

from experiments.network_cann_message_adapter import (
    preprocess_network_observation_messages,
)
from interfaces.data_objects import ObservationMessage
from cooperative.network_schmidt_runner import run_network_schmidt_filter
from cooperative.topology import chain_topology
from experiments.inter_satellite_observation_factory import (
    target_pointing_quaternion,
)
from orbital_core.measurements import (
    measure_relative_az_el,
    measure_relative_optical_uv,
    measure_relative_range,
    measure_relative_range_rate,
)


def _message(message_id, observer, target, timestamp):
    return ObservationMessage(
        message_id=message_id, physical_observation_id=f"physical:{message_id}",
        observer_id=observer, target_id=target, timestamp=timestamp,
        modality="RADAR", measurement=np.array([1000.0 + timestamp, 1.0]),
        covariance=np.diag([30.0, 0.05]) ** 2, frame="ECI",
        source_timestamp=timestamp, arrival_timestamp=timestamp + 2.0,
        metadata={"measurement_type": "RANGE_RANGE_RATE"},
    )


def test_network_cann_preserves_message_identity_and_does_not_fill_gaps():
    messages = [
        _message("a-b-0", "a", "b", 0.0),
        _message("a-b-4", "a", "b", 4.0),
        _message("a-c-0", "a", "c", 0.0),
        _message("a-c-4", "a", "c", 4.0),
    ]
    output = preprocess_network_observation_messages(
        messages, timestamps=np.array([0.0, 2.0, 4.0]),
    )
    assert len(output) == len(messages)
    by_id = {item.message_id: item for item in output}
    for source in messages:
        result = by_id[source.message_id]
        assert result.physical_observation_id == source.physical_observation_id
        assert result.source_timestamp == source.source_timestamp
        assert result.arrival_timestamp == source.arrival_timestamp
        assert result.observer_id == source.observer_id
        assert result.target_id == source.target_id
        assert result.metadata["cann_preprocessed"]


def test_unsupported_network_modality_passes_through_unchanged():
    message = _message("range-0", "a", "b", 0.0)
    message.modality = "RANGE"
    output = preprocess_network_observation_messages(
        [message], timestamps=np.array([0.0, 2.0]),
    )
    assert output == [message]


def test_preprocessed_message_is_idempotent():
    source = [_message("a-b-0", "a", "b", 0.0)]
    first = preprocess_network_observation_messages(
        source, timestamps=np.array([0.0, 2.0]),
    )
    second = preprocess_network_observation_messages(
        first, timestamps=np.array([0.0, 2.0]),
    )
    assert second == first


def test_network_cann_accepts_a_message_generator():
    source = [_message("a-b-0", "a", "b", 0.0)]
    output = preprocess_network_observation_messages(
        (message for message in source), timestamps=np.array([0.0, 2.0]),
    )
    assert [message.message_id for message in output] == ["a-b-0"]


def test_network_cann_rejects_ambiguous_duplicate_modal_samples():
    messages = [
        _message("a-b-first", "a", "b", 0.0),
        _message("a-b-second", "a", "b", 0.0),
    ]
    with pytest.raises(ValueError, match="at most one observation"):
        preprocess_network_observation_messages(
            messages, timestamps=np.array([0.0, 2.0]),
        )


def test_cann_messages_run_through_network_schmidt():
    timestamps = np.array([0.0, 2.0, 4.0])
    base = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    states = {
        "a": base.copy(),
        "b": base + np.array([1000.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    }
    messages = [
        _message("a-b-0", "a", "b", 0.0),
        _message("a-b-4", "a", "b", 4.0),
    ]
    history = run_network_schmidt_filter(
        timestamps=timestamps,
        initial_state_by_node=states,
        initial_covariance_by_node={name: np.eye(6) for name in states},
        topology=chain_topology(["a", "b"]),
        observation_messages=messages,
        observation_message_preprocessor=preprocess_network_observation_messages,
        process_noise_acceleration=0.0,
    )
    assert np.isfinite(history.active_state_history_by_node["a"]).all()
    assert set(history.nis_history_by_node["a"][0]) == {
        "physical:a-b-0",
    }


def test_three_modal_directed_edges_run_through_network_schmidt():
    timestamps = np.array([0.0, 2.0, 4.0])
    states = {
        "a": np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0]),
        "b": np.array([7.001e6, 100.0, 20.0, 0.1, 7500.0, 0.0]),
        "c": np.array([7.002e6, -50.0, 40.0, 0.0, 7500.1, 0.0]),
    }
    messages = []
    for observer, target in (("a", "b"), ("b", "c")):
        state_i, state_j = states[observer], states[target]
        quaternion = target_pointing_quaternion(state_i, state_j)
        measurements = {
            "RADAR": (
                np.array([
                    measure_relative_range(state_i, state_j),
                    measure_relative_range_rate(state_i, state_j),
                ]),
                np.diag([30.0, 0.05]) ** 2,
                "ECI",
                {"measurement_type": "RANGE_RANGE_RATE"},
            ),
            "INFRARED": (
                measure_relative_az_el(state_i, state_j, frame="ECI"),
                np.eye(2) * 1e-6,
                "ECI",
                {"measurement_type": "AZIMUTH_ELEVATION"},
            ),
            "OPTICAL": (
                measure_relative_optical_uv(
                    state_i, state_j, frame="BODY",
                    quaternion_i2b_wxyz=quaternion,
                ),
                np.eye(2) * 1e-6,
                "BODY",
                {
                    "measurement_type": "NORMALIZED_IMAGE_COORDINATES",
                    "quaternion_i2b_wxyz": quaternion,
                },
            ),
        }
        for timestamp in (0.0, 4.0):
            for modality, (measurement, covariance, frame, metadata) in (
                measurements.items()
            ):
                information_id = (
                    f"{observer}-{target}-{modality.lower()}-{timestamp:g}"
                )
                messages.append(ObservationMessage(
                    message_id=information_id,
                    physical_observation_id=f"physical:{information_id}",
                    observer_id=observer,
                    target_id=target,
                    timestamp=timestamp,
                    modality=modality,
                    measurement=measurement.copy(),
                    covariance=covariance.copy(),
                    frame=frame,
                    source_timestamp=timestamp,
                    arrival_timestamp=timestamp,
                    metadata=dict(metadata),
                ))

    history = run_network_schmidt_filter(
        timestamps=timestamps,
        initial_state_by_node=states,
        initial_covariance_by_node={name: np.eye(6) for name in states},
        topology=chain_topology(["a", "b", "c"]),
        observation_messages=messages,
        observation_message_preprocessor=preprocess_network_observation_messages,
        process_noise_acceleration=0.0,
    )

    assert np.isfinite(history.active_state_history_by_node["a"]).all()
    assert np.isfinite(history.active_state_history_by_node["b"]).all()
    assert len(history.nis_history_by_node["a"][0]) == 3
    assert len(history.nis_history_by_node["b"][0]) == 3
    assert set(history.nis_history_by_node["a"][0]) == {
        "physical:a-b-radar-0",
        "physical:a-b-infrared-0",
        "physical:a-b-optical-0",
    }
