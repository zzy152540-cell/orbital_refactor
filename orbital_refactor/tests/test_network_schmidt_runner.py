import numpy as np

from cooperative.neighbor_measurement_quality import (
    NeighborLinkQuality,
    NeighborMeasurementQualityPolicy,
)
from cooperative.network_schmidt_runner import run_network_schmidt_filter
from cooperative.exact_transport_protocol import build_exact_transport_state_message
from cooperative.topology import chain_topology, fully_connected_topology
from interfaces.data_objects import (
    AbsolutePositionObservation,
    CovarianceTransportEvent,
    ObservationMessage,
)
from orbital_core.measurements import (
    measure_relative_az_el,
    measure_relative_range,
    measure_relative_range_rate,
)


def _case():
    timestamps = np.array([0.0, 1.0])
    base = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    states = {
        "sat_01": base + np.array([-1000.0, 100.0, 0.0, 0.0, 0.0, 0.0]),
        "sat_02": base.copy(),
        "sat_03": base + np.array([1200.0, -80.0, 30.0, 0.0, 0.0, 0.0]),
    }
    covariances = {node_id: np.eye(6) for node_id in states}
    observations = []
    for source, target in (("sat_01", "sat_02"), ("sat_02", "sat_03")):
        state_i, state_j = states[source], states[target]
        observations.extend(
            [
                ObservationMessage(
                    message_id=f"{source}-{target}-range",
                    observer_id=source,
                    target_id=target,
                    timestamp=0.0,
                    modality="RANGE",
                    measurement=np.array([measure_relative_range(state_i, state_j)]),
                    covariance=np.array([[1.0]]),
                ),
                ObservationMessage(
                    message_id=f"{source}-{target}-rate",
                    observer_id=source,
                    target_id=target,
                    timestamp=0.0,
                    modality="RANGE_RATE",
                    measurement=np.array([
                        measure_relative_range_rate(state_i, state_j)
                    ]),
                    covariance=np.array([[0.01]]),
                ),
                ObservationMessage(
                    message_id=f"{source}-{target}-angles",
                    observer_id=source,
                    target_id=target,
                    timestamp=0.0,
                    modality="AZ_EL",
                    measurement=measure_relative_az_el(
                        state_i, state_j, frame="RTN"
                    ),
                    covariance=np.eye(2) * 1e-4,
                    frame="RTN",
                ),
            ]
        )
    return timestamps, states, covariances, observations


def test_chain_network_builds_expected_local_schmidt_dimensions():
    timestamps, states, covariances, observations = _case()
    history = run_network_schmidt_filter(
        timestamps=timestamps,
        initial_state_by_node=states,
        initial_covariance_by_node=covariances,
        topology=chain_topology(["sat_01", "sat_02", "sat_03"]),
        observation_messages=observations,
        process_noise_acceleration=0.0,
    )

    assert history.local_dimension_by_node == {
        "sat_01": 12,
        "sat_02": 18,
        "sat_03": 12,
    }
    assert set(history.active_cross_covariance_history_by_node["sat_02"]) == {
        "sat_01",
        "sat_03",
    }
    assert set(history.neighbor_state_history_by_node["sat_02"]) == {
        "sat_01",
        "sat_03",
    }
    assert np.allclose(
        history.neighbor_state_history_by_node["sat_02"]["sat_01"][0],
        states["sat_01"],
    )
    assert history.neighbor_state_history_by_node[
        "sat_02"
    ]["sat_01"].shape == (timestamps.size, 6)
    assert len(history.nis_history_by_node["sat_01"][0]) == 3
    relative_records = history.relative_update_history_by_node[
        "sat_01"
    ][0]
    assert len(relative_records) == 3
    assert relative_records[0]["prior_active_state"].shape == (6,)
    assert relative_records[0]["prior_neighbor_state"].shape == (6,)
    assert relative_records[0]["active_correction"].shape == (6,)
    assert relative_records[0]["active_jacobian"].shape[1] == 6
    assert relative_records[0]["neighbor_jacobian"].shape[1] == 6
    assert len(history.nis_history_by_node["sat_02"][0]) == 3
    assert history.nis_history_by_node["sat_03"][0] == {}
    first_integrity = next(
        iter(history.integrity_history_by_node["sat_01"][0].values())
    )
    assert first_integrity.raw_nis is not None
    assert first_integrity.processed_nis is not None
    assert first_integrity.measurement_covariance_scale >= 1.0
    assert first_integrity.status == "ACCEPTED"
    for node_id in history.node_ids:
        assert np.min(
            np.linalg.eigvalsh(history.joint_covariance_history_by_node[node_id][-1])
        ) >= -1e-8


def test_network_runner_batches_same_directed_link_modalities():
    timestamps, states, covariances, observations = _case()
    sequential = run_network_schmidt_filter(
        timestamps=timestamps,
        initial_state_by_node=states,
        initial_covariance_by_node=covariances,
        topology=chain_topology(["sat_01", "sat_02", "sat_03"]),
        observation_messages=observations,
        process_noise_acceleration=0.0,
    )
    batched = run_network_schmidt_filter(
        timestamps=timestamps,
        initial_state_by_node=states,
        initial_covariance_by_node=covariances,
        topology=chain_topology(["sat_01", "sat_02", "sat_03"]),
        observation_messages=observations,
        process_noise_acceleration=0.0,
        batch_relative_observations=True,
    )

    assert all(
        epoch == () for values in sequential.joint_nis_history_by_node.values()
        for epoch in values
    )
    records = batched.joint_nis_history_by_node["sat_01"][0]
    assert len(records) == 1
    assert set(records[0]["modalities"]) == {
        "RANGE", "RANGE_RATE", "AZ_EL",
    }
    assert records[0]["dimension"] == 4
    assert records[0]["raw_nis"] >= 0.0
    assert np.linalg.eigvalsh(
        batched.joint_covariance_history_by_node["sat_01"][0]
    ).min() >= -1e-9


def test_network_history_converts_each_satellite_to_formal_module_output():
    timestamps, states, covariances, observations = _case()
    history = run_network_schmidt_filter(
        timestamps=timestamps,
        initial_state_by_node=states,
        initial_covariance_by_node=covariances,
        topology=chain_topology(["sat_01", "sat_02", "sat_03"]),
        observation_messages=observations,
        process_noise_acceleration=0.0,
    )

    outputs = history.to_module_outputs(processing_time=0.25)

    assert set(outputs) == {"sat_01", "sat_02", "sat_03"}
    first = outputs["sat_01"]
    assert first.module_output.state_output.target_id == "sat_01"
    assert first.module_output.state_output.position_estimate.shape == (3,)
    assert first.module_output.state_output.velocity_estimate.shape == (3,)
    assert first.module_output.state_output.acceleration_estimate.shape == (3,)
    assert first.module_output.state_output.covariance.shape == (6, 6)
    assert first.module_output.runtime_status.processing_time == 0.25
    assert first.module_output.runtime_status.observation_count == 3
    assert first.module_output.runtime_status.status == "PREDICTION_ONLY"
    assert first.network_diagnostics.neighbor_count == 1
    assert first.network_diagnostics.replay_count == 0
    assert first.network_diagnostics.configured_neighbors == ("sat_02",)
    assert first.network_diagnostics.link_health_by_neighbor == {
        "sat_02": "UNKNOWN"
    }
    assert first.network_diagnostics.last_receive_timestamp_by_neighbor == {
        "sat_02": None
    }
    assert outputs["sat_03"].module_output.runtime_status.observation_count == 0


def test_absolute_position_anchor_updates_active_state_and_delayed_replay_matches():
    timestamps = np.array([0.0, 1.0])
    first = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    second = first + np.array([1000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    states = {"sat_01": first.copy(), "sat_02": second.copy()}
    covariances = {node: np.eye(6) * 100.0 for node in states}

    def run(arrival):
        anchor = AbsolutePositionObservation(
            timestamp=0.0, satellite_id="sat_01",
            measurement_eci=first[:3] + np.array([10.0, 0.0, 0.0]),
            covariance=np.eye(3), confidence=1.0, valid_flag=True,
            observation_id="sat_01:absolute:0", arrival_timestamp=arrival,
        )
        return run_network_schmidt_filter(
            timestamps=timestamps,
            initial_state_by_node=states,
            initial_covariance_by_node=covariances,
            topology=chain_topology(["sat_01", "sat_02"]),
            observation_messages=[],
            absolute_position_observations=[anchor],
            process_noise_acceleration=0.0,
            consider_refresh_mode="exact_transport_event_replay",
            replay_history_window=2.0,
        )

    immediate = run(None)
    delayed = run(1.0)

    assert immediate.active_state_history_by_node["sat_01"][0, 0] > first[0]
    np.testing.assert_allclose(
        delayed.active_state_history_by_node["sat_01"][-1],
        immediate.active_state_history_by_node["sat_01"][-1],
    )
    assert "sat_01:absolute:0" in delayed.nis_history_by_node["sat_01"][1]
    assert delayed.modality_history_by_node["sat_01"][1][
        "sat_01:absolute:0"
    ] == "ABSOLUTE_POSITION"


def test_exact_replay_routes_delayed_shared_observation_and_deduplicates_copy():
    timestamps = np.array([0.0, 1.0])
    first = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    second = first + np.array([1000.0, 100.0, 0.0, 0.0, 0.0, 0.0])
    states = {"sat_01": first, "sat_02": second}
    covariances = {node: np.eye(6) for node in states}
    physical_id = "physical-range-0"
    observation = ObservationMessage(
        message_id="range-original", physical_observation_id=physical_id,
        observer_id="sat_01", target_id="sat_02", timestamp=0.0,
        arrival_timestamp=1.0, modality="RANGE",
        measurement=np.array([measure_relative_range(first, second)]),
        covariance=np.array([[1.0]]),
    )
    retransmission = ObservationMessage(
        message_id="range-retransmission", physical_observation_id=physical_id,
        observer_id="sat_01", target_id="sat_02", timestamp=0.0,
        arrival_timestamp=1.0, modality="RANGE",
        measurement=observation.measurement.copy(),
        covariance=observation.covariance.copy(),
    )

    history = run_network_schmidt_filter(
        timestamps=timestamps,
        initial_state_by_node=states,
        initial_covariance_by_node=covariances,
        topology=chain_topology(["sat_01", "sat_02"]),
        observation_messages=[observation, retransmission],
        observation_usage="both_endpoints",
        process_noise_acceleration=0.0,
        consider_refresh_mode="exact_transport_event_replay",
        replay_history_window=2.0,
    )

    assert set(history.nis_history_by_node["sat_01"][0]) == {physical_id}
    assert history.nis_history_by_node["sat_02"][0] == {}
    assert set(history.nis_history_by_node["sat_02"][1]) == {physical_id}
    assert (
        history.replay_performance_by_node["sat_02"].replayed_observations == 1
    )


def test_both_endpoint_routing_updates_the_target_local_filter_too():
    timestamps, states, covariances, observations = _case()
    history = run_network_schmidt_filter(
        timestamps=timestamps,
        initial_state_by_node=states,
        initial_covariance_by_node=covariances,
        topology=chain_topology(["sat_01", "sat_02", "sat_03"]),
        observation_messages=observations,
        observation_usage="both_endpoints",
        process_noise_acceleration=0.0,
    )

    assert len(history.nis_history_by_node["sat_01"][0]) == 3
    assert len(history.nis_history_by_node["sat_02"][0]) == 6
    assert len(history.nis_history_by_node["sat_03"][0]) == 3


def test_observer_only_never_routes_a_to_b_observation_to_third_party_c():
    timestamps, states, covariances, _ = _case()
    observation = ObservationMessage(
        message_id="a-to-b-only", physical_observation_id="physical-a-to-b",
        observer_id="sat_01", target_id="sat_02", timestamp=0.0,
        modality="RANGE",
        measurement=np.array([
            measure_relative_range(states["sat_01"], states["sat_02"])
        ]),
        covariance=np.array([[1.0]]),
    )

    history = run_network_schmidt_filter(
        timestamps=timestamps,
        initial_state_by_node=states,
        initial_covariance_by_node=covariances,
        topology=fully_connected_topology(list(states)),
        observation_messages=[observation],
        observation_usage="observer_only",
        process_noise_acceleration=0.0,
    )

    assert set(history.nis_history_by_node["sat_01"][0]) == {
        "physical-a-to-b"
    }
    assert history.nis_history_by_node["sat_02"][0] == {}
    assert history.nis_history_by_node["sat_03"][0] == {}


def test_fully_connected_network_uses_eighteen_dimensions_at_every_node():
    timestamps, states, covariances, _ = _case()
    history = run_network_schmidt_filter(
        timestamps=timestamps,
        initial_state_by_node=states,
        initial_covariance_by_node=covariances,
        topology=fully_connected_topology(list(states)),
        observation_messages=[],
        process_noise_acceleration=0.0,
    )

    assert history.local_dimension_by_node == {
        "sat_01": 18,
        "sat_02": 18,
        "sat_03": 18,
    }
    for node_id in history.node_ids:
        assert history.active_state_history_by_node[node_id].shape == (2, 6)
        assert history.active_covariance_history_by_node[node_id].shape == (2, 6, 6)


def test_quality_policy_uses_link_quality_at_observation_timestamp():
    timestamps, states, covariances, observations = _case()
    range_observations = [
        observation for observation in observations
        if observation.modality == "RANGE"
        and observation.observer_id == "sat_01"
    ]
    history = run_network_schmidt_filter(
        timestamps=timestamps,
        initial_state_by_node=states,
        initial_covariance_by_node=covariances,
        topology=chain_topology(["sat_01", "sat_02", "sat_03"]),
        observation_messages=range_observations,
        process_noise_acceleration=0.0,
        neighbor_measurement_quality_policy=(
            NeighborMeasurementQualityPolicy(
                base_inflation_by_modality={"RANGE": 4.0},
                loss_inflation_per_packet=3.0,
            )
        ),
        neighbor_link_quality_by_node_and_time={
            ("sat_01", "sat_02", 0.0): NeighborLinkQuality(
                consecutive_losses=2
            ),
            ("sat_01", "sat_02", 1.0): NeighborLinkQuality(
                consecutive_losses=9
            ),
        },
    )

    record = history.relative_update_history_by_node["sat_01"][0][0]
    assert record["neighbor_uncertainty_inflation"] == 10.0


def test_synchronous_neighbor_refresh_modes_preserve_joint_psd():
    timestamps, states, covariances, observations = _case()
    for mode in ("safe_rescale", "zero_cross"):
        history = run_network_schmidt_filter(
            timestamps=timestamps,
            initial_state_by_node=states,
            initial_covariance_by_node=covariances,
            topology=chain_topology(["sat_01", "sat_02", "sat_03"]),
            observation_messages=observations,
            process_noise_acceleration=0.0,
            consider_refresh_mode=mode,
        )
        for covariance_history in history.joint_covariance_history_by_node.values():
            assert min(np.linalg.eigvalsh(value).min() for value in covariance_history) >= -1e-8


def test_exact_transport_is_only_accepted_for_matching_provenance_baseline():
    timestamps, states, covariances, observations = _case()
    history = run_network_schmidt_filter(
        timestamps=timestamps,
        initial_state_by_node=states,
        initial_covariance_by_node=covariances,
        topology=chain_topology(["sat_01", "sat_02", "sat_03"]),
        observation_messages=observations,
        process_noise_acceleration=0.0,
        consider_refresh_mode="exact_if_compatible",
    )
    assert sum(history.refresh_diagnostics.values()) == 4
    assert history.refresh_diagnostics["accepted"] < 4


def test_formal_network_mode_replays_neighbor_state_messages():
    timestamps, states, covariances, observations = _case()
    transition = np.eye(6) * 0.9
    noise = np.eye(6) * 0.1
    updated = states["sat_01"] + np.array([2.0, -1.0, 0, 0, 0, 0])
    event = CovarianceTransportEvent(
        timestamp=0.0, state_estimate=updated,
        error_transition=transition, independent_process_noise=noise,
        information_ids=("sat01-private-update",),
    )
    message = build_exact_transport_state_message(
        source_node_id="sat_01", timestamp=0.0, reference_timestamp=0.0,
        reference_state=states["sat_01"],
        reference_covariance=covariances["sat_01"],
        updated_state=updated, error_transition=transition,
        independent_process_noise=noise, lineage_id="sat01:0",
        transport_events=(event,),
    )
    history = run_network_schmidt_filter(
        timestamps=timestamps, initial_state_by_node=states,
        initial_covariance_by_node=covariances,
        topology=chain_topology(["sat_01", "sat_02", "sat_03"]),
        observation_messages=observations,
        consider_refresh_mode="exact_transport_event_replay",
        state_messages_by_receiver={"sat_02": [message]},
        process_noise_acceleration=0.0,
    )
    assert history.refresh_diagnostics["accepted"] == 1
    assert history.replay_performance_by_node["sat_02"].replay_count == 1
    assert history.replay_performance_by_node["sat_02"].maximum_batch_size == 1
    assert np.linalg.eigvalsh(
        history.joint_covariance_history_by_node["sat_02"][-1]
    ).min() >= -1e-8


def test_delayed_state_message_is_rejected_after_topology_version_changes():
    timestamps, states, covariances, _ = _case()
    transition = np.eye(6)
    event = CovarianceTransportEvent(
        timestamp=0.0, state_estimate=states["sat_01"],
        error_transition=transition,
        independent_process_noise=np.zeros((6, 6)),
    )
    message = build_exact_transport_state_message(
        source_node_id="sat_01", timestamp=0.0, reference_timestamp=0.0,
        reference_state=states["sat_01"],
        reference_covariance=covariances["sat_01"],
        updated_state=states["sat_01"], error_transition=transition,
        independent_process_noise=np.zeros((6, 6)),
        lineage_id="sat01:0", transport_events=(event,),
    )
    message.arrival_timestamp = 1.0
    message.metadata = {"topology_version": 0}

    history = run_network_schmidt_filter(
        timestamps=timestamps, initial_state_by_node=states,
        initial_covariance_by_node=covariances,
        topology=chain_topology(["sat_01", "sat_02", "sat_03"]),
        observation_messages=[],
        consider_refresh_mode="exact_transport_event_replay",
        state_messages_by_receiver={"sat_02": [message]},
        topology_version_by_timestamp={0.0: 0, 1.0: 1},
        active_neighbors_by_timestamp={
            0.0: {
                "sat_01": ("sat_02",),
                "sat_02": ("sat_01", "sat_03"),
                "sat_03": ("sat_02",),
            },
            1.0: {
                "sat_01": ("sat_02",),
                "sat_02": ("sat_01", "sat_03"),
                "sat_03": ("sat_02",),
            },
        },
        process_noise_acceleration=0.0,
    )

    assert history.refresh_diagnostics["topology_version_mismatch"] == 1
    assert history.replay_performance_by_node["sat_02"].replay_count == 0
