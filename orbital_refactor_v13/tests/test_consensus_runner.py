import numpy as np

from cooperative.communication_channel import CommunicationChannel
from cooperative.consensus_runner import run_distributed_consensus_history
from cooperative.delay_channel import DelayChannel
from cooperative.topology import chain_topology
from interfaces.data_objects import InterSatelliteObservation
from orbital_core.measurements import measure_relative_range, measure_relative_range_rate


def _histories():
    timestamps = np.array([0.0, 1.0, 2.0])
    radius = 7.0e6
    states = {
        "sat_01": np.array([
            [radius, 0.0, 0.0, 0.0, 7500.0, 0.0],
            [radius, 7500.0, 0.0, -8.0, 7500.0, 0.0],
            [radius - 16.0, 15000.0, 0.0, -16.0, 7500.0, 0.0],
        ]),
        "sat_02": np.array([
            [radius + 10.0, 0.0, 0.0, 0.0, 7500.0, 0.0],
            [radius + 10.0, 7500.0, 0.0, -8.0, 7500.0, 0.0],
            [radius - 6.0, 15000.0, 0.0, -16.0, 7500.0, 0.0],
        ]),
        "sat_03": np.array([
            [radius + 20.0, 0.0, 0.0, 0.0, 7500.0, 0.0],
            [radius + 20.0, 7500.0, 0.0, -8.0, 7500.0, 0.0],
            [radius + 4.0, 15000.0, 0.0, -16.0, 7500.0, 0.0],
        ]),
    }
    covariances = {
        "sat_01": np.tile(np.eye(6) * 9.0, (3, 1, 1)),
        "sat_02": np.tile(np.eye(6) * 4.0, (3, 1, 1)),
        "sat_03": np.tile(np.eye(6) * 16.0, (3, 1, 1)),
    }
    return timestamps, states, covariances


def test_distributed_consensus_history_keeps_per_node_outputs():
    timestamps, states, covariances = _histories()
    result = run_distributed_consensus_history(
        timestamps=timestamps,
        state_history_by_node=states,
        covariance_history_by_node=covariances,
        topology=chain_topology(["sat_01", "sat_02", "sat_03"]),
        grid_points=11,
    )

    assert set(result.node_ids) == {"sat_01", "sat_02", "sat_03"}
    assert not hasattr(result, "global_state")
    for node_id in result.node_ids:
        assert result.state_history_by_node[node_id].shape == (3, 6)
        assert result.covariance_history_by_node[node_id].shape == (3, 6, 6)
        assert len(result.node_weight_history_by_node[node_id]) == 3
        assert len(result.iteration_weight_history_by_node[node_id]) == 3
        assert len(result.received_reports_by_node[node_id]) == 3
    assert result.received_reports_by_node["sat_01"][0] == ["sat_02"]
    assert result.received_reports_by_node["sat_02"][0] == ["sat_01", "sat_03"]
    assert result.received_reports_by_node["sat_03"][0] == ["sat_02"]
    assert result.communication_stats.attempted_report_count == 12
    assert result.communication_stats.received_report_count == 12
    assert result.communication_stats.pending_report_count == 0
    assert result.communication_stats.packet_loss_rate == 0.0


def test_distributed_consensus_history_supports_multiple_consensus_iterations():
    timestamps, states, covariances = _histories()
    result = run_distributed_consensus_history(
        timestamps=timestamps,
        state_history_by_node=states,
        covariance_history_by_node=covariances,
        topology=chain_topology(["sat_01", "sat_02", "sat_03"]),
        consensus_iterations=2,
        grid_points=11,
    )

    for node_id in result.node_ids:
        assert len(result.iteration_weight_history_by_node[node_id][0]) == 2
        assert (
            result.node_weight_history_by_node[node_id][0]
            == result.iteration_weight_history_by_node[node_id][0][-1]
        )


def test_distributed_consensus_history_rejects_zero_iterations():
    timestamps, states, covariances = _histories()
    try:
        run_distributed_consensus_history(
            timestamps=timestamps,
            state_history_by_node=states,
            covariance_history_by_node=covariances,
            topology=chain_topology(["sat_01", "sat_02", "sat_03"]),
            consensus_iterations=0,
        )
    except ValueError as exc:
        assert "consensus_iterations" in str(exc)
    else:
        raise AssertionError("Expected zero consensus iterations to be rejected.")


def test_distributed_consensus_history_buffers_delayed_reports():
    timestamps, states, covariances = _histories()
    result = run_distributed_consensus_history(
        timestamps=timestamps,
        state_history_by_node={
            "sat_01": states["sat_01"],
            "sat_02": states["sat_02"],
        },
        covariance_history_by_node={
            "sat_01": covariances["sat_01"],
            "sat_02": covariances["sat_02"],
        },
        topology=chain_topology(["sat_01", "sat_02"]),
        delay_channel=DelayChannel(delay_by_node={"sat_01": 1.0, "sat_02": 1.0}),
        grid_points=11,
    )

    assert result.received_reports_by_node["sat_01"][0] == []
    assert result.received_reports_by_node["sat_02"][0] == []
    assert result.received_reports_by_node["sat_01"][1] == ["sat_02"]
    assert result.received_reports_by_node["sat_02"][1] == ["sat_01"]
    assert result.communication_stats.attempted_report_count == 6
    assert result.communication_stats.received_report_count == 4
    assert result.communication_stats.pending_report_count == 2
    assert result.communication_stats.average_delay == 1.0


def test_distributed_consensus_history_tracks_packet_loss_statistics():
    timestamps, states, covariances = _histories()
    result = run_distributed_consensus_history(
        timestamps=timestamps,
        state_history_by_node={
            "sat_01": states["sat_01"],
            "sat_02": states["sat_02"],
        },
        covariance_history_by_node={
            "sat_01": covariances["sat_01"],
            "sat_02": covariances["sat_02"],
        },
        topology=chain_topology(["sat_01", "sat_02"]),
        communication_channel=CommunicationChannel(
            packet_loss_rate={"sat_01": 1.0, "sat_02": 0.0},
            random_seed=1,
        ),
        grid_points=11,
    )

    assert result.communication_stats.attempted_report_count == 6
    assert result.communication_stats.received_report_count == 3
    assert result.communication_stats.dropped_report_count == 3
    assert result.communication_stats.pending_report_count == 0
    assert result.communication_stats.packet_loss_rate == 0.5
    assert result.received_reports_by_node["sat_02"] == [[], [], []]


def test_distributed_consensus_history_applies_optional_range_updates():
    timestamps, states, covariances = _histories()
    state_subset = {
        "sat_01": states["sat_01"],
        "sat_02": states["sat_02"],
    }
    covariance_subset = {
        "sat_01": covariances["sat_01"],
        "sat_02": covariances["sat_02"],
    }
    ranges = {
        "sat_01": {
            "sat_02": np.array([
                measure_relative_range(state_subset["sat_01"][i], state_subset["sat_02"][i])
                for i in range(len(timestamps))
            ])
        },
        "sat_02": {
            "sat_01": np.array([
                measure_relative_range(state_subset["sat_02"][i], state_subset["sat_01"][i])
                for i in range(len(timestamps))
            ])
        },
    }

    result = run_distributed_consensus_history(
        timestamps=timestamps,
        state_history_by_node=state_subset,
        covariance_history_by_node=covariance_subset,
        topology=chain_topology(["sat_01", "sat_02"]),
        range_measurements_by_node=ranges,
        range_variance=1.0,
        grid_points=11,
    )

    assert len(result.range_nis_history_by_node["sat_01"]) == 3
    assert set(result.range_nis_history_by_node["sat_01"][0]) == {"sat_02"}
    assert result.range_nis_history_by_node["sat_01"][0]["sat_02"] >= 0.0
    assert result.covariance_history_by_node["sat_01"][0, 0, 0] < covariances["sat_01"][0, 0, 0]


def test_distributed_consensus_history_accepts_inter_satellite_observation_objects():
    timestamps, states, covariances = _histories()
    state_subset = {
        "sat_01": states["sat_01"],
        "sat_02": states["sat_02"],
    }
    covariance_subset = {
        "sat_01": covariances["sat_01"],
        "sat_02": covariances["sat_02"],
    }
    observations = []
    for index, timestamp in enumerate(timestamps):
        observations.append(
            InterSatelliteObservation(
                timestamp=float(timestamp),
                source_node_id="sat_01",
                target_node_id="sat_02",
                modality="RANGE",
                measurement=np.array([
                    measure_relative_range(
                        state_subset["sat_01"][index],
                        state_subset["sat_02"][index],
                    )
                ]),
                covariance=np.array([[1.0]]),
                confidence=1.0,
                valid_flag=True,
            )
        )
        observations.append(
            InterSatelliteObservation(
                timestamp=float(timestamp),
                source_node_id="sat_01",
                target_node_id="sat_02",
                modality="RANGE_RATE",
                measurement=np.array([
                    measure_relative_range_rate(
                        state_subset["sat_01"][index],
                        state_subset["sat_02"][index],
                    )
                ]),
                covariance=np.array([[0.01]]),
                confidence=1.0,
                valid_flag=True,
            )
        )

    result = run_distributed_consensus_history(
        timestamps=timestamps,
        state_history_by_node=state_subset,
        covariance_history_by_node=covariance_subset,
        topology=chain_topology(["sat_01", "sat_02"]),
        inter_satellite_observations=observations,
        grid_points=11,
    )

    assert set(result.range_nis_history_by_node["sat_01"][0]) == {"sat_02"}
    assert set(result.inter_satellite_nis_history_by_node["sat_01"][0]) == {
        "sat_02:BLOCK",
        "sat_02:RANGE",
        "sat_02:RANGE_RATE",
    }
    assert result.range_nis_history_by_node["sat_02"][0] == {}


def test_distributed_consensus_history_records_inter_satellite_gates():
    timestamps, states, covariances = _histories()
    state_subset = {
        "sat_01": states["sat_01"],
        "sat_02": states["sat_02"],
    }
    covariance_subset = {
        "sat_01": covariances["sat_01"],
        "sat_02": covariances["sat_02"],
    }
    observations = [
        InterSatelliteObservation(
            timestamp=float(timestamp),
            source_node_id="sat_01",
            target_node_id="sat_02",
            modality="RANGE",
            measurement=np.array([1.0e6]),
            covariance=np.array([[1.0]]),
            confidence=1.0,
            valid_flag=True,
        )
        for timestamp in timestamps
    ]

    result = run_distributed_consensus_history(
        timestamps=timestamps,
        state_history_by_node=state_subset,
        covariance_history_by_node=covariance_subset,
        topology=chain_topology(["sat_01", "sat_02"]),
        inter_satellite_observations=observations,
        inter_satellite_gate_enable=True,
        inter_satellite_gate_threshold=1.0,
        inter_satellite_gate_mode="hard",
        grid_points=11,
    )

    assert result.inter_satellite_gate_history_by_node["sat_01"][0]["sat_02:BLOCK"]
    assert result.inter_satellite_gate_history_by_node["sat_01"][0]["sat_02:RANGE"]
