import numpy as np

from cooperative.network_schmidt_runner import run_network_schmidt_filter
from cooperative.topology import fully_connected_topology
from experiments.relative_measurement_error_projection import (
    relative_error_projection_diagnostics,
    relative_update_truth_decomposition,
    summarize_relative_update_truth_diagnostics,
)
from interfaces.data_objects import ObservationMessage
from orbital_core.measurements import measure_relative_range


def test_projection_uses_stored_epoch_means_and_keeps_filter_read_only():
    times = np.array([0.0])
    truth = {
        "sat_01": np.array([7e6, 0, 0, 0, 7500, 0], dtype=float)[None, :],
        "sat_02": np.array([7e6 + 1000, 0, 0, 0, 7500, 0], dtype=float)[None, :],
    }
    initial = {
        "sat_01": truth["sat_01"][0] + np.array([10, 0, 0, 0, 0, 0]),
        "sat_02": truth["sat_02"][0] + np.array([30, 0, 0, 0, 0, 0]),
    }
    observation = ObservationMessage(
        message_id="range",
        observer_id="sat_01",
        target_id="sat_02",
        timestamp=0.0,
        modality="RANGE",
        measurement=np.array([
            measure_relative_range(truth["sat_01"][0], truth["sat_02"][0])
        ]),
        covariance=np.array([[4.0]]),
    )
    history = run_network_schmidt_filter(
        timestamps=times,
        initial_state_by_node=initial,
        initial_covariance_by_node={
            node: np.eye(6) for node in initial
        },
        topology=fully_connected_topology(tuple(initial)),
        observation_messages=(),
    )

    records = relative_error_projection_diagnostics(
        history=history, truth_by_node=truth, observations=(observation,)
    )

    assert len(records) == 1
    assert np.allclose(records[0].active_projection, [-10.0])
    assert np.allclose(records[0].neighbor_projection, [30.0])
    assert np.allclose(records[0].total_linearized_projection, [20.0])


def test_preupdate_truth_decomposition_detects_neighbor_error_transfer():
    times = np.array([0.0])
    truth = {
        "sat_01": np.array([7e6, 0, 0, 0, 7500, 0], dtype=float)[None, :],
        "sat_02": np.array([7e6 + 1000, 0, 0, 0, 7500, 0], dtype=float)[None, :],
    }
    initial = {
        "sat_01": truth["sat_01"][0] + np.array([10, 0, 0, 0, 0, 0]),
        "sat_02": truth["sat_02"][0] + np.array([30, 0, 0, 0, 0, 0]),
    }
    observation = ObservationMessage(
        message_id="range",
        observer_id="sat_01",
        target_id="sat_02",
        timestamp=0.0,
        modality="RANGE",
        measurement=np.array([1000.0]),
        covariance=np.array([[4.0]]),
    )
    history = run_network_schmidt_filter(
        timestamps=times,
        initial_state_by_node=initial,
        initial_covariance_by_node={
            node: np.eye(6) * 100.0 for node in initial
        },
        topology=fully_connected_topology(tuple(initial)),
        observation_messages=(observation,),
    )

    records = relative_update_truth_decomposition(
        history=history, truth_by_node=truth
    )

    assert len(records) == 1
    assert np.allclose(records[0].active_projection, [-10.0])
    assert np.allclose(records[0].neighbor_projection, [30.0])
    assert np.allclose(records[0].innovation, [-20.0])
    assert np.allclose(records[0].unexplained_innovation, [0.0])
    assert records[0].active_error_norm_change > 0.0
    assert np.isclose(records[0].active_error_norm_after, 19.8039215686)
    summary = summarize_relative_update_truth_diagnostics(records)[0]
    assert summary.modality == "RANGE"
    assert summary.sample_count == 1
    assert summary.active_error_worsening_fraction == 1.0
    assert summary.mean_velocity_injection_risk >= 0.0
    assert summary.mean_position_injection_risk >= 0.0
    assert np.isfinite(
        summary.mean_neighbor_to_measurement_covariance_trace_ratio
    )
