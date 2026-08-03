from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cooperative.exact_transport_accumulator import ExactTransportAccumulator
from cooperative.message_transport import MessageChannel
from cooperative.network_schmidt_runner import run_network_schmidt_filter
from cooperative.topology import chain_topology
from interfaces.data_objects import ObservationMessage, StateMessage
from orbital_core.constants import R_EARTH
from orbital_core.dynamics import make_process_noise, numerical_jacobian_discrete, rk4_step_absolute
from orbital_core.measurements import measure_relative_range
from orbital_core.metrics import compute_nees_history, compute_rmse
from orbital_core.orbit_elements import keplerian_to_eci
from scenarios.fleet_scenario import generate_fleet_scenario

Array = np.ndarray
NEES_95_DOF6 = (1.2373442458, 14.4493753354)
NIS_95_DOF1 = (0.0009820691, 5.0238861873)


@dataclass(frozen=True)
class ExactTransportScanSummary:
    scenario: str
    mode: str
    run_count: int
    mean_position_rmse: float
    mean_velocity_rmse: float
    mean_nees: float
    mean_nees_95_coverage: float
    mean_nis: float
    mean_nis_95_coverage: float
    message_acceptance_rate: float
    message_rejection_count: int
    psd_failure_count: int
    minimum_joint_eigenvalue: float
    transmitted_state_messages: int
    rejection_counts: dict[str, int]


@dataclass(frozen=True)
class ExactTransportScaleScanResult:
    summary_by_scenario_and_mode: dict[tuple[str, str], ExactTransportScanSummary]


def run_v14_exact_transport_smoke_scan(
    *, seeds: int = 20, duration: float = 60.0, dt: float = 2.0,
    range_sigma: float = 2.0, absolute_sigma: float = 3.0,
    process_noise_acceleration: float = 1e-8,
) -> ExactTransportScaleScanResult:
    """Run the production network API over five three-satellite communication cases."""
    if seeds < 1:
        raise ValueError("seeds must be at least one.")
    scenarios = {
        "ideal": (0.0, 0.0, 10.0),
        "loss_20_percent": (0.2, 0.0, 10.0),
        "delay_one_epoch": (0.0, dt, 10.0),
        "delay_loss": (0.2, dt, 10.0),
        "insufficient_history": (0.0, 3.0 * dt, 2.0 * dt),
    }
    modes = ("propagate_only", "exact_transport_event_replay")
    collected = {(scenario, mode): [] for scenario in scenarios for mode in modes}
    for seed in range(seeds):
        for scenario, (loss, delay, history_window) in scenarios.items():
            case = _build_case(
                seed=seed, duration=duration, dt=dt, range_sigma=range_sigma,
                absolute_sigma=absolute_sigma,
                process_noise_acceleration=process_noise_acceleration,
                packet_loss=loss, delay=delay,
                acknowledge_messages=delay <= history_window,
            )
            for mode in modes:
                history = run_network_schmidt_filter(
                    timestamps=case["timestamps"],
                    initial_state_by_node=case["initial_states"],
                    initial_covariance_by_node=case["initial_covariances"],
                    topology=case["topology"],
                    observation_messages=case["observations"],
                    observation_usage="observer_only",
                    process_noise_acceleration=process_noise_acceleration,
                    consider_refresh_mode=mode,
                    state_messages_by_receiver=(case["state_messages"] if mode == "exact_transport_event_replay" else None),
                    replay_history_window=(history_window if mode == "exact_transport_event_replay" else None),
                    expected_lineage_by_link=(case["lineages"] if mode == "exact_transport_event_replay" else None),
                )
                collected[(scenario, mode)].append(
                    _metrics(history, case["truth"], len(case["transmitted_messages"]))
                )
    summaries = {}
    for key, values in collected.items():
        scenario, mode = key
        attempted = sum(value[8] for value in values)
        accepted = sum(value[7] for value in values)
        summaries[key] = ExactTransportScanSummary(
            scenario=scenario, mode=mode, run_count=len(values),
            mean_position_rmse=float(np.mean([value[0] for value in values])),
            mean_velocity_rmse=float(np.mean([value[1] for value in values])),
            mean_nees=float(np.mean([value[2] for value in values])),
            mean_nees_95_coverage=float(np.mean([value[3] for value in values])),
            mean_nis=float(np.mean([value[4] for value in values])),
            mean_nis_95_coverage=float(np.mean([value[5] for value in values])),
            message_acceptance_rate=(accepted / attempted if attempted else 0.0),
            message_rejection_count=sum(value[9] for value in values),
            psd_failure_count=sum(value[10] for value in values),
            minimum_joint_eigenvalue=min(value[6] for value in values),
            transmitted_state_messages=attempted,
            rejection_counts=_sum_rejection_counts([value[11] for value in values]),
        )
    return ExactTransportScaleScanResult(summaries)


def _build_case(*, seed, duration, dt, range_sigma, absolute_sigma,
                process_noise_acceleration, packet_loss, delay, acknowledge_messages):
    rng = np.random.default_rng(20260830 + seed)
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt)
    base = keplerian_to_eci(R_EARTH + 700e3, 0.001, np.deg2rad(23.0), 0.0, 0.0, 0.0)
    truth_initials = {
        "sat_01": base + np.array([-1200, 100, 20, 0, -0.02, 0.0]),
        "sat_02": base.copy(),
        "sat_03": base + np.array([1300, -80, 30, 0, 0.03, 0.0]),
    }
    scenario = generate_fleet_scenario(timestamps=timestamps, initial_state_by_node=truth_initials)
    truth = scenario.truth_state_history_by_node
    covariance = np.diag([10, 10, 10, 0.02, 0.02, 0.02]) ** 2
    initial_states = {node: truth_initials[node] + rng.multivariate_normal(np.zeros(6), covariance) for node in scenario.node_ids}
    initial_covariances = {node: covariance.copy() for node in scenario.node_ids}
    topology = chain_topology(list(scenario.node_ids))
    edges = tuple((receiver, source) for receiver in topology.node_ids for source in topology.neighbors(receiver))
    sender_state = {node: value.copy() for node, value in initial_states.items()}
    sender_covariance = {node: covariance.copy() for node in scenario.node_ids}
    accumulators = {
        edge: ExactTransportAccumulator(
            source_node_id=edge[1], lineage_id=f"{edge[1]}->{edge[0]}:0",
            reference_timestamp=0.0, reference_state=sender_state[edge[1]],
            reference_covariance=sender_covariance[edge[1]],
        ) for edge in edges
    }
    channels = {
        edge: MessageChannel(
            packet_loss_rate={edge[1]: packet_loss}, delay_by_source={edge[1]: delay},
            random_seed=seed * 101 + index,
        ) for index, edge in enumerate(edges)
    }
    state_messages = {node: [] for node in topology.node_ids}
    transmitted_messages: list[StateMessage] = []
    pending_acks = []
    observations = []
    h = np.zeros((3, 6)); h[:, :3] = np.eye(3)
    absolute_covariance = np.eye(3) * absolute_sigma**2
    for index, timestamp in enumerate(timestamps):
        for arrival, edge, message in sorted(pending_acks, key=lambda item: item[0]):
            if arrival <= timestamp and acknowledge_messages:
                accumulators[edge].acknowledge(message)
        pending_acks = [item for item in pending_acks if item[0] > timestamp or not acknowledge_messages]
        prediction_transition = {node: np.eye(6) for node in topology.node_ids}
        prediction_noise = {node: np.zeros((6, 6)) for node in topology.node_ids}
        if index > 0:
            for node in topology.node_ids:
                transition = numerical_jacobian_discrete(lambda value: rk4_step_absolute(value, dt), sender_state[node])
                sender_state[node] = rk4_step_absolute(sender_state[node], dt)
                noise = make_process_noise(dt, process_noise_acceleration)
                sender_covariance[node] = transition @ sender_covariance[node] @ transition.T + noise
                prediction_transition[node] = transition; prediction_noise[node] = noise
        for node in topology.node_ids:
            innovation_covariance = h @ sender_covariance[node] @ h.T + absolute_covariance
            gain = sender_covariance[node] @ h.T @ np.linalg.inv(innovation_covariance)
            update_transition = np.eye(6) - gain @ h
            update_noise = gain @ absolute_covariance @ gain.T
            measurement = truth[node][index, :3] + rng.normal(0.0, absolute_sigma, 3)
            sender_state[node] += gain @ (measurement - h @ sender_state[node])
            sender_covariance[node] = update_transition @ sender_covariance[node] @ update_transition.T + update_noise
            for receiver, source in edges:
                if source != node:
                    continue
                combined_transition = update_transition @ prediction_transition[node]
                combined_noise = update_transition @ prediction_noise[node] @ update_transition.T + update_noise
                accumulator = accumulators[(receiver, source)]
                accumulator.append(
                    timestamp=float(timestamp), updated_state=sender_state[node],
                    error_transition=combined_transition,
                    independent_process_noise=combined_noise,
                    information_ids=(f"{node}:absolute:{index}",),
                    event_error_transition=update_transition,
                    event_process_noise=update_noise,
                )
                message = accumulator.build_message()
                transmitted = channels[(receiver, source)].transmit(message)
                if transmitted is not None:
                    state_messages[receiver].append(transmitted)
                    transmitted_messages.append(transmitted)
                    pending_acks.append((float(transmitted.arrival_timestamp), (receiver, source), message))
        for observer in topology.node_ids:
            for target in topology.neighbors(observer):
                information_id = f"{observer}->{target}:range:{index}"
                observations.append(ObservationMessage(
                    message_id=information_id, observer_id=observer, target_id=target,
                    timestamp=float(timestamp), modality="RANGE",
                    measurement=np.array([measure_relative_range(truth[observer][index], truth[target][index]) + rng.normal(0.0, range_sigma)]),
                    covariance=np.array([[range_sigma**2]]),
                ))
    return {
        "timestamps": timestamps, "truth": truth, "initial_states": initial_states,
        "initial_covariances": initial_covariances, "topology": topology,
        "observations": observations, "state_messages": state_messages,
        "transmitted_messages": transmitted_messages,
        "lineages": {(receiver, source): f"{source}->{receiver}:0" for receiver, source in edges},
    }


def _metrics(history, truth, transmitted_count):
    position = []; velocity = []; nees = []; nis = []; minimum = float("inf"); failures = 0
    for node in history.node_ids:
        error = history.active_state_history_by_node[node] - truth[node]
        position.append(error[:, :3]); velocity.append(error[:, 3:])
        nees.extend(compute_nees_history(history.active_state_history_by_node[node], truth[node], history.active_covariance_history_by_node[node]))
        nis.extend(value for epoch in history.nis_history_by_node[node] for value in epoch.values())
        for covariance in history.joint_covariance_history_by_node[node]:
            value = float(np.linalg.eigvalsh(covariance).min()); minimum = min(minimum, value)
            failures += int(value < -1e-8)
    nees = np.asarray(nees); nis = np.asarray(nis)
    accepted = int(history.refresh_diagnostics.get("accepted", 0))
    rejected = sum(value for key, value in history.refresh_diagnostics.items() if key != "accepted")
    rejection_counts = {
        key: int(value) for key, value in history.refresh_diagnostics.items()
        if key != "accepted" and value
    }
    return (
        compute_rmse(np.vstack(position)), compute_rmse(np.vstack(velocity)),
        float(np.mean(nees)), _coverage(nees, NEES_95_DOF6),
        float(np.mean(nis)), _coverage(nis, NIS_95_DOF1), minimum,
        accepted, transmitted_count, rejected, failures, rejection_counts,
    )


def _coverage(values, interval):
    lower, upper = interval
    return float(np.mean((values >= lower) & (values <= upper)))


def _sum_rejection_counts(values):
    result = {}
    for counts in values:
        for key, count in counts.items():
            result[key] = result.get(key, 0) + int(count)
    return result
