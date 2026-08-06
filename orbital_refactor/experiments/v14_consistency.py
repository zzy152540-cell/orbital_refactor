from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cooperative.recursive_cooperative_runner import (
    run_recursive_distributed_cooperative_filter,
)
from cooperative.dual_track_runner import (
    run_dual_track_distributed_cooperative_filter,
)
from cooperative.schmidt_consider import run_schmidt_consider_history
from cooperative.network_schmidt_runner import run_network_schmidt_filter
from cooperative.topology import chain_topology
from experiments.consistency_metrics import fleet_consistency_metrics
from interfaces.data_objects import ObservationMessage
from experiments.summary_statistics import mean_metric_dict
from orbital_core.constants import R_EARTH
from orbital_core.measurements import measure_relative_range
from orbital_core.orbit_elements import keplerian_to_eci
from scenarios.fleet_scenario import generate_fleet_scenario

Array = np.ndarray

STRATEGIES = (
    "single_endpoint",
    "single_endpoint_schmidt",
    "shared",
    "shared_dual_track",
    "independent_reciprocal",
)


@dataclass(frozen=True)
class ConsistencyRun:
    strategy: str
    seed: int
    fleet_position_rmse: float
    fleet_velocity_rmse: float
    mean_nees: float
    nees_95_coverage: float
    mean_nis: float
    nis_95_coverage: float


@dataclass(frozen=True)
class ConsistencySummary:
    strategy: str
    run_count: int
    mean_position_rmse: float
    mean_velocity_rmse: float
    mean_nees: float
    mean_nees_95_coverage: float
    mean_nis: float
    mean_nis_95_coverage: float


@dataclass(frozen=True)
class V14ConsistencyResult:
    runs: tuple[ConsistencyRun, ...]
    summary_by_strategy: dict[str, ConsistencySummary]


def run_v14_network_schmidt_monte_carlo(
    *, seeds: int = 20, duration: float = 20.0, dt: float = 2.0,
    range_sigma: float = 2.0, process_noise_acceleration: float = 1e-8,
) -> V14ConsistencyResult:
    """Compare approximate and one-hop Schmidt filters on a three-node chain."""
    if seeds < 1:
        raise ValueError("seeds must be at least one.")
    runs = []
    for seed in range(seeds):
        runs.extend(_run_network_pair(
            seed=seed, duration=duration, dt=dt, range_sigma=range_sigma,
            process_noise_acceleration=process_noise_acceleration,
        ))
    return V14ConsistencyResult(
        tuple(runs), _summarize_runs(runs, ("network_approximate", "network_schmidt"))
    )


def run_v14_network_refresh_monte_carlo(
    *, seeds: int = 20, duration: float = 20.0, dt: float = 2.0,
    range_sigma: float = 2.0, process_noise_acceleration: float = 1e-8,
) -> V14ConsistencyResult:
    """Compare synchronous consider-block refresh policies on identical cases."""
    if seeds < 1:
        raise ValueError("seeds must be at least one.")
    modes = ("propagate_only", "safe_rescale", "zero_cross", "exact_if_compatible")
    runs = []
    for seed in range(seeds):
        candidates = _run_network_pair(
            seed=seed, duration=duration, dt=dt, range_sigma=range_sigma,
            process_noise_acceleration=process_noise_acceleration,
            refresh_modes=modes,
        )
        runs.extend(run for run in candidates if run.strategy.startswith("refresh_"))
    strategies = tuple(f"refresh_{mode}" for mode in modes)
    return V14ConsistencyResult(tuple(runs), _summarize_runs(runs, strategies))


def run_v14_range_consistency_monte_carlo(
    *,
    seeds: int = 30,
    duration: float = 20.0,
    dt: float = 2.0,
    range_sigma: float = 2.0,
    process_noise_acceleration: float = 1e-8,
) -> V14ConsistencyResult:
    if seeds < 1:
        raise ValueError("seeds must be at least one.")
    if duration < 0.0 or dt <= 0.0 or range_sigma <= 0.0:
        raise ValueError("duration, dt, and range_sigma are invalid.")
    runs: list[ConsistencyRun] = []
    for seed in range(seeds):
        for strategy in STRATEGIES:
            runs.append(
                _run_one(
                    seed=seed,
                    strategy=strategy,
                    duration=duration,
                    dt=dt,
                    range_sigma=range_sigma,
                    process_noise_acceleration=process_noise_acceleration,
                )
            )
    return V14ConsistencyResult(tuple(runs), _summarize_runs(runs, STRATEGIES))


def _summarize_runs(runs, strategies):
    summaries = {}
    for strategy in strategies:
        selected = [run for run in runs if run.strategy == strategy]
        metric_names = (
            "fleet_position_rmse", "fleet_velocity_rmse", "mean_nees",
            "nees_95_coverage", "mean_nis", "nis_95_coverage",
        )
        means = mean_metric_dict([
            {name: getattr(run, name) for name in metric_names}
            for run in selected
        ])
        summaries[strategy] = ConsistencySummary(
            strategy=strategy,
            run_count=len(selected),
            mean_position_rmse=means["fleet_position_rmse"],
            mean_velocity_rmse=means["fleet_velocity_rmse"],
            mean_nees=means["mean_nees"],
            mean_nees_95_coverage=means["nees_95_coverage"],
            mean_nis=means["mean_nis"],
            mean_nis_95_coverage=means["nis_95_coverage"],
        )
    return summaries


def _run_one(
    *,
    seed: int,
    strategy: str,
    duration: float,
    dt: float,
    range_sigma: float,
    process_noise_acceleration: float,
) -> ConsistencyRun:
    rng = np.random.default_rng(20260801 + seed)
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt)
    base = keplerian_to_eci(
        R_EARTH + 700e3,
        0.001,
        np.deg2rad(23.0),
        0.0,
        0.0,
        0.0,
    )
    truth_initials = {
        "sat_a": base.copy(),
        "sat_b": base + np.array([1000.0, 200.0, 100.0, 0.0, 0.02, 0.0]),
    }
    scenario = generate_fleet_scenario(
        timestamps=timestamps,
        initial_state_by_node=truth_initials,
    )
    covariance = np.diag([10.0, 10.0, 10.0, 0.02, 0.02, 0.02]) ** 2
    initial_states = {
        node_id: truth_initials[node_id]
        + rng.multivariate_normal(np.zeros(6), covariance)
        for node_id in scenario.node_ids
    }
    initial_covariances = {
        node_id: covariance.copy() for node_id in scenario.node_ids
    }
    messages = _range_messages(
        scenario.truth_state_history_by_node,
        timestamps=timestamps,
        strategy=strategy,
        range_sigma=range_sigma,
        rng=rng,
    )
    topology = chain_topology(["sat_a", "sat_b"])
    if strategy == "single_endpoint_schmidt":
        schmidt = run_schmidt_consider_history(
            timestamps=timestamps,
            active_node_id="sat_a",
            consider_node_id="sat_b",
            initial_active_state=initial_states["sat_a"],
            initial_consider_state=initial_states["sat_b"],
            initial_active_covariance=initial_covariances["sat_a"],
            initial_consider_covariance=initial_covariances["sat_b"],
            observations=messages,
            process_noise_acceleration=process_noise_acceleration,
        )
        estimate_by_node = {
            "sat_a": schmidt.active_state_history,
            "sat_b": schmidt.consider_state_history,
        }
        covariance_by_node = {
            "sat_a": schmidt.active_covariance_history,
            "sat_b": schmidt.consider_covariance_history,
        }
        nis_history_by_node = {"sat_a": schmidt.nis_history, "sat_b": []}
    elif strategy == "shared_dual_track":
        dual = run_dual_track_distributed_cooperative_filter(
            timestamps=timestamps,
            initial_state_by_node=initial_states,
            initial_covariance_by_node=initial_covariances,
            topology=topology,
            observation_messages=messages,
            cooperative_observation_usage="both_endpoints",
            process_noise_acceleration=process_noise_acceleration,
        )
        history = dual.cooperative_history
        estimate_by_node = history.posterior_state_history_by_node
        covariance_by_node = history.posterior_covariance_history_by_node
        nis_history_by_node = history.nis_history_by_node
    else:
        usage = "both_endpoints" if strategy == "shared" else "observer_only"
        history = run_recursive_distributed_cooperative_filter(
            timestamps=timestamps,
            initial_state_by_node=initial_states,
            initial_covariance_by_node=initial_covariances,
            topology=topology,
            observation_messages=messages,
            observation_usage=usage,
            process_noise_acceleration=process_noise_acceleration,
        )
        estimate_by_node = history.posterior_state_history_by_node
        covariance_by_node = history.posterior_covariance_history_by_node
        nis_history_by_node = history.nis_history_by_node
    metrics = fleet_consistency_metrics(
        truth=scenario.truth_state_history_by_node,
        estimates=estimate_by_node,
        covariances=covariance_by_node,
        nis_history=nis_history_by_node,
    )
    return ConsistencyRun(
        strategy=strategy,
        seed=seed,
        fleet_position_rmse=metrics.position_rmse,
        fleet_velocity_rmse=metrics.velocity_rmse,
        mean_nees=metrics.mean_nees,
        nees_95_coverage=metrics.nees_95_coverage,
        mean_nis=metrics.mean_nis,
        nis_95_coverage=metrics.nis_95_coverage,
    )


def _range_messages(
    truth: dict[str, Array],
    *,
    timestamps: Array,
    strategy: str,
    range_sigma: float,
    rng: np.random.Generator,
) -> list[ObservationMessage]:
    directions = (
        [("sat_a", "sat_b"), ("sat_b", "sat_a")]
        if strategy == "independent_reciprocal"
        else [("sat_a", "sat_b")]
    )
    messages = []
    for observer, target in directions:
        for index, timestamp in enumerate(timestamps):
            physical_id = (
                f"{strategy}:{observer}->{target}:RANGE:{float(timestamp):.9f}"
            )
            messages.append(
                ObservationMessage(
                    message_id=f"message:{physical_id}",
                    physical_observation_id=physical_id,
                    observer_id=observer,
                    target_id=target,
                    timestamp=float(timestamp),
                    modality="RANGE",
                    measurement=np.array([
                        measure_relative_range(
                            truth[observer][index],
                            truth[target][index],
                        )
                        + rng.normal(0.0, range_sigma)
                    ]),
                    covariance=np.array([[range_sigma**2]]),
                )
            )
    return messages


def _run_network_pair(
    *, seed: int, duration: float, dt: float, range_sigma: float,
    process_noise_acceleration: float,
    refresh_modes: tuple[str, ...] | None = None,
) -> list[ConsistencyRun]:
    rng = np.random.default_rng(20260811 + seed)
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt)
    base = keplerian_to_eci(
        R_EARTH + 700e3, 0.001, np.deg2rad(23.0), 0.0, 0.0, 0.0
    )
    truth_initials = {
        "sat_01": base + np.array([-1200.0, 100.0, 20.0, 0.0, -0.02, 0.0]),
        "sat_02": base.copy(),
        "sat_03": base + np.array([1300.0, -80.0, 30.0, 0.0, 0.03, 0.0]),
    }
    scenario = generate_fleet_scenario(
        timestamps=timestamps, initial_state_by_node=truth_initials
    )
    covariance = np.diag([10.0, 10.0, 10.0, 0.02, 0.02, 0.02]) ** 2
    initial_states = {
        node_id: truth_initials[node_id]
        + rng.multivariate_normal(np.zeros(6), covariance)
        for node_id in scenario.node_ids
    }
    initial_covariances = {node_id: covariance.copy() for node_id in scenario.node_ids}
    topology = chain_topology(list(scenario.node_ids))
    messages = []
    for observer in topology.node_ids:
        for target in topology.neighbors(observer):
            for index, timestamp in enumerate(timestamps):
                information_id = f"{observer}->{target}:range:{float(timestamp):.9f}"
                messages.append(ObservationMessage(
                    message_id=f"message:{information_id}",
                    physical_observation_id=information_id,
                    observer_id=observer, target_id=target,
                    timestamp=float(timestamp), modality="RANGE",
                    measurement=np.array([
                        measure_relative_range(
                            scenario.truth_state_history_by_node[observer][index],
                            scenario.truth_state_history_by_node[target][index],
                        ) + rng.normal(0.0, range_sigma)
                    ]),
                    covariance=np.array([[range_sigma**2]]),
                ))
    approximate = run_recursive_distributed_cooperative_filter(
        timestamps=timestamps, initial_state_by_node=initial_states,
        initial_covariance_by_node=initial_covariances, topology=topology,
        observation_messages=messages, observation_usage="observer_only",
        process_noise_acceleration=process_noise_acceleration,
    )
    modes = ("propagate_only",) if refresh_modes is None else refresh_modes
    schmidt_by_mode = {
        mode: run_network_schmidt_filter(
            timestamps=timestamps, initial_state_by_node=initial_states,
            initial_covariance_by_node=initial_covariances, topology=topology,
            observation_messages=messages, observation_usage="observer_only",
            process_noise_acceleration=process_noise_acceleration,
            consider_refresh_mode=mode,
        )
        for mode in modes
    }
    runs = [
        _network_run_metrics(
            strategy="network_approximate", seed=seed,
            truth=scenario.truth_state_history_by_node,
            estimates=approximate.posterior_state_history_by_node,
            covariances=approximate.posterior_covariance_history_by_node,
            nis_history=approximate.nis_history_by_node,
        ),
    ]
    for mode, schmidt in schmidt_by_mode.items():
        runs.append(_network_run_metrics(
            strategy=("network_schmidt" if refresh_modes is None else f"refresh_{mode}"), seed=seed,
            truth=scenario.truth_state_history_by_node,
            estimates=schmidt.active_state_history_by_node,
            covariances=schmidt.active_covariance_history_by_node,
            nis_history=schmidt.nis_history_by_node,
        ))
    return runs


def _network_run_metrics(
    *, strategy: str, seed: int, truth: dict[str, Array],
    estimates: dict[str, Array], covariances: dict[str, Array],
    nis_history: dict[str, list[dict[str, float]]],
) -> ConsistencyRun:
    metrics = fleet_consistency_metrics(
        truth=truth, estimates=estimates, covariances=covariances,
        nis_history=nis_history,
    )
    return ConsistencyRun(
        strategy=strategy, seed=seed,
        fleet_position_rmse=metrics.position_rmse,
        fleet_velocity_rmse=metrics.velocity_rmse,
        mean_nees=metrics.mean_nees,
        nees_95_coverage=metrics.nees_95_coverage,
        mean_nis=metrics.mean_nis,
        nis_95_coverage=metrics.nis_95_coverage,
    )
