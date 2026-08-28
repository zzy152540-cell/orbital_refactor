from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cooperative.network_schmidt_orchestrator import NetworkSchmidtOrchestrator
from cooperative.online_graph_observation import build_online_graph_observation
from cooperative.topology import (
    chain_topology,
    fully_connected_topology,
    ring_topology,
    star_topology,
)
from cooperative.topology_action_space import (
    ResolvedTopologyAction,
    TopologyActionSpace,
    build_topology_action_space,
    resolve_topology_action,
)
from cooperative.topology_candidate_selection import select_top_k_addition_edges
from cooperative.v15_policy_tensor import V15PolicyTensor, tensorize_v15_policy_observation
from experiments.v14_exact_transport_scale_scan import build_exact_transport_case
from experiments.v14_online_topology_resynchronization import (
    _items_by_timestamp,
    _source_updates_from_messages,
)
from experiments.v14_walker_dynamic_topology import (
    build_v14_walker_dynamic_topology_plan,
)
from experiments.walker_filter_setup import (
    build_walker_filter_case,
)
from experiments.counterfactual_physical_scenarios import (
    FIVE_NODE_PHYSICAL_FAMILIES,
    sample_five_satellite_physical_scenario,
)
from orbital_core.constants import R_EARTH
from scenarios.walker_scenario import WalkerDeltaConfig


@dataclass(frozen=True)
class TopologyRewardTerms:
    position_rmse_improvement: float
    worst_node_rmse_improvement: float
    covariance_logdet_reduction: float

    @property
    def task_reward(self) -> float:
        return self.position_rmse_improvement


@dataclass(frozen=True)
class TopologyConstraintCosts:
    transmitted_messages: float
    dropped_messages: float
    replay_count: float
    resynchronization_count: float
    topology_switch: float
    action_fallback: float


@dataclass(frozen=True)
class TopologyEnvironmentState:
    observation: object
    policy_tensor: V15PolicyTensor
    action_space: TopologyActionSpace


@dataclass(frozen=True)
class TopologyEnvironmentStep:
    state: TopologyEnvironmentState
    reward: float
    reward_terms: TopologyRewardTerms
    constraint_costs: TopologyConstraintCosts
    terminated: bool
    truncated: bool
    action_resolution: ResolvedTopologyAction
    diagnostics: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class WalkerInitializationDistribution:
    """Bounded, seeded Walker truth initialization for curriculum studies."""

    altitude_range: tuple[float, float] = (550e3, 850e3)
    eccentricity_range: tuple[float, float] = (0.0, 0.002)
    inclination_range: tuple[float, float] = (
        float(np.deg2rad(35.0)), float(np.deg2rad(75.0)),
    )
    plane_phasing_options: tuple[tuple[int, int], ...] = ()

    def validate(self, node_count: int) -> None:
        altitude_low, altitude_high = self.altitude_range
        eccentricity_low, eccentricity_high = self.eccentricity_range
        inclination_low, inclination_high = self.inclination_range
        if not 0.0 < altitude_low <= altitude_high:
            raise ValueError("Walker altitude range must be positive and ordered.")
        if not 0.0 <= eccentricity_low <= eccentricity_high < 1.0:
            raise ValueError("Walker eccentricity range must lie within [0, 1).")
        if not 0.0 <= inclination_low <= inclination_high <= np.pi:
            raise ValueError("Walker inclination range must lie within [0, pi].")
        if not self.plane_phasing_options:
            raise ValueError("Walker plane/phasing options cannot be empty.")
        if len(set(self.plane_phasing_options)) != len(self.plane_phasing_options):
            raise ValueError("Walker plane/phasing options must be unique.")
        for plane_count, phasing in self.plane_phasing_options:
            if (
                plane_count < 1 or node_count % plane_count
                or not 0 <= phasing < plane_count
            ):
                raise ValueError("Invalid Walker plane/phasing option.")


@dataclass(frozen=True)
class CompactFleetScenarioDistribution:
    """Seeded Stage-1 disturbance distribution for compact fleets."""

    packet_loss_range: tuple[float, float] = (0.0, 0.2)
    communication_delay_range: tuple[float, float] = (0.0, 2.0)
    navigation_dropout_node_count: int = 1
    initial_topology_types: tuple[str, ...] = ("chain",)
    link_condition_mode: str = "homogeneous"
    physical_scenario_families: tuple[str, ...] = ()
    physical_family_assignment_mode: str = "random"
    dynamic_link_event_count: int = 0
    dynamic_packet_loss_range: tuple[float, float] = (0.6, 1.0)
    dynamic_delay_range: tuple[float, float] = (2.0, 6.0)
    walker_initialization: WalkerInitializationDistribution | None = None

    def validate(self, node_count: int) -> None:
        loss_low, loss_high = self.packet_loss_range
        delay_low, delay_high = self.communication_delay_range
        if not (0.0 <= loss_low <= loss_high <= 1.0):
            raise ValueError("Packet-loss range must lie within [0, 1].")
        if not (0.0 <= delay_low <= delay_high):
            raise ValueError("Communication-delay range must be nonnegative.")
        if not 0 <= self.navigation_dropout_node_count <= node_count:
            raise ValueError("Navigation-dropout count exceeds fleet size.")
        if self.dynamic_link_event_count < 0:
            raise ValueError("Dynamic-link event count cannot be negative.")
        dynamic_loss_low, dynamic_loss_high = self.dynamic_packet_loss_range
        dynamic_delay_low, dynamic_delay_high = self.dynamic_delay_range
        if not (0.0 <= dynamic_loss_low <= dynamic_loss_high <= 1.0):
            raise ValueError("Dynamic packet-loss range must lie within [0, 1].")
        if not (0.0 <= dynamic_delay_low <= dynamic_delay_high):
            raise ValueError("Dynamic communication-delay range must be nonnegative.")
        supported = {"chain", "ring", "star"}
        if (
            not self.initial_topology_types
            or set(self.initial_topology_types) - supported
            or len(set(self.initial_topology_types)) != len(
                self.initial_topology_types
            )
        ):
            raise ValueError("Initial topology types must be unique supported names.")
        if self.link_condition_mode not in {
            "homogeneous", "undirected_independent",
        }:
            raise ValueError("Unsupported compact-fleet link-condition mode.")
        if (
            len(set(self.physical_scenario_families))
            != len(self.physical_scenario_families)
            or set(self.physical_scenario_families)
            - set(FIVE_NODE_PHYSICAL_FAMILIES)
        ):
            raise ValueError("Unsupported or duplicate physical scenario family.")
        if self.physical_scenario_families and node_count != 5:
            raise ValueError("Randomized physical scenario families require five nodes.")
        if self.physical_family_assignment_mode not in {"random", "seed_cycle"}:
            raise ValueError("Unsupported physical family-assignment mode.")
        if self.walker_initialization is not None:
            self.walker_initialization.validate(node_count)


class TopologyControlEnvironment:
    """Minimal truth-safe observation / truth-aware training environment."""

    def __init__(
        self, *, node_count: int = 5, episode_epochs: int = 6,
        decision_interval_epochs: int = 1, dt: float = 2.0,
        relative_modalities=("RANGE", "RANGE_RATE", "AZ_EL"),
        packet_loss: float = 0.0, communication_delay: float = 0.0,
        visibility_by_modality=None,
        minimum_topology_dwell_decisions: int = 0,
        maximum_topology_switches_per_episode: int | None = None,
        top_k_candidate_neighbors: int | None = None,
        scenario_type: str = "compact_fleet",
        walker_maximum_range: float = 7000e3,
        walker_plane_count: int = 5,
        walker_phasing: int = 3,
        treat_horizon_as_truncation: bool = False,
        randomize_stage1_conditions: bool = False,
        compact_scenario_distribution: CompactFleetScenarioDistribution | None = None,
    ) -> None:
        if scenario_type not in {
            "compact_fleet", "walker_20_5_3", "walker_delta",
        }:
            raise ValueError("Unsupported topology environment scenario type.")
        if scenario_type == "compact_fleet" and node_count not in {3, 5}:
            raise ValueError("The compact V15 environment supports 3 or 5 nodes.")
        if scenario_type == "walker_20_5_3" and node_count != 20:
            raise ValueError("Walker 20/5/3 requires node_count=20.")
        if scenario_type == "walker_delta" and (
            node_count < 2
            or walker_plane_count < 1
            or node_count % walker_plane_count
        ):
            raise ValueError(
                "Walker delta requires node_count divisible by plane_count."
            )
        if episode_epochs < 1 or decision_interval_epochs < 1 or dt <= 0.0:
            raise ValueError("Environment horizon and time step must be positive.")
        if minimum_topology_dwell_decisions < 0:
            raise ValueError("Minimum topology dwell cannot be negative.")
        if (
            maximum_topology_switches_per_episode is not None
            and maximum_topology_switches_per_episode < 0
        ):
            raise ValueError("Maximum topology switches cannot be negative.")
        self.node_count = int(node_count)
        self.episode_epochs = int(episode_epochs)
        self.decision_interval_epochs = int(decision_interval_epochs)
        self.dt = float(dt)
        self.relative_modalities = tuple(relative_modalities)
        self.packet_loss = float(packet_loss)
        self.communication_delay = float(communication_delay)
        self.visibility_by_modality = visibility_by_modality
        self.minimum_topology_dwell_decisions = int(
            minimum_topology_dwell_decisions
        )
        self.maximum_topology_switches_per_episode = (
            None if maximum_topology_switches_per_episode is None
            else int(maximum_topology_switches_per_episode)
        )
        if top_k_candidate_neighbors is not None and top_k_candidate_neighbors < 0:
            raise ValueError("Top-K candidate neighbors cannot be negative.")
        self.top_k_candidate_neighbors = top_k_candidate_neighbors
        self.scenario_type = scenario_type
        self.walker_maximum_range = float(walker_maximum_range)
        self.walker_plane_count = int(walker_plane_count)
        self.walker_phasing = int(walker_phasing)
        self.treat_horizon_as_truncation = bool(treat_horizon_as_truncation)
        self.randomize_stage1_conditions = bool(randomize_stage1_conditions)
        self.compact_scenario_distribution = (
            compact_scenario_distribution or CompactFleetScenarioDistribution()
        )
        self.compact_scenario_distribution.validate(self.node_count)
        self._case = self._orchestrator = None

    def reset(
        self, *, seed: int = 0, condition_seed: int | None = None,
    ) -> TopologyEnvironmentState:
        condition_seed = int(seed if condition_seed is None else condition_seed)
        self._episode_conditions = self._sample_episode_conditions(condition_seed)
        self._condition_seed = condition_seed
        candidate, baseline, self._case = self._build_case(int(seed))
        self._episode_conditions["dynamic_link_events_by_link"] = (
            self._sample_dynamic_link_events(
                np.random.default_rng(20261003 + condition_seed),
                _topology_edges(candidate),
                self.compact_scenario_distribution,
            )
        )
        self._source_updates = _source_updates_from_messages(
            self._case["transmitted_messages"], candidate.node_ids
        )
        self._relative_by_time = _items_by_timestamp(self._case["observations"])
        self._absolute_by_time = _items_by_timestamp(
            self._case["absolute_observations"]
        )
        self._orchestrator = NetworkSchmidtOrchestrator(
            initial_state_by_node=self._case["initial_states"],
            initial_covariance_by_node=self._case["initial_covariances"],
            topology=candidate,
            initial_timestamp=float(self._case["timestamps"][0]),
            process_noise_acceleration=1e-8, history_window=10.0,
            packet_loss_rate=self._episode_conditions["packet_loss"],
            communication_delay=self._episode_conditions["communication_delay"],
            packet_loss_rate_by_link=self._episode_conditions[
                "packet_loss_rate_by_link"
            ],
            communication_delay_by_link=self._episode_conditions[
                "communication_delay_by_link"
            ],
            random_seed=int(seed) + 31000,
            resynchronize_on_resume=True,
            batch_relative_observations=True,
        )
        self._active_edges = _topology_edges(baseline)
        self._topology_version = 1
        self._epoch_index = 0
        self._cooldown_remaining = 0
        self._decisions_since_switch = 0
        self._topology_switch_count = 0
        self._last_metrics = None
        initial_step = self._advance_one_epoch()
        self._last_metrics = self._metrics()
        return self._state(initial_step)

    def _build_case(self, seed):
        if self.scenario_type in {"walker_20_5_3", "walker_delta"}:
            default_walker_config = (
                None
                if self.scenario_type == "walker_20_5_3"
                else WalkerDeltaConfig(
                    total_satellites=self.node_count,
                    plane_count=self.walker_plane_count,
                    phasing=self.walker_phasing,
                    semi_major_axis=R_EARTH + 700e3,
                    eccentricity=0.0,
                    inclination=np.deg2rad(53.0),
                )
            )
            walker_candidates = self._episode_conditions.get(
                "walker_config_candidates", ()
            )
            if walker_candidates:
                walker_candidates = tuple(walker_candidates) + (default_walker_config,)
            else:
                walker_candidates = (
                    self._episode_conditions.get("walker_config")
                    or default_walker_config,
                )
            plan = None
            for candidate_index, walker_config in enumerate(walker_candidates):
                try:
                    plan = build_v14_walker_dynamic_topology_plan(
                        duration=self.episode_epochs * self.dt, dt=self.dt,
                        maximum_range=self.walker_maximum_range,
                        walker_config=walker_config,
                    )
                except ValueError as error:
                    if "visibility graph is disconnected" not in str(error):
                        raise
                    continue
                self._episode_conditions["walker_config"] = plan.scenario.config
                self._episode_conditions["walker_randomization_attempt"] = candidate_index
                self._episode_conditions["walker_randomization_fallback"] = (
                    bool(walker_candidates[:-1])
                    and candidate_index == len(walker_candidates) - 1
                )
                break
            if plan is None:  # pragma: no cover - fixed fallback is validated
                raise ValueError("No connected Walker initialization is available.")
            physical_edges = {
                edge.nodes
                for observation in plan.graph_observation_by_timestamp.values()
                for edge in observation.candidate_edges
            }
            candidate = _topology_from_edges(
                plan.scenario.node_ids, physical_edges
            )
            baseline = _topology_from_edges(
                candidate.node_ids,
                plan.epoch_records[0].active_undirected_edges,
            )
            case = build_walker_filter_case(
                seed=seed, duration=self.episode_epochs * self.dt, dt=self.dt,
                maximum_range=self.walker_maximum_range, topology=candidate,
                truth_history_by_node=(
                    plan.scenario.truth_state_history_by_node
                ),
                topology_type="v15_walker_environment_union",
                absolute_navigation_dropout_windows_by_node=(
                    self._episode_conditions["navigation_dropout_by_node"]
                ),
            )
            return candidate, baseline, case
        candidate = fully_connected_topology(tuple(
            f"sat_{index + 1:02d}" for index in range(self.node_count)
        ))
        randomized_truth = self._episode_conditions["truth_initial_states"]
        case = build_exact_transport_case(
            seed=int(seed), duration=self.episode_epochs * self.dt, dt=self.dt,
            range_sigma=2.0, range_rate_sigma=0.05,
            az_el_sigma=np.deg2rad(0.05), optical_sigma=1e-3,
            absolute_sigma=3.0, process_noise_acceleration=1e-8,
            packet_loss=0.0, delay=0.0, acknowledge_messages=True,
            node_count=self.node_count, topology_type="v15_environment_complete",
            topology_override=candidate, relative_modalities=self.relative_modalities,
            visibility_by_modality=self.visibility_by_modality,
            absolute_navigation_dropout_windows_by_node=(
                self._episode_conditions["navigation_dropout_by_node"]
            ),
            truth_initial_state_by_node=(
                None if not randomized_truth else {
                    node: np.asarray(state, dtype=float)
                    for node, state in randomized_truth
                }
            ),
        )
        baseline = _compact_initial_topology(
            candidate.node_ids,
            self._episode_conditions["initial_topology_type"],
        )
        return candidate, baseline, case

    def step(self, action_id: int) -> TopologyEnvironmentStep:
        self._require_reset()
        before = self._last_metrics
        action_space = self._state().action_space
        resolution = resolve_topology_action(action_space, action_id)
        selected = resolution.executed_action.topology.active_edges
        switched = tuple(selected) != tuple(self._active_edges)
        if switched:
            self._topology_switch_count += 1
            self._topology_version += 1
            self._active_edges = tuple(selected)
            self._cooldown_remaining = self.minimum_topology_dwell_decisions
            self._decisions_since_switch = 0
        else:
            if self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1
            self._decisions_since_switch += 1
        transmitted = dropped = resync = 0
        replay_before = self._replay_total()
        last_step = None
        for _ in range(self.decision_interval_epochs):
            if self._epoch_index + 1 >= len(self._case["timestamps"]):
                break
            self._epoch_index += 1
            last_step = self._advance_one_epoch()
            transmitted += last_step.transmitted_message_count
            dropped += last_step.dropped_message_count
            resync += len(last_step.resynchronized_links)
        after = self._metrics()
        self._last_metrics = after
        reward_terms = TopologyRewardTerms(
            position_rmse_improvement=before[0] - after[0],
            worst_node_rmse_improvement=before[1] - after[1],
            covariance_logdet_reduction=before[2] - after[2],
        )
        costs = TopologyConstraintCosts(
            transmitted_messages=float(transmitted),
            dropped_messages=float(dropped),
            replay_count=float(self._replay_total() - replay_before),
            resynchronization_count=float(resync),
            topology_switch=float(switched),
            action_fallback=float(resolution.used_fallback),
        )
        horizon_reached = self._epoch_index + 1 >= len(self._case["timestamps"])
        terminated = horizon_reached and not self.treat_horizon_as_truncation
        truncated = horizon_reached and self.treat_horizon_as_truncation
        state = self._state(last_step)
        return TopologyEnvironmentStep(
            state=state, reward=reward_terms.task_reward,
            reward_terms=reward_terms, constraint_costs=costs,
            terminated=terminated, truncated=truncated,
            action_resolution=resolution,
            diagnostics=(
                ("position_rmse", after[0]),
                ("worst_node_position_rmse", after[1]),
                ("mean_covariance_logdet", after[2]),
            ),
        )

    def _advance_one_epoch(self):
        timestamp = float(self._case["timestamps"][self._epoch_index])
        self._apply_dynamic_link_conditions(timestamp)
        return self._orchestrator.step(
            timestamp,
            topology_version=self._topology_version,
            active_neighbors_by_node=_neighbors(
                self._orchestrator.topology.node_ids, self._active_edges
            ),
            source_update_by_node={
                node: self._source_updates[(node, timestamp)]
                for node in self._orchestrator.topology.node_ids
            },
            observations=self._relative_by_time.get(timestamp, ()),
            absolute_observations=self._absolute_by_time.get(timestamp, ()),
        )

    def _state(self, latest_step=None):
        current_observations = self._relative_by_time.get(
            float(self._case["timestamps"][self._epoch_index]), ()
        )
        visible_modalities = _visible_modalities_by_edge(current_observations)
        candidate_edges = _topology_edges(self._orchestrator.topology)
        visibility_enabled = (
            self.visibility_by_modality is not None
            or self.scenario_type in {"walker_20_5_3", "walker_delta"}
        )
        observation = build_online_graph_observation(
            self._orchestrator,
            measurement_modalities_by_edge={
                edge: (
                    visible_modalities.get(edge, ())
                    if visibility_enabled else self.relative_modalities
                )
                for edge in candidate_edges
            },
            geometrically_visible_by_edge=(
                {edge: edge in visible_modalities for edge in candidate_edges}
                if visibility_enabled else None
            ),
            additional_graph_metrics={
                "decisions_since_topology_switch": float(
                    self._decisions_since_switch
                ),
                "topology_cooldown_remaining": float(
                    self._cooldown_remaining
                ),
                "minimum_topology_dwell_decisions": float(
                    self.minimum_topology_dwell_decisions
                ),
                "topology_switch_count": float(self._topology_switch_count),
                "topology_switch_budget_remaining": float(
                    -1 if self._remaining_topology_switch_budget() is None
                    else self._remaining_topology_switch_budget()
                ),
            },
            additional_node_metrics_by_node={
                node: {
                    "absolute_navigation_available": float(
                        not self._node_navigation_is_in_dropout(
                            node, float(self._case["timestamps"][self._epoch_index])
                        )
                    )
                }
                for node in self._orchestrator.topology.node_ids
            },
        )
        eligible = select_top_k_addition_edges(
            observation, top_k_per_node=self.top_k_candidate_neighbors,
        )
        return TopologyEnvironmentState(
            observation=observation,
            policy_tensor=tensorize_v15_policy_observation(observation),
            action_space=build_topology_action_space(
                observation, cooldown_remaining=self._cooldown_remaining,
                topology_switches_remaining=(
                    self._remaining_topology_switch_budget()
                ),
                eligible_addition_edges=eligible,
            ),
        )

    def _metrics(self):
        errors, node_rmse, logdets = [], [], []
        for node, session in self._orchestrator.sessions.items():
            estimate = session.state.active_state
            truth = self._case["truth"][node][self._epoch_index]
            error = estimate[:3] - truth[:3]
            errors.extend(error.tolist())
            node_rmse.append(float(np.sqrt(np.mean(error ** 2))))
            sign, value = np.linalg.slogdet(session.state.active_covariance)
            logdets.append(float(value if sign > 0 else np.inf))
        return (
            float(np.sqrt(np.mean(np.asarray(errors) ** 2))),
            float(max(node_rmse)), float(np.mean(logdets)),
        )

    def _remaining_topology_switch_budget(self):
        maximum = self.maximum_topology_switches_per_episode
        return (
            None
            if maximum is None
            else max(0, maximum - self._topology_switch_count)
        )

    def _sample_episode_conditions(self, seed):
        if not self.randomize_stage1_conditions:
            return {
                "packet_loss": self.packet_loss,
                "communication_delay": self.communication_delay,
                "packet_loss_rate_by_link": {},
                "communication_delay_by_link": {},
                "navigation_dropout_by_node": {},
                "initial_topology_type": "chain",
                "physical_scenario_family": "legacy_compact",
                "truth_initial_states": (),
            }
        rng = np.random.default_rng(20260901 + seed)
        distribution = self.compact_scenario_distribution
        topology_types = distribution.initial_topology_types
        initial_topology_type = (
            topology_types[0]
            if len(topology_types) == 1
            else str(rng.choice(topology_types))
        )
        dropout_count = distribution.navigation_dropout_node_count
        sampled_walker = None
        walker_candidates = ()
        if self.scenario_type in {"walker_20_5_3", "walker_delta"}:
            plane_count = (
                5 if self.scenario_type == "walker_20_5_3"
                else self.walker_plane_count
            )
            walker_candidates = tuple(
                self._sample_walker_initialization(rng) for _ in range(8)
            ) if distribution.walker_initialization is not None else ()
            sampled_walker = walker_candidates[0] if walker_candidates else None
            if sampled_walker is not None:
                plane_count = sampled_walker.plane_count
            slots = self.node_count // plane_count
            node_ids = tuple(
                f"sat_p{plane + 1:02d}_s{slot + 1:02d}"
                for plane in range(plane_count)
                for slot in range(slots)
            )
        else:
            node_ids = tuple(
                f"sat_{index + 1:02d}" for index in range(self.node_count)
            )
        dropout_indices = rng.choice(
            self.node_count, size=dropout_count, replace=False
        )
        start_epoch = int(rng.integers(1, max(2, self.episode_epochs // 2 + 1)))
        end_epoch = int(rng.integers(
            start_epoch + 1, self.episode_epochs + 1
        ))
        packet_loss = float(rng.uniform(*distribution.packet_loss_range))
        communication_delay = float(rng.uniform(
            *distribution.communication_delay_range
        ))
        packet_loss_by_link, delay_by_link = {}, {}
        if distribution.link_condition_mode == "undirected_independent":
            for left_index, left in enumerate(node_ids):
                for right in node_ids[left_index + 1:]:
                    edge_loss = float(rng.uniform(
                        *distribution.packet_loss_range
                    ))
                    edge_delay = float(rng.uniform(
                        *distribution.communication_delay_range
                    ))
                    packet_loss_by_link[left, right] = edge_loss
                    packet_loss_by_link[right, left] = edge_loss
                    delay_by_link[left, right] = edge_delay
                    delay_by_link[right, left] = edge_delay
        physical_scenario = (
            sample_five_satellite_physical_scenario(
                seed, families=distribution.physical_scenario_families,
                family_assignment_mode=distribution.physical_family_assignment_mode,
            )
            if (
                self.scenario_type == "compact_fleet"
                and distribution.physical_scenario_families
            ) else None
        )
        return {
            "packet_loss": packet_loss,
            "communication_delay": communication_delay,
            "packet_loss_rate_by_link": packet_loss_by_link,
            "communication_delay_by_link": delay_by_link,
            "navigation_dropout_by_node": {
                node_ids[int(index)]: (
                    (start_epoch * self.dt, end_epoch * self.dt),
                )
                for index in dropout_indices
            },
            "initial_topology_type": initial_topology_type,
            "physical_scenario_family": (
                "legacy_compact" if physical_scenario is None
                else physical_scenario.family
            ),
            "truth_initial_states": (
                () if physical_scenario is None
                else physical_scenario.truth_initial_states
            ),
            "walker_config": sampled_walker,
            "walker_config_candidates": walker_candidates,
            "dynamic_link_events_by_link": {},
        }

    def _sample_walker_initialization(self, rng):
        distribution = self.compact_scenario_distribution.walker_initialization
        if self.scenario_type not in {"walker_20_5_3", "walker_delta"} or (
            distribution is None
        ):
            return None
        option_index = int(rng.integers(len(distribution.plane_phasing_options)))
        plane_count, phasing = distribution.plane_phasing_options[option_index]
        return WalkerDeltaConfig(
            total_satellites=self.node_count,
            plane_count=int(plane_count),
            phasing=int(phasing),
            semi_major_axis=float(
                R_EARTH + rng.uniform(*distribution.altitude_range)
            ),
            eccentricity=float(rng.uniform(*distribution.eccentricity_range)),
            inclination=float(rng.uniform(*distribution.inclination_range)),
            raan_origin=float(rng.uniform(0.0, 2.0 * np.pi)),
            argument_of_perigee=float(rng.uniform(0.0, 2.0 * np.pi)),
            base_true_anomaly=float(rng.uniform(0.0, 2.0 * np.pi)),
        )

    def _sample_dynamic_link_events(self, rng, undirected_edges, distribution):
        undirected_edges = list(undirected_edges)
        count = min(distribution.dynamic_link_event_count, len(undirected_edges))
        if not count:
            return {}
        selected = rng.choice(len(undirected_edges), size=count, replace=False)
        events = {}
        for edge_index in np.atleast_1d(selected):
            left, right = undirected_edges[int(edge_index)]
            start_epoch = int(rng.integers(
                1, max(2, 2 * self.episode_epochs // 3)
            ))
            end_epoch = int(rng.integers(
                start_epoch + 1, self.episode_epochs + 1
            ))
            event = (
                start_epoch * self.dt,
                end_epoch * self.dt,
                float(rng.uniform(*distribution.dynamic_packet_loss_range)),
                float(rng.uniform(*distribution.dynamic_delay_range)),
            )
            events[left, right] = event
            events[right, left] = event
        return events

    def _apply_dynamic_link_conditions(self, timestamp):
        conditions = self._episode_conditions
        base_loss = float(conditions["packet_loss"])
        base_delay = float(conditions["communication_delay"])
        loss_by_link = conditions["packet_loss_rate_by_link"]
        delay_by_link = conditions["communication_delay_by_link"]
        events = conditions.get("dynamic_link_events_by_link", {})
        for edge, channel in self._orchestrator.channels.items():
            receiver, source = edge
            loss = float(loss_by_link.get(edge, base_loss))
            delay = float(delay_by_link.get(edge, base_delay))
            event = events.get(edge)
            if event is not None and event[0] <= timestamp <= event[1]:
                loss = max(loss, float(event[2]))
                delay = max(delay, float(event[3]))
            channel.packet_loss_rate[source] = loss
            channel.delay_by_source[source] = delay

    def _node_navigation_is_in_dropout(self, node, timestamp):
        return any(
            float(start) <= timestamp <= float(end)
            for start, end in self._episode_conditions[
                "navigation_dropout_by_node"
            ].get(node, ())
        )
    def _replay_total(self):
        return sum(session.coordinator.performance.replay_count
                   for session in self._orchestrator.sessions.values())

    def _require_reset(self):
        if self._orchestrator is None:
            raise RuntimeError("Call reset before stepping the environment.")


def _topology_edges(topology):
    return tuple(sorted({
        tuple(sorted((node, neighbor)))
        for node in topology.node_ids for neighbor in topology.neighbors(node)
    }))


def _compact_initial_topology(node_ids, topology_type):
    builders = {
        "chain": chain_topology,
        "ring": ring_topology,
        "star": star_topology,
    }
    return builders[topology_type](node_ids)


def _topology_from_edges(node_ids, edges):
    from cooperative.topology import NetworkTopology

    return NetworkTopology(_neighbors(node_ids, edges))


def _neighbors(nodes, edges):
    values = {node: [] for node in nodes}
    for left, right in edges:
        values[left].append(right)
        values[right].append(left)
    return {node: tuple(sorted(neighbors)) for node, neighbors in values.items()}


def _visible_modalities_by_edge(observations):
    modalities = {}
    for observation in observations:
        edge = tuple(sorted((observation.observer_id, observation.target_id)))
        modalities.setdefault(edge, set()).add(str(observation.modality))
    return {
        edge: tuple(sorted(values)) for edge, values in modalities.items()
    }
