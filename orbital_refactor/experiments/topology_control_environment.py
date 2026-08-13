from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cooperative.network_schmidt_orchestrator import NetworkSchmidtOrchestrator
from cooperative.online_graph_observation import build_online_graph_observation
from cooperative.topology import chain_topology, fully_connected_topology
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


class TopologyControlEnvironment:
    """Minimal truth-safe observation / truth-aware training environment."""

    def __init__(
        self, *, node_count: int = 5, episode_epochs: int = 6,
        decision_interval_epochs: int = 1, dt: float = 2.0,
        relative_modalities=("RANGE", "RANGE_RATE", "AZ_EL"),
        packet_loss: float = 0.0, communication_delay: float = 0.0,
        visibility_by_modality=None,
        minimum_topology_dwell_decisions: int = 0,
        top_k_candidate_neighbors: int | None = None,
        scenario_type: str = "compact_fleet",
        walker_maximum_range: float = 7000e3,
        randomize_stage1_conditions: bool = False,
    ) -> None:
        if scenario_type not in {"compact_fleet", "walker_20_5_3"}:
            raise ValueError("Unsupported topology environment scenario type.")
        if scenario_type == "compact_fleet" and node_count not in {3, 5}:
            raise ValueError("The compact V15 environment supports 3 or 5 nodes.")
        if scenario_type == "walker_20_5_3" and node_count != 20:
            raise ValueError("Walker 20/5/3 requires node_count=20.")
        if episode_epochs < 1 or decision_interval_epochs < 1 or dt <= 0.0:
            raise ValueError("Environment horizon and time step must be positive.")
        if minimum_topology_dwell_decisions < 0:
            raise ValueError("Minimum topology dwell cannot be negative.")
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
        if top_k_candidate_neighbors is not None and top_k_candidate_neighbors < 0:
            raise ValueError("Top-K candidate neighbors cannot be negative.")
        self.top_k_candidate_neighbors = top_k_candidate_neighbors
        self.scenario_type = scenario_type
        self.walker_maximum_range = float(walker_maximum_range)
        self.randomize_stage1_conditions = bool(randomize_stage1_conditions)
        self._case = self._orchestrator = None

    def reset(self, *, seed: int = 0) -> TopologyEnvironmentState:
        self._episode_conditions = self._sample_episode_conditions(int(seed))
        candidate, baseline, self._case = self._build_case(int(seed))
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
            random_seed=int(seed) + 31000,
            resynchronize_on_resume=True,
            batch_relative_observations=True,
        )
        self._active_edges = _topology_edges(baseline)
        self._topology_version = 1
        self._epoch_index = 0
        self._cooldown_remaining = 0
        self._decisions_since_switch = 0
        self._last_metrics = None
        initial_step = self._advance_one_epoch()
        self._last_metrics = self._metrics()
        return self._state(initial_step)

    def _build_case(self, seed):
        if self.scenario_type == "walker_20_5_3":
            plan = build_v14_walker_dynamic_topology_plan(
                duration=self.episode_epochs * self.dt, dt=self.dt,
                maximum_range=self.walker_maximum_range,
            )
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
            )
            return candidate, baseline, case
        candidate = fully_connected_topology(tuple(
            f"sat_{index + 1:02d}" for index in range(self.node_count)
        ))
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
        )
        baseline = chain_topology(candidate.node_ids)
        return candidate, baseline, case

    def step(self, action_id: int) -> TopologyEnvironmentStep:
        self._require_reset()
        before = self._last_metrics
        action_space = self._state().action_space
        resolution = resolve_topology_action(action_space, action_id)
        selected = resolution.executed_action.topology.active_edges
        switched = tuple(selected) != tuple(self._active_edges)
        if switched:
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
        terminated = self._epoch_index + 1 >= len(self._case["timestamps"])
        state = self._state(last_step)
        return TopologyEnvironmentStep(
            state=state, reward=reward_terms.task_reward,
            reward_terms=reward_terms, constraint_costs=costs,
            terminated=terminated, truncated=False,
            action_resolution=resolution,
            diagnostics=(
                ("position_rmse", after[0]),
                ("worst_node_position_rmse", after[1]),
                ("mean_covariance_logdet", after[2]),
            ),
        )

    def _advance_one_epoch(self):
        timestamp = float(self._case["timestamps"][self._epoch_index])
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
            or self.scenario_type == "walker_20_5_3"
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

    def _sample_episode_conditions(self, seed):
        if not self.randomize_stage1_conditions:
            return {
                "packet_loss": self.packet_loss,
                "communication_delay": self.communication_delay,
                "navigation_dropout_by_node": {},
            }
        if self.scenario_type != "compact_fleet":
            raise ValueError("Stage 1 randomization currently supports compact fleets.")
        rng = np.random.default_rng(20260901 + seed)
        node = f"sat_{int(rng.integers(1, self.node_count + 1)):02d}"
        start_epoch = int(rng.integers(1, max(2, self.episode_epochs // 2 + 1)))
        end_epoch = int(rng.integers(
            start_epoch + 1, self.episode_epochs + 1
        ))
        return {
            "packet_loss": float(rng.uniform(0.0, 0.2)),
            "communication_delay": float(rng.uniform(0.0, 2.0)),
            "navigation_dropout_by_node": {
                node: ((start_epoch * self.dt, end_epoch * self.dt),)
            },
        }

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
