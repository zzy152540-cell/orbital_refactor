from __future__ import annotations

from dataclasses import dataclass, replace

from experiments.topology_control_environment import (
    CompactFleetScenarioDistribution,
)
from experiments.topology_ppo_stage1 import (
    Stage1Configuration,
    five_node_stratified_physical_configuration,
)


@dataclass(frozen=True)
class VariableScaleTopologyCurriculum:
    """Seeded 5/10/20-node curriculum for a shared graph policy."""

    node_count_cycle: tuple[int, ...] = (
        5, 10, 20, 5, 10, 5, 20, 10, 5, 10,
        20, 5, 10, 5, 20, 10, 5, 10, 20, 5,
    )
    episode_epochs: int = 30
    decision_interval_epochs: int = 3
    maximum_topology_switches_per_episode: int = 5
    minimum_topology_dwell_decisions: int = 1
    top_k_candidate_neighbors: int = 3

    def validate(self) -> None:
        if not self.node_count_cycle:
            raise ValueError("Variable-scale node-count cycle cannot be empty.")
        if set(self.node_count_cycle) - {5, 10, 20}:
            raise ValueError("Variable-scale curriculum supports 5, 10, or 20 nodes.")
        if min(
            self.episode_epochs,
            self.decision_interval_epochs,
            self.maximum_topology_switches_per_episode,
            self.minimum_topology_dwell_decisions,
            self.top_k_candidate_neighbors,
        ) <= 0:
            raise ValueError("Variable-scale curriculum controls must be positive.")

    def node_count_for_condition(self, condition_seed: int) -> int:
        self.validate()
        return self.node_count_cycle[int(condition_seed) % len(self.node_count_cycle)]

    def configuration_for_condition(
        self, condition_seed: int, **changes,
    ) -> Stage1Configuration:
        """Build one reproducible scale-specific environment configuration."""

        node_count = self.node_count_for_condition(condition_seed)
        common = dict(
            node_count=node_count,
            episode_epochs=self.episode_epochs,
            decision_interval_epochs=self.decision_interval_epochs,
            minimum_topology_dwell_decisions=(
                self.minimum_topology_dwell_decisions
            ),
            maximum_topology_switches_per_episode=(
                self.maximum_topology_switches_per_episode
            ),
            top_k_candidate_neighbors=self.top_k_candidate_neighbors,
        )
        if node_count == 5:
            configuration = five_node_stratified_physical_configuration(
                **common,
            )
            configuration = replace(
                configuration,
                scenario_distribution=replace(
                    configuration.scenario_distribution,
                    dynamic_link_event_count=2,
                ),
            )
        else:
            configuration = Stage1Configuration(
                **common,
                scenario_type="walker_delta",
                walker_plane_count=1 if node_count == 10 else 5,
                walker_phasing=0 if node_count == 10 else 3,
                scenario_distribution=CompactFleetScenarioDistribution(
                    packet_loss_range=(0.0, 0.2),
                    communication_delay_range=(0.0, 2.0),
                    navigation_dropout_node_count=max(1, node_count // 10),
                    initial_topology_types=("chain", "ring", "star"),
                    link_condition_mode="undirected_independent",
                    dynamic_link_event_count=max(2, node_count // 4),
                ),
            )
        return replace(configuration, **changes)
