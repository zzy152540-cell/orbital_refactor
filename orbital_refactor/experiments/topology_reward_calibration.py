from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from experiments.topology_control_baselines import (
    AlwaysKeepPolicy,
    InformationGreedyPolicy,
    ShortHorizonOraclePolicy,
    run_topology_control_baseline_episode,
)
from experiments.topology_control_environment import TopologyControlEnvironment


@dataclass(frozen=True)
class RewardCostWeights:
    communication: float = 0.0
    topology_switch: float = 0.0
    resynchronization: float = 0.0

    def __post_init__(self) -> None:
        if min(asdict(self).values()) < 0.0:
            raise ValueError("Reward cost weights cannot be negative.")


@dataclass(frozen=True)
class RewardCalibrationRecord:
    seed: int
    policy: str
    configuration_id: str
    oracle_lookahead_steps: int
    communication_weight: float
    topology_switch_weight: float
    resynchronization_weight: float
    final_position_rmse: float
    cumulative_task_reward: float
    cumulative_penalized_return: float
    transmitted_messages: float
    dropped_messages: float
    replay_count: float
    resynchronization_count: float
    topology_switch_count: float

    @property
    def key(self) -> tuple[int, str, str, int, float, float, float]:
        return (
            self.seed, self.policy, self.configuration_id,
            self.oracle_lookahead_steps,
            self.communication_weight,
            self.topology_switch_weight, self.resynchronization_weight,
        )


@dataclass(frozen=True)
class RewardCalibrationSummary:
    policy: str
    configuration_id: str
    oracle_lookahead_steps: int
    weights: RewardCostWeights
    sample_count: int
    mean_final_position_rmse: float
    mean_penalized_return: float
    mean_topology_switch_count: float
    mean_resynchronization_count: float


POLICIES = ("keep", "information_greedy", "short_horizon_oracle")


def run_reward_calibration_scan(
    output_path: str | Path, *, seeds: Iterable[int],
    weight_grid: Iterable[RewardCostWeights],
    environment_factory: Callable[[], TopologyControlEnvironment],
    policies: Iterable[str] = POLICIES,
    oracle_lookahead_steps: int = 1,
    configuration_id: str = "unspecified",
) -> tuple[RewardCalibrationRecord, ...]:
    """Run a resumable calibration scan, flushing each completed cell."""

    seeds = tuple(int(seed) for seed in seeds)
    weight_grid = tuple(weight_grid)
    policies = tuple(policies)
    if oracle_lookahead_steps < 1:
        raise ValueError("Oracle lookahead must be at least one step.")
    if not configuration_id:
        raise ValueError("Calibration configuration ID cannot be empty.")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _validate_append_schema(path)
    existing = list(load_reward_calibration_records(path))
    completed = {record.key for record in existing}
    fieldnames = tuple(RewardCalibrationRecord.__dataclass_fields__)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for weights in weight_grid:
            for seed in seeds:
                for policy_name in policies:
                    key = _key(
                        seed, policy_name, configuration_id,
                        oracle_lookahead_steps, weights,
                    )
                    if key in completed:
                        continue
                    record = _run_cell(
                        environment_factory, seed, policy_name, weights,
                        oracle_lookahead_steps, configuration_id,
                    )
                    writer.writerow(asdict(record))
                    handle.flush()
                    existing.append(record)
                    completed.add(record.key)
    return tuple(existing)


def load_reward_calibration_records(
    path: str | Path,
) -> tuple[RewardCalibrationRecord, ...]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return ()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return tuple(_record_from_row(row) for row in csv.DictReader(handle))


def summarize_reward_calibration(
    records: Iterable[RewardCalibrationRecord],
) -> tuple[RewardCalibrationSummary, ...]:
    groups = {}
    for record in records:
        group = (
            record.policy, record.configuration_id,
            record.oracle_lookahead_steps,
            record.communication_weight,
            record.topology_switch_weight, record.resynchronization_weight,
        )
        groups.setdefault(group, []).append(record)
    summaries = []
    for group, values in sorted(groups.items()):
        (
            policy, configuration_id, lookahead, communication, switch,
            resynchronization,
        ) = group
        summaries.append(RewardCalibrationSummary(
            policy=policy, configuration_id=configuration_id,
            oracle_lookahead_steps=lookahead,
            weights=RewardCostWeights(communication, switch, resynchronization),
            sample_count=len(values),
            mean_final_position_rmse=float(np.mean([
                value.final_position_rmse for value in values
            ])),
            mean_penalized_return=float(np.mean([
                value.cumulative_penalized_return for value in values
            ])),
            mean_topology_switch_count=float(np.mean([
                value.topology_switch_count for value in values
            ])),
            mean_resynchronization_count=float(np.mean([
                value.resynchronization_count for value in values
            ])),
        ))
    return tuple(summaries)


def _run_cell(
    environment_factory, seed, policy_name, weights, oracle_lookahead_steps,
    configuration_id,
):
    environment = environment_factory()
    if policy_name == "keep":
        policy = AlwaysKeepPolicy()
    elif policy_name == "information_greedy":
        policy = InformationGreedyPolicy()
    elif policy_name == "short_horizon_oracle":
        policy = ShortHorizonOraclePolicy(
            environment, lookahead_steps=oracle_lookahead_steps,
            communication_cost_weight=weights.communication,
            switch_cost_weight=weights.topology_switch,
            resynchronization_cost_weight=weights.resynchronization,
        )
    else:
        raise ValueError(f"Unknown calibration policy: {policy_name}")
    result = run_topology_control_baseline_episode(
        environment, policy, seed=seed,
    )
    costs = result.cumulative_costs
    penalized = (
        result.cumulative_reward
        - weights.communication * costs.transmitted_messages
        - weights.topology_switch * costs.topology_switch
        - weights.resynchronization * costs.resynchronization_count
    )
    return RewardCalibrationRecord(
        seed=seed, policy=policy_name, configuration_id=configuration_id,
        oracle_lookahead_steps=oracle_lookahead_steps,
        communication_weight=weights.communication,
        topology_switch_weight=weights.topology_switch,
        resynchronization_weight=weights.resynchronization,
        final_position_rmse=result.final_position_rmse,
        cumulative_task_reward=result.cumulative_reward,
        cumulative_penalized_return=penalized,
        transmitted_messages=costs.transmitted_messages,
        dropped_messages=costs.dropped_messages,
        replay_count=costs.replay_count,
        resynchronization_count=costs.resynchronization_count,
        topology_switch_count=costs.topology_switch,
    )


def _key(seed, policy, configuration_id, oracle_lookahead_steps, weights):
    return (
        int(seed), policy, configuration_id, int(oracle_lookahead_steps),
        weights.communication,
        weights.topology_switch, weights.resynchronization,
    )


def _record_from_row(row):
    return RewardCalibrationRecord(
        seed=int(row["seed"]), policy=row["policy"],
        configuration_id=row.get("configuration_id", "legacy-unspecified"),
        oracle_lookahead_steps=int(row.get("oracle_lookahead_steps", 1)),
        **{
            name: float(row[name])
            for name in RewardCalibrationRecord.__dataclass_fields__
            if name not in {
                "seed", "policy", "configuration_id",
                "oracle_lookahead_steps",
            }
        },
    )


def _validate_append_schema(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open(newline="", encoding="utf-8-sig") as handle:
        fields = tuple(csv.DictReader(handle).fieldnames or ())
    required = tuple(RewardCalibrationRecord.__dataclass_fields__)
    if fields != required:
        raise ValueError(
            "Existing calibration CSV uses an incompatible schema; choose a "
            "new output path. Legacy files remain readable for analysis."
        )
