from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

from experiments.topology_control_environment import TopologyControlEnvironment
from experiments.topology_reward_calibration import (
    POLICIES,
    RewardCostWeights,
    run_reward_calibration_scan,
    summarize_reward_calibration,
)
from scenarios.measurement_visibility import VisibilityConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or resume the V15 topology reward calibration scan.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/v15_reward_calibration.csv"),
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--switch-weights", default="0,0.005")
    parser.add_argument("--resync-weights", default="0,0.002")
    parser.add_argument("--communication-weights", default="0")
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--nodes", type=int, choices=(3, 5, 20), default=5)
    parser.add_argument(
        "--scenario", choices=("compact_fleet", "walker_20_5_3"),
        default="compact_fleet",
    )
    parser.add_argument("--walker-maximum-range", type=float, default=7000e3)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--decision-interval", type=int, default=1)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--modalities", default="RANGE")
    parser.add_argument("--packet-loss", type=float, default=0.1)
    parser.add_argument("--communication-delay", type=float, default=1.0)
    parser.add_argument("--oracle-lookahead", type=int, default=1)
    parser.add_argument("--minimum-topology-dwell", type=int, default=0)
    parser.add_argument("--top-k-candidate-neighbors", type=int, default=None)
    parser.add_argument(
        "--maximum-measurement-range", type=float, default=None,
        help="Enable physical visibility gating with this maximum range.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = _csv_values(args.seeds, int, "seeds")
    policies = _csv_values(args.policies, str, "policies")
    unknown = set(policies) - set(POLICIES)
    if unknown:
        raise ValueError(f"Unknown policies: {sorted(unknown)}")
    modalities = _csv_values(args.modalities, str, "modalities")
    visibility = (
        None if args.maximum_measurement_range is None else {
            modality: VisibilityConfig(
                maximum_range=args.maximum_measurement_range
            )
            for modality in modalities
        }
    )
    weight_grid = tuple(
        RewardCostWeights(communication, switch, resynchronization)
        for communication in _csv_values(
            args.communication_weights, float, "communication weights"
        )
        for switch in _csv_values(args.switch_weights, float, "switch weights")
        for resynchronization in _csv_values(
            args.resync_weights, float, "resynchronization weights"
        )
    )
    environment_factory = partial(
        TopologyControlEnvironment,
        node_count=args.nodes, episode_epochs=args.epochs,
        decision_interval_epochs=args.decision_interval, dt=args.dt,
        relative_modalities=modalities, packet_loss=args.packet_loss,
        communication_delay=args.communication_delay,
        visibility_by_modality=visibility,
        minimum_topology_dwell_decisions=args.minimum_topology_dwell,
        top_k_candidate_neighbors=args.top_k_candidate_neighbors,
        scenario_type=args.scenario,
        walker_maximum_range=args.walker_maximum_range,
    )
    configuration_id = json.dumps({
        "nodes": args.nodes, "epochs": args.epochs,
        "decision_interval": args.decision_interval, "dt": args.dt,
        "modalities": modalities, "packet_loss": args.packet_loss,
        "communication_delay": args.communication_delay,
        "maximum_measurement_range": args.maximum_measurement_range,
        "minimum_topology_dwell": args.minimum_topology_dwell,
        "top_k_candidate_neighbors": args.top_k_candidate_neighbors,
        "scenario": args.scenario,
        "walker_maximum_range": args.walker_maximum_range,
    }, sort_keys=True, separators=(",", ":"))
    records = run_reward_calibration_scan(
        args.output, seeds=seeds, weight_grid=weight_grid,
        environment_factory=environment_factory, policies=policies,
        oracle_lookahead_steps=args.oracle_lookahead,
        configuration_id=configuration_id,
    )
    print(f"records={len(records)} output={args.output}")
    for summary in summarize_reward_calibration(records):
        weights = summary.weights
        print(
            f"policy={summary.policy} "
            f"configuration={summary.configuration_id} "
            f"lookahead={summary.oracle_lookahead_steps} "
            f"n={summary.sample_count} "
            f"communication={weights.communication:g} "
            f"switch={weights.topology_switch:g} "
            f"resync={weights.resynchronization:g} "
            f"rmse={summary.mean_final_position_rmse:.6f} "
            f"return={summary.mean_penalized_return:.6f} "
            f"switches={summary.mean_topology_switch_count:.2f} "
            f"resyncs={summary.mean_resynchronization_count:.2f}"
        )
    return 0


def _csv_values(text, converter, label):
    fields = tuple(value.strip() for value in text.split(",") if value.strip())
    if not fields:
        raise ValueError(f"At least one {label} value is required.")
    return tuple(converter(value) for value in fields)


if __name__ == "__main__":
    raise SystemExit(main())
