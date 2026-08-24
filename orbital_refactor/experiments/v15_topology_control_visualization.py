from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.topology_control_baselines import (
    AlwaysKeepPolicy,
    EnvironmentPolicy,
)


@dataclass(frozen=True)
class TopologyControlTrace:
    policy_name: str
    times: np.ndarray
    position_rmse: np.ndarray
    position_three_sigma: np.ndarray
    active_edge_count: np.ndarray
    action_kinds: tuple[str, ...]
    cumulative_transmitted: np.ndarray
    cumulative_dropped: np.ndarray
    cumulative_replay: np.ndarray
    cumulative_resynchronization: np.ndarray
    truth_by_node: dict[str, np.ndarray]
    initial_edges: tuple[tuple[str, str], ...]
    final_edges: tuple[tuple[str, str], ...]


def generate_v15_topology_control_visualization(
    output_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    trace_bundle_path: str | Path | None = None,
    condition_seed: int = 76,
    noise_seed: int = 0,
    duration: float = 120.0,
    dt: float = 2.0,
) -> Path:
    """Compare the frozen LCB reference with keep in one V15 episode."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if (checkpoint_path is None) == (trace_bundle_path is None):
        raise ValueError("Provide exactly one checkpoint or trace bundle.")
    if trace_bundle_path is not None:
        keep, reference, metadata = load_topology_control_trace_bundle(
            trace_bundle_path
        )
        condition_seed = int(metadata["condition_seed"])
        noise_seed = int(metadata["noise_seed"])
    else:
        keep, reference = collect_v15_topology_control_trace_bundle(
            checkpoint_path=checkpoint_path,
            condition_seed=condition_seed, noise_seed=noise_seed,
            duration=duration, dt=dt,
        )

    figure = plt.figure(figsize=(14, 9), constrained_layout=True)
    orbit_axis = figure.add_subplot(2, 2, 1, projection="3d")
    error_axis = figure.add_subplot(2, 2, 2)
    topology_axis = figure.add_subplot(2, 2, 3)
    resource_axis = figure.add_subplot(2, 2, 4)

    _plot_truth_and_topologies(orbit_axis, reference)
    error_axis.plot(
        keep.times, keep.position_rmse, label="always-keep RMSE",
        color="tab:gray",
    )
    error_axis.plot(
        reference.times, reference.position_rmse, label="LCB reference RMSE",
        color="tab:blue",
    )
    error_axis.plot(
        reference.times, reference.position_three_sigma, "--",
        label="LCB equivalent 3-sigma", color="tab:orange",
    )
    error_axis.set_title("Fleet position estimation")
    error_axis.set_xlabel("time (s)")
    error_axis.set_ylabel("position component RMSE / bound (m)")
    error_axis.grid(alpha=0.3)
    error_axis.legend()

    topology_axis.step(
        reference.times, reference.active_edge_count, where="post",
        label="active edge count", color="tab:purple",
    )
    kind_level = {"keep": 0, "add": 1, "swap": 2, "remove": 3}
    action_times = reference.times[1:1 + len(reference.action_kinds)]
    action_levels = [kind_level[kind] for kind in reference.action_kinds]
    action_axis = topology_axis.twinx()
    action_axis.scatter(
        action_times, action_levels, s=22, color="tab:red", label="action type",
    )
    action_axis.set_yticks(tuple(kind_level.values()), tuple(kind_level))
    topology_axis.set_title("LCB topology decisions")
    topology_axis.set_xlabel("time (s)")
    topology_axis.set_ylabel("active undirected edges")
    action_axis.set_ylabel("executed action")
    topology_axis.grid(alpha=0.3)
    lines, labels = topology_axis.get_legend_handles_labels()
    extra_lines, extra_labels = action_axis.get_legend_handles_labels()
    topology_axis.legend(lines + extra_lines, labels + extra_labels, loc="best")

    resource_axis.plot(
        reference.times, reference.cumulative_transmitted,
        label="transmitted messages",
    )
    resource_axis.plot(
        reference.times, reference.cumulative_dropped,
        label="dropped messages",
    )
    resource_axis.plot(
        reference.times, reference.cumulative_replay, label="replays",
    )
    resource_axis.plot(
        reference.times, reference.cumulative_resynchronization,
        label="resynchronizations",
    )
    resource_axis.set_title("LCB communication and recovery cost")
    resource_axis.set_xlabel("time (s)")
    resource_axis.set_ylabel("cumulative count")
    resource_axis.grid(alpha=0.3)
    resource_axis.legend()

    figure.suptitle(
        "V15 five-node topology control | heterogeneous links | "
        f"condition {condition_seed} | noise {noise_seed}",
        fontsize=14,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=160)
    plt.close(figure)
    return target


def collect_v15_topology_control_trace_bundle(
    *,
    checkpoint_path: str | Path,
    condition_seed: int = 76,
    noise_seed: int = 0,
    duration: float = 120.0,
    dt: float = 2.0,
):
    from experiments.topology_control_baselines import HierarchicalGNNPolicy
    from experiments.topology_ppo_stage1 import (
        build_stage1_environment,
        five_node_heterogeneous_link_configuration,
    )

    episode_epochs = max(2, int(round(duration / dt)))
    configuration = five_node_heterogeneous_link_configuration(
        episode_epochs=episode_epochs,
        decision_interval_epochs=2,
        maximum_topology_switches_per_episode=1,
    )
    keep = collect_topology_control_trace(
        build_stage1_environment(configuration), AlwaysKeepPolicy(),
        condition_seed=condition_seed, noise_seed=noise_seed,
    )
    reference = collect_topology_control_trace(
        build_stage1_environment(configuration),
        HierarchicalGNNPolicy(checkpoint_path),
        condition_seed=condition_seed, noise_seed=noise_seed,
    )
    return keep, reference


def save_topology_control_trace_bundle(
    output_path: str | Path,
    keep: TopologyControlTrace,
    reference: TopologyControlTrace,
    *,
    condition_seed: int,
    noise_seed: int,
) -> Path:
    """Save two visualization traces without pickle-backed object arrays."""

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    manifest = {
        "schema_version": "v15.0-topology-control-visualization",
        "condition_seed": int(condition_seed), "noise_seed": int(noise_seed),
        "traces": {},
    }
    for prefix, trace in (("keep", keep), ("reference", reference)):
        nodes = tuple(trace.truth_by_node)
        manifest["traces"][prefix] = {
            "policy_name": trace.policy_name, "nodes": nodes,
            "initial_edges": trace.initial_edges, "final_edges": trace.final_edges,
            "action_kinds": trace.action_kinds,
        }
        for name in (
            "times", "position_rmse", "position_three_sigma",
            "active_edge_count", "cumulative_transmitted",
            "cumulative_dropped", "cumulative_replay",
            "cumulative_resynchronization",
        ):
            arrays[f"{prefix}_{name}"] = getattr(trace, name)
        for index, node in enumerate(nodes):
            arrays[f"{prefix}_truth_{index:03d}"] = trace.truth_by_node[node]
    arrays["manifest"] = np.asarray(json.dumps(manifest))
    np.savez_compressed(target, **arrays)
    return target


def load_topology_control_trace_bundle(input_path: str | Path):
    """Load a pickle-free visualization trace bundle."""

    with np.load(Path(input_path), allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest"]))
        traces = []
        for prefix in ("keep", "reference"):
            metadata = manifest["traces"][prefix]
            arrays = {
                name: np.array(archive[f"{prefix}_{name}"], copy=True)
                for name in (
                    "times", "position_rmse", "position_three_sigma",
                    "active_edge_count", "cumulative_transmitted",
                    "cumulative_dropped", "cumulative_replay",
                    "cumulative_resynchronization",
                )
            }
            truth = {
                node: np.array(archive[f"{prefix}_truth_{index:03d}"], copy=True)
                for index, node in enumerate(metadata["nodes"])
            }
            traces.append(TopologyControlTrace(
                policy_name=metadata["policy_name"], truth_by_node=truth,
                action_kinds=tuple(metadata["action_kinds"]),
                initial_edges=tuple(tuple(edge) for edge in metadata["initial_edges"]),
                final_edges=tuple(tuple(edge) for edge in metadata["final_edges"]),
                **arrays,
            ))
    return traces[0], traces[1], manifest


def collect_topology_control_trace(
    environment,
    policy: EnvironmentPolicy,
    *,
    condition_seed: int,
    noise_seed: int,
) -> TopologyControlTrace:
    """Run one deterministic policy and retain filter/topology diagnostics."""

    state = environment.reset(seed=noise_seed, condition_seed=condition_seed)
    initial_edges = tuple(state.observation.previous_active_edges)
    times = [float(environment._case["timestamps"][environment._epoch_index])]
    rmse = [environment._metrics()[0]]
    bounds = [_position_three_sigma(environment)]
    edge_counts = [len(initial_edges)]
    action_kinds = []
    cumulative = np.zeros(4, dtype=float)
    transmitted = [0.0]
    dropped = [0.0]
    replay = [0.0]
    resynchronization = [0.0]
    while True:
        step = environment.step(policy.select_action(state))
        state = step.state
        action_kinds.append(step.action_resolution.executed_action.kind)
        cumulative += np.asarray((
            step.constraint_costs.transmitted_messages,
            step.constraint_costs.dropped_messages,
            step.constraint_costs.replay_count,
            step.constraint_costs.resynchronization_count,
        ))
        times.append(float(environment._case["timestamps"][environment._epoch_index]))
        rmse.append(environment._metrics()[0])
        bounds.append(_position_three_sigma(environment))
        edge_counts.append(len(state.observation.previous_active_edges))
        transmitted.append(cumulative[0])
        dropped.append(cumulative[1])
        replay.append(cumulative[2])
        resynchronization.append(cumulative[3])
        if step.terminated or step.truncated:
            break
    truth = {
        node: np.asarray(values, dtype=float)
        for node, values in environment._case["truth"].items()
    }
    return TopologyControlTrace(
        policy_name=policy.name,
        times=np.asarray(times), position_rmse=np.asarray(rmse),
        position_three_sigma=np.asarray(bounds),
        active_edge_count=np.asarray(edge_counts),
        action_kinds=tuple(action_kinds),
        cumulative_transmitted=np.asarray(transmitted),
        cumulative_dropped=np.asarray(dropped),
        cumulative_replay=np.asarray(replay),
        cumulative_resynchronization=np.asarray(resynchronization),
        truth_by_node=truth, initial_edges=initial_edges,
        final_edges=tuple(state.observation.previous_active_edges),
    )


def _position_three_sigma(environment):
    traces = [
        np.trace(session.state.active_covariance[:3, :3]) / 3.0
        for session in environment._orchestrator.sessions.values()
    ]
    return float(3.0 * np.sqrt(np.mean(traces)))


def _plot_truth_and_topologies(axis, trace):
    colors = {}
    for index, (node, history) in enumerate(trace.truth_by_node.items()):
        position = history[:, :3] / 1000.0
        line, = axis.plot(
            position[:, 0], position[:, 1], position[:, 2], linewidth=1.0,
            label=node,
        )
        colors[node] = line.get_color()
        axis.scatter(*position[0], s=12, color=line.get_color())
    initial_positions = {
        node: history[0, :3] / 1000.0
        for node, history in trace.truth_by_node.items()
    }
    final_positions = {
        node: history[-1, :3] / 1000.0
        for node, history in trace.truth_by_node.items()
    }
    for left, right in trace.initial_edges:
        points = np.vstack((initial_positions[left], initial_positions[right]))
        axis.plot(*points.T, color="tab:gray", linestyle=":", linewidth=0.8)
    for left, right in trace.final_edges:
        points = np.vstack((final_positions[left], final_positions[right]))
        axis.plot(*points.T, color="black", linewidth=1.0)
    axis.set_title("Truth trajectories; dotted initial / solid final topology")
    axis.set_xlabel("ECI x (km)")
    axis.set_ylabel("ECI y (km)")
    axis.set_zlabel("ECI z (km)")
    axis.legend(fontsize=7, loc="best")
