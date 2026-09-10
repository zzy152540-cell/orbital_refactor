from __future__ import annotations

import argparse
from pathlib import Path

from cooperative.network_schmidt_runner import run_network_schmidt_filter
from experiments.v14_walker_geometry_audit import run_v14_walker_geometry_audit
from experiments.walker_filter_setup import build_walker_filter_case
from experiments.v15_visualization_raw_frontend import (
    replace_selected_link_with_raw_frontends,
)
from visualization.network_history_adapter import network_history_visualization_frames
from visualization.cann_history_adapter import build_cann_visualization_history
from visualization.recording import VisualizationRecordingWriter


def generate_walker_visualization_recording(
    output: str | Path, *, duration: float = 20.0, dt: float = 2.0,
    seed: int = 0, maximum_range: float = 6000e3,
    raw_capture_link: tuple[str, str] | None = None,
    include_cann: bool = False,
) -> Path:
    """Generate a short Walker-20 replay without modifying the filter runner."""
    audit = run_v14_walker_geometry_audit(
        total_satellites=20, plane_count=10, phasing=1,
        duration=1800.0, dt=30.0, maximum_range=maximum_range,
    )
    case = build_walker_filter_case(
        seed=seed, duration=duration, dt=dt, maximum_range=maximum_range,
        topology=audit.persistent_topology,
        truth_history_by_node=audit.scenario.truth_state_history_by_node,
        topology_type="walker_persistent",
    )
    observations = case["observations"]
    raw_sensor_data = None
    if raw_capture_link is not None:
        observations, raw_sensor_data = replace_selected_link_with_raw_frontends(
            observations, timestamps=case["timestamps"],
            truth_state_history_by_node=case["truth"],
            capture_link=raw_capture_link, seed=seed,
        )
    history = run_network_schmidt_filter(
        timestamps=case["timestamps"],
        initial_state_by_node=case["initial_states"],
        initial_covariance_by_node=case["initial_covariances"],
        topology=case["topology"],
        observation_messages=observations,
        absolute_position_observations=case["absolute_observations"],
        observation_usage="observer_only",
        process_noise_acceleration=1e-8,
        consider_refresh_mode="exact_transport_event_replay",
        state_messages_by_receiver=case["state_messages"],
        replay_history_window=10.0,
        expected_lineage_by_link=case["lineages"],
    )
    cann_history = None
    if include_cann:
        cann_history = build_cann_visualization_history(
            history=history, initial_state_by_node=case["initial_states"],
            absolute_position_observations=case["absolute_observations"],
        )
    frames = network_history_visualization_frames(
        history=history, truth_history_by_node=case["truth"],
        topology=case["topology"],
        observation_messages=observations,
        absolute_position_observations=case["absolute_observations"],
        raw_sensor_data_by_information_id=raw_sensor_data,
        navigation_by_epoch=(
            None if cann_history is None else cann_history.navigation_by_epoch
        ),
        cann_by_epoch=(
            None if cann_history is None else cann_history.cann_by_epoch
        ),
        scenario_id="walker-20-10-1", run_id=f"seed-{seed}",
    )
    target = Path(output)
    with VisualizationRecordingWriter(target) as writer:
        for frame in frames:
            writer.append(frame)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/visualization_recordings/walker20_20s_seed0"),
    )
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-observer")
    parser.add_argument("--raw-target")
    parser.add_argument("--include-cann", action="store_true")
    args = parser.parse_args()
    if (args.raw_observer is None) != (args.raw_target is None):
        parser.error("--raw-observer and --raw-target must be supplied together")
    raw_link = (
        None if args.raw_observer is None
        else (args.raw_observer, args.raw_target)
    )
    output = generate_walker_visualization_recording(
        args.output, duration=args.duration, dt=args.dt, seed=args.seed,
        raw_capture_link=raw_link,
        include_cann=args.include_cann,
    )
    print(output)


if __name__ == "__main__":
    main()
