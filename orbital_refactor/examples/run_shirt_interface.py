"""Run the standard interface with externally supplied SHIRT files."""
from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
from time import perf_counter

from adapters import (
    build_shirt_module_input,
    create_infrared_observations,
    create_nn_observations,
    create_radar_observations,
    load_aligned_nn_positions,
    load_shirt_orbit_dataset,
)
from interfaces.state_awareness_module import StateAwarenessModule


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--orbit", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--scenario-key", default="roe2")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    dataset = load_shirt_orbit_dataset(args.metadata, args.orbit, args.scenario_key)
    nn_position, nn_valid = load_aligned_nn_positions(
        args.predictions, dataset.filenames
    )

    observations = create_nn_observations(
        timestamps=dataset.timestamps,
        positions=nn_position,
        valid_positions=nn_valid,
        covariance_position=np.diag(np.array([0.1, 0.1, 0.1]) ** 2),
        covariance_velocity=np.diag(np.array([0.1, 0.1, 0.1]) ** 2),
        observer_id="sat_01",
        target_id="target_01",
        frame="SPRI",
        use_pseudo_velocity=True,
    )
    rng = np.random.default_rng(42)
    observations += create_infrared_observations(
        timestamps=dataset.timestamps,
        relative_position_spri=dataset.relative_position_spri,
        covariance=np.diag(np.deg2rad(np.array([1.5, 1.5])) ** 2),
        observer_id="sat_01",
        target_id="target_01",
        rng=rng,
    )
    observations += create_radar_observations(
        timestamps=dataset.timestamps,
        relative_position_spri=dataset.relative_position_spri,
        relative_velocity_spri=dataset.relative_velocity_spri,
        covariance=np.diag(np.array([1.0, 0.02]) ** 2),
        observer_id="sat_01",
        target_id="target_01",
        rng=rng,
    )

    module_input = build_shirt_module_input(
        dataset,
        node_id="sat_01",
        target_id="target_01",
        process_noise_acceleration=1e-4,
        initial_position_std=10.0,
        initial_velocity_std=0.05,
        observations=observations,
        filter_config={
            "reset_feedback": True,
            "ci_objective": "trace",
            "ci_grid_points": 41,
            "gate_enable": True,
            "gate_mode": "soft",
            "soft_scale": 20.0,
        },
        modalities_config={
            "nn": {
                "nn_meas_frame": "spri",
                "nn_use_pseudo_velocity": True,
                "gate_threshold": 16.0,
            },
            "infrared": {"gate_threshold": 16.0},
            "radar": {"gate_threshold": 16.0},
        },
    )
    start = perf_counter()
    output = StateAwarenessModule().run(module_input)
    elapsed = perf_counter() - start
    print("position:", output.state_output.position_estimate)
    print("velocity:", output.state_output.velocity_estimate)
    print("acceleration:", output.state_output.acceleration_estimate)
    print("modality weights:", output.fusion_status.modality_weights)
    print(f"wall-clock filter time: {elapsed:.3f} s")
    print(f"reported processing time: {output.runtime_status.processing_time:.3f} s")


if __name__ == "__main__":
    main()
