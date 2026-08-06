"""Generate a basic three-observer cooperative orbital scenario.

This example focuses on truth trajectories and synthetic observations. It does
not require SHIRT images or neural-network checkpoints.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters.synthetic_measurement_adapter import (
    create_infrared_observations,
    create_nn_state_observations,
    create_radar_observations,
    visibility_flags,
)
from orbital_core.constants import R_EARTH
from orbital_core.orbit_elements import keplerian_to_eci
from scenarios.multi_satellite_scenario import generate_cooperative_scenario


def main() -> None:
    timestamps = np.arange(0.0, 301.0, 10.0)
    altitude = 700e3
    inclination = np.deg2rad(23.0)
    target_initial = keplerian_to_eci(
        R_EARTH + altitude, 0.001, inclination, 0.0, 0.0, 0.0
    )
    observer_initials = {
        "sat_01": keplerian_to_eci(
            R_EARTH + altitude, 0.001, inclination, 0.0, 0.0, np.deg2rad(0.05)
        ),
        "sat_02": keplerian_to_eci(
            R_EARTH + altitude + 1000.0, 0.001, inclination,
            np.deg2rad(0.02), 0.0, np.deg2rad(-0.05)
        ),
        "sat_03": keplerian_to_eci(
            R_EARTH + altitude - 1000.0, 0.001, inclination,
            np.deg2rad(-0.02), 0.0, np.deg2rad(0.08)
        ),
    }
    scenario = generate_cooperative_scenario(
        timestamps=timestamps,
        target_id="target_01",
        target_initial_state_eci=target_initial,
        observer_initial_states_eci=observer_initials,
    )

    modality_by_node = {
        "sat_01": ("ir", "nn"),
        "sat_02": ("ir", "rad"),
        "sat_03": ("rad", "nn"),
    }
    observations_by_node = {}
    for index, (node_id, modalities) in enumerate(modality_by_node.items(), start=1):
        rel_eci = scenario.relative_state_eci_by_node[node_id]
        rel_spri = scenario.relative_state_spri_by_node[node_id]
        rng = np.random.default_rng(100 + index)
        range_valid = visibility_flags(
            relative_position_spri=rel_spri[:, :3], max_range=2.0e6
        )
        observations = []
        if "ir" in modalities:
            observations += create_infrared_observations(
                timestamps=timestamps,
                relative_position_spri=rel_spri[:, :3],
                covariance=np.diag(np.deg2rad([0.02, 0.02])) ** 2,
                observer_id=node_id,
                target_id=scenario.target_id,
                rng=rng,
                valid_flags=range_valid,
            )
        if "rad" in modalities:
            observations += create_radar_observations(
                timestamps=timestamps,
                relative_position_spri=rel_spri[:, :3],
                relative_velocity_spri=rel_spri[:, 3:],
                covariance=np.diag([20.0, 0.05]) ** 2,
                observer_id=node_id,
                target_id=scenario.target_id,
                rng=rng,
                valid_flags=range_valid,
            )
        if "nn" in modalities:
            observations += create_nn_state_observations(
                timestamps=timestamps,
                relative_state_eci=rel_eci,
                covariance=np.diag([30.0, 30.0, 30.0, 0.1, 0.1, 0.1]) ** 2,
                observer_id=node_id,
                target_id=scenario.target_id,
                rng=rng,
                valid_flags=range_valid,
            )
        observations_by_node[node_id] = observations

    print(f"samples: {len(timestamps)}")
    for node_id in observer_initials:
        ranges = np.linalg.norm(scenario.relative_state_eci_by_node[node_id][:, :3], axis=1)
        print(
            f"{node_id}: range={ranges.min()/1e3:.2f}..{ranges.max()/1e3:.2f} km, "
            f"observations={len(observations_by_node[node_id])}"
        )


if __name__ == "__main__":
    main()
