"""Fixed communication delay experiment."""

from __future__ import annotations

from __future__ import annotations

import numpy as np

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from examples.run_multi_sat_interface import build_demo_case
from cooperative.multi_sat_pipeline import run_cooperative_pipeline
from cooperative.delay_channel import DelayChannel


def main():
    scenario, observations, initial_errors, modality_config = build_demo_case()

    channel = DelayChannel(
        delay_by_node={
            "sat_01": 0.0,
            "sat_02": 5.0,
            "sat_03": 10.0,
        }
    )

    print("Delay configuration:")
    for node, delay in channel.delay_by_node.items():
        print(f"  {node}: {delay:.1f} s")

    result = run_cooperative_pipeline(
        scenario=scenario,
        observations_by_node=observations,
        initial_error_by_node=initial_errors,
        architecture="federated_ci",
        modality_config_by_node=modality_config,
        delay_channel=channel,
        age_aware=True,
        age_penalty=1e-2,
    )

    print("Delay experiment finished.")
    print("Position RMSE:", result.metrics.cooperative_position_rmse)
    print("Velocity RMSE:", result.metrics.cooperative_velocity_rmse)

    print("\nReceived node history:")
    for t in [0, 100, 200, 300]:
        idx = int(min(range(len(scenario.timestamps)),
                      key=lambda i: abs(scenario.timestamps[i]-t)))
        print(f"time={scenario.timestamps[idx]:.1f}s")
        print(result.cooperative_history.received_node_history[idx])

    print("\nNode timestamp example:")
    for node, delay in channel.delay_by_node.items():
        print(f"{node}: source=100.0, arrival={100.0+delay:.1f}")


if __name__ == "__main__":
    main()
