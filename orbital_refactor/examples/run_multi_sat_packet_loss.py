"""Random packet loss experiment."""

from __future__ import annotations

from examples.run_multi_sat_interface import build_demo_case
from cooperative.multi_sat_pipeline import run_cooperative_pipeline
from cooperative.communication_channel import CommunicationChannel


def main():
    scenario, observations, initial_errors, modality_config = build_demo_case()

    channel = CommunicationChannel(
        packet_loss_rate={
            "sat_01": 0.0,
            "sat_02": 0.2,
            "sat_03": 0.4,
        },
        random_seed=42,
    )

    result = run_cooperative_pipeline(
        scenario=scenario,
        observations_by_node=observations,
        initial_error_by_node=initial_errors,
        architecture="federated_ci",
        modality_config_by_node=modality_config,
        communication_channel=channel,
    )

    print("Packet loss experiment finished.")
    print(
        "Position RMSE:",
        result.metrics.cooperative_position_rmse,
    )
    print(
        "Velocity RMSE:",
        result.metrics.cooperative_velocity_rmse,
    )

    for t in [0, 100, 200, 300]:
        idx = abs(scenario.timestamps - t).argmin()
        print("=" * 30)
        print("time:", scenario.timestamps[idx])
        print(
            "received:",
            result.cooperative_history.received_node_history[idx],
        )


if __name__ == "__main__":
    main()
