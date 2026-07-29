"""Node dropout robustness experiment."""

from __future__ import annotations

import numpy as np

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from cooperative.communication import (
    CommunicationConfig,
    build_validity_history,
)
from examples.run_multi_sat_interface import build_demo_case
from cooperative.multi_sat_pipeline import run_cooperative_pipeline

from visualization.plot_dropout_results import (
    plot_dropout_status,
    plot_ci_weights,
)


def compute_rmse_segment(
    error,
    timestamps,
    start,
    end,
):
    """Compute RMSE in a specified time interval."""
    mask = (timestamps >= start) & (timestamps < end)

    if not np.any(mask):
        return float("nan")

    return float(
        np.sqrt(np.mean(error[mask] ** 2))
    )


def main():
    scenario, observations, initial_errors, modality_config = build_demo_case()

    communication = CommunicationConfig(
        node_dropout_windows={
            "sat_02": [(100.0, 200.0)]
        }
    )

    validity = build_validity_history(
        list(scenario.observer_trajectories.keys()),
        scenario.timestamps,
        communication,
    )

    result = run_cooperative_pipeline(
        scenario=scenario,
        observations_by_node=observations,
        initial_error_by_node=initial_errors,
        initial_covariance=np.diag(
            [150.0,150.0,150.0,0.3,0.3,0.3]
        ) ** 2,
        architecture="federated_ci",
        modality_config_by_node=modality_config,
        node_validity_by_node=validity,
    )

    truth = scenario.target_trajectory.state_history_eci

    estimate = (
        result
        .cooperative_history
        .state_history_eci
    )


    position_error = np.linalg.norm(
        estimate[:, :3] - truth[:, :3],
        axis=1,
    )

    velocity_error = np.linalg.norm(
        estimate[:, 3:] - truth[:, 3:],
        axis=1,
    )


    print("\nStage RMSE:")

    for name, start, end in [
        ("normal", 0, 100),
        ("dropout", 100, 200),
        ("recovery", 200, 300),
    ]:
        pos_rmse = compute_rmse_segment(
            position_error,
            scenario.timestamps,
            start,
            end,
        )

        vel_rmse = compute_rmse_segment(
            velocity_error,
            scenario.timestamps,
            start,
            end,
        )

        print(
            f"{name:10s}: "
            f"position={pos_rmse:.3f} m, "
            f"velocity={vel_rmse:.6f} m/s"
        )


    position_rmse = np.sqrt(
        np.mean(position_error**2)
    )

    velocity_rmse = np.sqrt(
        np.mean(velocity_error**2)
    )


    print("="*40)

    print(
        "Cooperative position RMSE:",
        f"{position_rmse:.3f} m"
    )

    print(
        "Cooperative velocity RMSE:",
        f"{velocity_rmse:.6f} m/s"
    )

    active_count = []

    for nodes in result.cooperative_history.active_node_history:
        active_count.append(len(nodes))


    print(
    "Active node count:"
    )

    for t in [0,100,200,300]:

        idx=np.argmin(
            abs(scenario.timestamps-t)
        )

        print(
            scenario.timestamps[idx],
            len(
                result.cooperative_history.active_node_history[idx]
            )
        )

    for k in [0, 100, 150, 200, 300]:

        idx = np.argmin(
            abs(scenario.timestamps-k)
        )

        print("="*40)

        print(
            "time:",
            scenario.timestamps[idx],
        )

        print(
            "active nodes:",
            result.cooperative_history.active_node_history[idx]
        )

        print(
            "CI weights:",
            result.cooperative_history.node_weight_history[idx]
        )

    plot_dropout_status(
    scenario.timestamps,
    result.cooperative_history.active_node_history,
    dropout_windows=[(100,200)]
    )

    plot_ci_weights(
        scenario.timestamps,
        result.cooperative_history.node_weight_history,
    )

    plot_dropout_status(
        scenario.timestamps,
        result.cooperative_history.active_node_history,
        dropout_windows=[(100,200)],
    )

if __name__ == "__main__":
    main()
