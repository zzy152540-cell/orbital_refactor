"""Minimal example of the documented ModuleInput -> ModuleOutput entry point."""

import numpy as np

from interfaces.data_objects import InitialState, ModuleInput, Observation
from interfaces.state_awareness_module import StateAwarenessModule
from orbital_core.dynamics import make_process_noise


def build_demo_input() -> ModuleInput:
    timestamps = np.array([0.0, 1.0, 2.0])
    chief_history = np.array(
        [
            [7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0],
            [7.0e6, 7500.0, 0.0, -8.0, 7500.0, 0.0],
            [7.0e6 - 8.0, 15000.0, 0.0, -16.0, 7499.99, 0.0],
        ]
    )
    q_history = np.tile([1.0, 0.0, 0.0, 0.0], (len(timestamps), 1))
    observations: list[Observation] = []
    for index, timestamp in enumerate(timestamps):
        observations.append(
            Observation(
                timestamp=float(timestamp),
                observer_id="sat_01",
                target_id="target_01",
                modality="OPTICAL",
                source_type="LEARNING",
                measurement=np.array([100.0 + 0.1 * index, 50.0, 500.0]),
                covariance=np.diag([9.0, 9.0, 16.0]),
                confidence=0.9,
                frame="ECI",
                valid_flag=True,
            )
        )
        observations.append(
            Observation(
                timestamp=float(timestamp),
                observer_id="sat_01",
                target_id="target_01",
                modality="RADAR",
                source_type="TRADITIONAL",
                measurement=np.array([512.35, 0.02]),
                covariance=np.diag([25.0, 0.04]),
                confidence=1.0,
                frame="SPRI",
                valid_flag=True,
            )
        )

    return ModuleInput(
        initial_state=InitialState(
            target_id="target_01",
            timestamp=0.0,
            state_estimate=np.array([100.0, 50.0, 500.0, 0.1, 0.0, 0.0]),
            covariance=np.diag([100.0, 100.0, 100.0, 0.1, 0.1, 0.1]),
        ),
        sensor_measurements=observations,
        config={
            "runtime": {
                "timestamps": timestamps,
                "chief_state_history_eci": chief_history,
                "q_eci2pri_history": q_history,
                "node_id": "sat_01",
            },
            "filter": {
                "process_noise": make_process_noise(1.0, 1e-4),
                "reset_feedback": True,
                "ci_objective": "trace",
                "ci_grid_points": 101,
            },
            "modalities": {
                "nn": {
                    "nn_meas_frame": "eci",
                    "nn_use_pseudo_velocity": False,
                }
            },
        },
    )


if __name__ == "__main__":
    output = StateAwarenessModule().run(build_demo_input())
    print("position:", output.state_output.position_estimate)
    print("velocity:", output.state_output.velocity_estimate)
    print("acceleration:", output.state_output.acceleration_estimate)
    print("weights:", output.fusion_status.modality_weights)
