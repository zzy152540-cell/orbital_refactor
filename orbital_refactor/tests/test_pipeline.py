import numpy as np

from orbital_core.dynamics import make_process_noise
from orbital_core.filters import LocalDynamicsEKF
from orbital_core.measurements import h_radar_spri
from pipelines.single_modal import run_single_modal_filter


def test_single_modal_pipeline_keeps_prediction_output_when_measurement_missing():
    n = 4
    t = np.arange(n, dtype=float)
    chief = np.tile(np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0]), (n, 1))
    q_hist = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    truth = np.tile(np.array([100.0, 20.0, -10.0, 0.0, 0.01, 0.0]), (n, 1))
    z = np.vstack([h_radar_spri(truth[k], q_hist[k]) for k in range(n)])
    valid = np.array([True, True, False, True])
    ekf = LocalDynamicsEKF(
        make_process_noise(1.0, 1e-6),
        np.diag([1.0, 0.01]),
        "rad",
    )
    history = run_single_modal_filter(
        timestamps=t,
        chief_state_history_eci=chief,
        q_eci2pri_history=q_hist,
        measurements=z,
        measurement_valid_flags=valid,
        ekf=ekf,
        initial_state=truth[0],
        initial_covariance=np.eye(6),
    )
    assert history.state_history.shape == (n, 6)
    assert history.covariance_history.shape == (n, 6, 6)
    assert history.statistics["skipped"] == 1
    assert history.valid_history[2]
