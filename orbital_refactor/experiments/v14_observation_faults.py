from __future__ import annotations

from dataclasses import replace

import numpy as np

from orbital_core.inter_satellite_model import RelativeMeasurementModel


def apply_observation_faults(
    observations, *, truth, seed, radar_actual_noise_scale=1.0,
    optical_outlier_bias=None, optical_outlier_window=None,
    dropout_windows_by_modality=None,
    infrared_outlier_bias=None, infrared_outlier_window=None,
):
    rng = np.random.default_rng(20270104 + seed)
    ordered_timestamps = sorted({float(item.timestamp) for item in observations})
    timestamp_to_index = {
        timestamp: index for index, timestamp in enumerate(ordered_timestamps)
    }
    result = []
    for observation in observations:
        timestamp = float(observation.timestamp)
        if any(
            start <= timestamp <= end
            for start, end in (dropout_windows_by_modality or {}).get(
                observation.modality, ()
            )
        ):
            continue
        modified = observation
        if observation.modality == "RADAR" and radar_actual_noise_scale != 1.0:
            index = timestamp_to_index[timestamp]
            model = RelativeMeasurementModel("RADAR", observation.frame)
            ideal = model.predict(
                truth[observation.observer_id][index],
                truth[observation.target_id][index],
            )
            actual_noise = rng.multivariate_normal(
                np.zeros(2),
                observation.covariance * radar_actual_noise_scale**2,
            )
            modified = replace(observation, measurement=ideal + actual_noise)
        if (
            observation.modality == "OPTICAL"
            and optical_outlier_bias is not None
            and optical_outlier_window[0] <= timestamp <= optical_outlier_window[1]
        ):
            modified = replace(
                modified,
                measurement=(
                    np.asarray(modified.measurement, dtype=float)
                    + np.asarray(optical_outlier_bias, dtype=float)
                ),
            )
        if (
            observation.modality == "INFRARED"
            and infrared_outlier_bias is not None
            and infrared_outlier_window[0] <= timestamp <= infrared_outlier_window[1]
        ):
            modified = replace(
                modified,
                measurement=(
                    np.asarray(modified.measurement, dtype=float)
                    + np.asarray(infrared_outlier_bias, dtype=float)
                ),
            )
        result.append(modified)
    return result
