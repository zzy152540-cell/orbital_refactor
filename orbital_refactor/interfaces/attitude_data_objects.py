from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class GyroObservation:
    timestamp: float
    satellite_id: str
    angular_rate_body: Array
    covariance: Array
    valid_flag: bool = True


@dataclass(frozen=True)
class StarTrackerObservation:
    timestamp: float
    satellite_id: str
    quaternion_i2b_wxyz: Array
    covariance_small_angle: Array
    valid_flag: bool = True


@dataclass(frozen=True)
class AttitudeEstimate:
    timestamp: float
    satellite_id: str
    quaternion_i2b_wxyz: Array
    angular_velocity_body: Array
    gyro_bias: Array
    error_covariance: Array

    @property
    def attitude_covariance(self) -> Array:
        return np.asarray(self.error_covariance, dtype=float)[:3, :3]
