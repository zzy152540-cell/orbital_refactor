from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from interfaces.data_objects import ObservationMessage
from interfaces.data_objects import Observation
from interfaces.interface_contracts import validate_observation
from orbital_core.attitude import quat_to_dcm_i2b
from orbital_core.coordinates import rotate_eci_to_pri

from .data_contracts import IODObservation


def iod_observations_from_messages(
    messages,
    *,
    observer_state_by_epoch: Mapping[tuple[str, float], np.ndarray],
) -> tuple[IODObservation, ...]:
    """Convert canonical raw-frontend messages into inertial IOD observations."""

    result = []
    for message in messages:
        if not isinstance(message, ObservationMessage):
            raise TypeError("IOD frontend input must contain ObservationMessage values.")
        key = (str(message.observer_id), float(message.timestamp))
        if key not in observer_state_by_epoch:
            raise ValueError(f"Missing observer J2000 state for {key!r}.")
        observer_state = observer_state_by_epoch[key]
        modality = str(message.modality).upper()
        if modality == "RADAR":
            result.append(IODObservation(
                timestamp=message.timestamp,
                observer_id=message.observer_id,
                target_id=message.target_id,
                modality="RADAR",
                observer_state_eci=observer_state,
                measurement=message.measurement,
                covariance=message.covariance,
                valid_flag=message.valid_flag,
                metadata={
                    **message.metadata,
                    "sourceModality": "RADAR",
                    "sourceMessageId": message.information_id,
                },
            ))
        elif modality in {"OPTICAL", "INFRARED"}:
            quaternion = message.metadata.get("quaternion_i2b_wxyz")
            if quaternion is None:
                raise ValueError(f"{modality} message requires attitude metadata.")
            line_body = _body_line_of_sight(modality, message.measurement)
            dcm_i2b = quat_to_dcm_i2b(quaternion)
            line_eci = dcm_i2b.T @ line_body
            line_eci /= np.linalg.norm(line_eci)
            result.append(IODObservation(
                timestamp=message.timestamp,
                observer_id=message.observer_id,
                target_id=message.target_id,
                modality="LOS",
                observer_state_eci=observer_state,
                measurement=line_eci,
                covariance=message.covariance,
                valid_flag=message.valid_flag,
                metadata={
                    **message.metadata,
                    "sourceModality": modality,
                    "sourceMessageId": message.information_id,
                    "lineOfSightFrame": "J2000_ECI",
                },
            ))
        else:
            raise ValueError(f"Unsupported raw IOD modality {modality!r}.")
    return tuple(result)


def _body_line_of_sight(modality, measurement):
    first, second = np.asarray(measurement, dtype=float).reshape(2)
    if modality == "OPTICAL":
        line = np.array([1.0, first, second])
    else:
        tangent_azimuth = np.tan(first)
        line = np.array([
            1.0,
            tangent_azimuth,
            np.tan(second) * np.sqrt(1.0 + tangent_azimuth**2),
        ])
    return line / np.linalg.norm(line)


def tracking_observation_from_message(message, *, q_eci2pri):
    """Convert a canonical raw message into the legacy SPRI filter contract."""

    modality = str(message.modality).upper()
    metadata = {
        **message.metadata,
        "observation_id": message.information_id,
        "source_timestamp": message.source_timestamp,
        "arrival_timestamp": message.arrival_timestamp,
        "source_measurement_frame": str(message.frame),
    }
    if modality == "RADAR":
        observation = Observation(
            timestamp=message.timestamp, observer_id=message.observer_id,
            target_id=message.target_id, modality=modality,
            source_type="RAW_SENSOR", measurement=np.asarray(message.measurement).copy(),
            covariance=np.asarray(message.covariance).copy(), confidence=message.confidence,
            frame="SPRI", valid_flag=message.valid_flag, metadata=metadata,
        )
        validate_observation(observation)
        return observation
    if modality not in {"OPTICAL", "INFRARED"}:
        raise ValueError(f"Unsupported tracking modality {modality!r}.")
    camera_quaternion = message.metadata.get("quaternion_i2b_wxyz")
    if camera_quaternion is None:
        raise ValueError(f"{modality} message requires camera attitude metadata.")

    def transform(values):
        line_body = _body_line_of_sight(modality, values)
        line_eci = quat_to_dcm_i2b(camera_quaternion).T @ line_body
        line_spri = rotate_eci_to_pri(line_eci, q_eci2pri)
        if modality == "INFRARED":
            return np.array([
                np.arctan2(line_spri[1], line_spri[0]),
                np.arctan2(line_spri[2], np.linalg.norm(line_spri[:2])),
            ])
        return line_spri[:2] / line_spri[2]

    measurement = transform(message.measurement)
    epsilon = 1e-6
    jacobian = np.empty((2, 2))
    for axis in range(2):
        delta = np.zeros(2)
        delta[axis] = epsilon
        jacobian[:, axis] = (
            transform(np.asarray(message.measurement) + delta)
            - transform(np.asarray(message.measurement) - delta)
        ) / (2.0 * epsilon)
    covariance = jacobian @ np.asarray(message.covariance) @ jacobian.T
    valid = bool(message.valid_flag)
    if modality == "OPTICAL":
        line_body = _body_line_of_sight(modality, message.measurement)
        line_eci = quat_to_dcm_i2b(camera_quaternion).T @ line_body
        valid = valid and bool(rotate_eci_to_pri(line_eci, q_eci2pri)[2] > 1e-12)
    observation = Observation(
        timestamp=message.timestamp, observer_id=message.observer_id,
        target_id=message.target_id, modality=modality,
        source_type="RAW_SENSOR", measurement=measurement,
        covariance=covariance, confidence=message.confidence,
        frame="SPRI", valid_flag=valid,
        metadata={**metadata, "tracking_frame_adapter": "BODY_ECI_TO_SPRI"},
    )
    validate_observation(observation)
    return observation
