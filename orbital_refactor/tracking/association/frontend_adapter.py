from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from orbital_core.attitude import quat_to_dcm_i2b
from tracking.initial_orbit.frontend_adapter import _body_line_of_sight

from .data_contracts import UnlabeledIODObservation, UnlabeledObservationMessage


def unlabeled_iod_observations_from_messages(
    messages,
    *,
    observer_state_by_epoch: Mapping[tuple[str, float], np.ndarray],
) -> tuple[UnlabeledIODObservation, ...]:
    """Convert target-free raw sensor messages into inertial association inputs."""

    result = []
    for message in messages:
        if not isinstance(message, UnlabeledObservationMessage):
            raise TypeError("Raw association input must contain unlabeled messages.")
        key = (message.observer_id, message.timestamp)
        if key not in observer_state_by_epoch:
            raise ValueError(f"Missing observer J2000 state for {key!r}.")
        observer_state = observer_state_by_epoch[key]
        metadata = {
            **message.metadata,
            "sourceModality": message.modality,
            "sourceMessageId": message.message_id,
            "confidence": message.confidence,
        }
        if message.modality == "RADAR":
            result.append(UnlabeledIODObservation(
                timestamp=message.timestamp,
                observer_id=message.observer_id,
                modality="RADAR",
                observer_state_eci=observer_state,
                measurement=message.measurement,
                covariance=message.covariance,
                valid_flag=message.valid_flag,
                detection_group_id=message.detection_group_id,
                metadata=metadata,
            ))
            continue
        if message.modality == "LOS":
            result.append(UnlabeledIODObservation(
                timestamp=message.timestamp,
                observer_id=message.observer_id,
                modality="LOS",
                observer_state_eci=observer_state,
                measurement=message.measurement,
                covariance=message.covariance,
                valid_flag=message.valid_flag,
                detection_group_id=message.detection_group_id,
                metadata={**metadata, "lineOfSightFrame": "J2000_ECI"},
            ))
            continue
        quaternion = message.metadata.get("quaternion_i2b_wxyz")
        if quaternion is None:
            raise ValueError(f"{message.modality} message requires attitude metadata.")
        line_body = _body_line_of_sight(message.modality, message.measurement)
        line_eci = quat_to_dcm_i2b(quaternion).T @ line_body
        line_eci /= np.linalg.norm(line_eci)
        result.append(UnlabeledIODObservation(
            timestamp=message.timestamp,
            observer_id=message.observer_id,
            modality="LOS",
            observer_state_eci=observer_state,
            measurement=line_eci,
            covariance=message.covariance,
            valid_flag=message.valid_flag,
            detection_group_id=message.detection_group_id,
            metadata={**metadata, "lineOfSightFrame": "J2000_ECI"},
        ))
    return tuple(result)

