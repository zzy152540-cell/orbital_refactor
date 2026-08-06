from __future__ import annotations

import numpy as np

from experiments.scenario_controls import (
    measurement_is_due,
    topology_edge_is_inactive,
)
from interfaces.data_objects import ObservationMessage
from orbital_core.coordinates import dcm_to_quat_wxyz
from orbital_core.inter_satellite_model import body_angle_effective_covariance
from orbital_core.measurement_semantics import inter_satellite_semantic_metadata
from orbital_core.measurements import (
    measure_relative_az_el,
    measure_relative_optical_uv,
    measure_relative_range,
    measure_relative_range_rate,
)


def generate_inter_satellite_observations_for_epoch(
    *, index, timestamp, truth, topology, topology_inactive_windows,
    relative_modalities, measurement_periods, visible_opportunities,
    range_rng, range_rate_rng, az_el_rng, optical_rng,
    range_sigma, range_rate_sigma, radar_correlation, az_el_sigma,
    optical_sigma, az_el_frame, attitude_history_by_node,
    estimated_attitude_history_by_node, attitude_covariance,
) -> list[ObservationMessage]:
    """Generate directed relative observations for one estimator epoch.

    The random-number calls intentionally follow the legacy loop order. In
    particular, one scalar range sample is drawn for every active directed
    edge before modality checks, even when the physical RADAR mode is used.
    This preserves fixed-seed experiment results during the refactor.
    """

    observations = []
    timestamp = float(timestamp)
    for observer in topology.node_ids:
        for target in topology.neighbors(observer):
            if topology_edge_is_inactive(
                topology_inactive_windows,
                first=observer,
                second=target,
                timestamp=timestamp,
            ):
                continue
            range_noise = range_rng.normal(0.0, range_sigma)
            if "RADAR" in relative_modalities and measurement_is_due(
                "RADAR", timestamp, measurement_periods
            ) and _is_visible(
                visible_opportunities, timestamp, observer, target, "RADAR"
            ):
                radar_covariance = np.array([
                    [
                        range_sigma**2,
                        radar_correlation * range_sigma * range_rate_sigma,
                    ],
                    [
                        radar_correlation * range_sigma * range_rate_sigma,
                        range_rate_sigma**2,
                    ],
                ])
                radar_noise = range_rng.multivariate_normal(
                    np.zeros(2), radar_covariance
                )
                information_id = f"{observer}->{target}:radar:{index}"
                observations.append(ObservationMessage(
                    message_id=information_id,
                    physical_observation_id=information_id,
                    observer_id=observer,
                    target_id=target,
                    timestamp=timestamp,
                    modality="RADAR",
                    measurement=np.array([
                        measure_relative_range(
                            truth[observer][index], truth[target][index]
                        ),
                        measure_relative_range_rate(
                            truth[observer][index], truth[target][index]
                        ),
                    ]) + radar_noise,
                    covariance=radar_covariance,
                    metadata=inter_satellite_semantic_metadata("RADAR"),
                ))
            if "RANGE" in relative_modalities and measurement_is_due(
                "RANGE", timestamp, measurement_periods
            ) and _is_visible(
                visible_opportunities, timestamp, observer, target, "RANGE"
            ):
                information_id = f"{observer}->{target}:range:{index}"
                observations.append(ObservationMessage(
                    message_id=information_id,
                    physical_observation_id=information_id,
                    observer_id=observer,
                    target_id=target,
                    timestamp=timestamp,
                    modality="RANGE",
                    measurement=np.array([
                        measure_relative_range(
                            truth[observer][index], truth[target][index]
                        ) + range_noise
                    ]),
                    covariance=np.array([[range_sigma**2]]),
                    metadata=inter_satellite_semantic_metadata("RANGE"),
                ))
            if "RANGE_RATE" in relative_modalities and measurement_is_due(
                "RANGE_RATE", timestamp, measurement_periods
            ):
                rate_noise = range_rate_rng.normal(0.0, range_rate_sigma)
                if _is_visible(
                    visible_opportunities,
                    timestamp,
                    observer,
                    target,
                    "RANGE_RATE",
                ):
                    information_id = (
                        f"{observer}->{target}:range_rate:{index}"
                    )
                    observations.append(ObservationMessage(
                        message_id=information_id,
                        physical_observation_id=information_id,
                        observer_id=observer,
                        target_id=target,
                        timestamp=timestamp,
                        modality="RANGE_RATE",
                        measurement=np.array([
                            measure_relative_range_rate(
                                truth[observer][index], truth[target][index]
                            ) + rate_noise
                        ]),
                        covariance=np.array([[range_rate_sigma**2]]),
                        metadata=inter_satellite_semantic_metadata("RANGE_RATE"),
                    ))
            angular_modality = (
                "INFRARED" if "INFRARED" in relative_modalities
                else "AZ_EL" if "AZ_EL" in relative_modalities
                else None
            )
            if angular_modality is not None and measurement_is_due(
                angular_modality, timestamp, measurement_periods
            ):
                angle_noise = az_el_rng.normal(0.0, az_el_sigma, 2)
                if _is_visible(
                    visible_opportunities,
                    timestamp,
                    observer,
                    target,
                    angular_modality,
                ):
                    observations.append(_angular_observation(
                        index=index,
                        timestamp=timestamp,
                        observer=observer,
                        target=target,
                        modality=angular_modality,
                        truth=truth,
                        angle_noise=angle_noise,
                        az_el_sigma=az_el_sigma,
                        az_el_frame=az_el_frame,
                        attitude_history_by_node=attitude_history_by_node,
                        estimated_attitude_history_by_node=(
                            estimated_attitude_history_by_node
                        ),
                        attitude_covariance=attitude_covariance,
                    ))
            if "OPTICAL" in relative_modalities and measurement_is_due(
                "OPTICAL", timestamp, measurement_periods
            ):
                optical_noise = optical_rng.normal(0.0, optical_sigma, 2)
                if _is_visible(
                    visible_opportunities,
                    timestamp,
                    observer,
                    target,
                    "OPTICAL",
                ):
                    observations.append(_optical_observation(
                        index=index,
                        timestamp=timestamp,
                        observer=observer,
                        target=target,
                        truth=truth,
                        optical_noise=optical_noise,
                        optical_sigma=optical_sigma,
                        attitude_history_by_node=attitude_history_by_node,
                        estimated_attitude_history_by_node=(
                            estimated_attitude_history_by_node
                        ),
                    ))
    return observations


def _is_visible(opportunities, timestamp, observer, target, modality):
    return opportunities is None or (
        timestamp, observer, target, modality
    ) in opportunities


def _angular_observation(
    *, index, timestamp, observer, target, modality, truth, angle_noise,
    az_el_sigma, az_el_frame, attitude_history_by_node,
    estimated_attitude_history_by_node, attitude_covariance,
):
    information_id = f"{observer}->{target}:{modality.lower()}:{index}"
    truth_quaternion = (
        attitude_history_by_node[observer][index]
        if az_el_frame == "BODY" else None
    )
    estimate_quaternion = (
        estimated_attitude_history_by_node[observer][index]
        if az_el_frame == "BODY" else None
    )
    sensor_covariance = np.eye(2) * az_el_sigma**2
    measurement_covariance = (
        body_angle_effective_covariance(
            truth[observer][index],
            truth[target][index],
            quaternion_i2b_wxyz=estimate_quaternion,
            sensor_covariance=sensor_covariance,
            attitude_covariance=attitude_covariance,
        )
        if attitude_covariance is not None
        else sensor_covariance
    )
    return ObservationMessage(
        message_id=information_id,
        physical_observation_id=information_id,
        observer_id=observer,
        target_id=target,
        timestamp=timestamp,
        modality=modality,
        frame=az_el_frame,
        measurement=measure_relative_az_el(
            truth[observer][index],
            truth[target][index],
            frame=az_el_frame,
            quaternion_i2b_wxyz=truth_quaternion,
        ) + angle_noise,
        covariance=measurement_covariance,
        metadata={
            **inter_satellite_semantic_metadata(modality),
            **(
                {"quaternion_i2b_wxyz": estimate_quaternion.copy()}
                if estimate_quaternion is not None else {}
            ),
        },
    )


def _optical_observation(
    *, index, timestamp, observer, target, truth, optical_noise,
    optical_sigma, attitude_history_by_node, estimated_attitude_history_by_node,
):
    information_id = f"{observer}->{target}:optical:{index}"
    truth_quaternion = (
        attitude_history_by_node[observer][index]
        if attitude_history_by_node is not None
        else target_pointing_quaternion(
            truth[observer][index], truth[target][index]
        )
    )
    estimate_quaternion = (
        estimated_attitude_history_by_node[observer][index]
        if estimated_attitude_history_by_node is not None
        else truth_quaternion
    )
    return ObservationMessage(
        message_id=information_id,
        physical_observation_id=information_id,
        observer_id=observer,
        target_id=target,
        timestamp=timestamp,
        modality="OPTICAL",
        frame="BODY",
        measurement=measure_relative_optical_uv(
            truth[observer][index],
            truth[target][index],
            frame="BODY",
            quaternion_i2b_wxyz=truth_quaternion,
        ) + optical_noise,
        covariance=np.eye(2) * optical_sigma**2,
        metadata={
            **inter_satellite_semantic_metadata("OPTICAL"),
            "quaternion_i2b_wxyz": estimate_quaternion.copy(),
        },
    )


def target_pointing_quaternion(observer_state, target_state):
    """Construct an observer BODY frame whose x-axis points at the target."""

    observer = np.asarray(observer_state, dtype=float).reshape(6)
    target = np.asarray(target_state, dtype=float).reshape(6)
    x_body_eci = target[:3] - observer[:3]
    x_body_eci /= np.linalg.norm(x_body_eci)
    reference = observer[:3] / np.linalg.norm(observer[:3])
    z_body_eci = reference - np.dot(reference, x_body_eci) * x_body_eci
    if np.linalg.norm(z_body_eci) < 1e-10:
        reference = np.array([0.0, 0.0, 1.0])
        z_body_eci = reference - np.dot(reference, x_body_eci) * x_body_eci
    z_body_eci /= np.linalg.norm(z_body_eci)
    y_body_eci = np.cross(z_body_eci, x_body_eci)
    return dcm_to_quat_wxyz(np.vstack((x_body_eci, y_body_eci, z_body_eci)))
