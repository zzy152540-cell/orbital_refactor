from __future__ import annotations

import numpy as np

from orbital_core.attitude import (
    quat_multiply_wxyz,
    quat_normalize_wxyz,
    small_angle_quaternion_wxyz,
)
from orbital_core.coordinates import dcm_to_quat_wxyz


def two_node_target_pointing_attitude_history(truth_by_node):
    """Build BODY attitudes whose x axes point at the other spacecraft."""

    node_ids = tuple(truth_by_node)
    if len(node_ids) != 2:
        raise ValueError("Target-pointing attitude helper requires exactly two nodes.")
    result = {}
    for observer, target in (node_ids, tuple(reversed(node_ids))):
        quaternions = []
        for observer_state, target_state in zip(
            truth_by_node[observer], truth_by_node[target]
        ):
            x_body_eci = target_state[:3] - observer_state[:3]
            x_body_eci /= np.linalg.norm(x_body_eci)
            reference = observer_state[:3] / np.linalg.norm(observer_state[:3])
            z_body_eci = reference - np.dot(reference, x_body_eci) * x_body_eci
            if np.linalg.norm(z_body_eci) < 1e-10:
                reference = np.array([0.0, 0.0, 1.0])
                z_body_eci = reference - np.dot(reference, x_body_eci) * x_body_eci
            z_body_eci /= np.linalg.norm(z_body_eci)
            y_body_eci = np.cross(z_body_eci, x_body_eci)
            quaternions.append(dcm_to_quat_wxyz(
                np.vstack((x_body_eci, y_body_eci, z_body_eci))
            ))
        result[observer] = np.vstack(quaternions)
    return result


def perturb_attitude_history(attitude_by_node, *, sigma, seed):
    """Apply repeatable independent small-angle errors to attitude histories."""

    rng = np.random.default_rng(20261130 + seed)
    return {
        node_id: np.vstack([
            quat_normalize_wxyz(quat_multiply_wxyz(
                small_angle_quaternion_wxyz(rng.normal(0.0, sigma, 3)),
                quaternion,
            ))
            for quaternion in history
        ])
        for node_id, history in attitude_by_node.items()
    }


def slew_target_pointing_attitude_history(
    attitude_by_node, *, maximum_offset, transition_type, jitter_amplitude=0.0,
):
    """Add a deterministic boresight slew and optional oscillatory jitter."""

    result = {}
    for node_id, history in attitude_by_node.items():
        fractions = np.linspace(0.0, 1.0, len(history))
        if transition_type == "recovery":
            fractions = fractions[::-1]
        offsets = (
            maximum_offset * fractions
            + jitter_amplitude * np.sin(2.0 * np.pi * 12.0 * fractions)
        )
        result[node_id] = np.vstack([
            quat_normalize_wxyz(quat_multiply_wxyz(
                small_angle_quaternion_wxyz(np.array([0.0, 0.0, offset])),
                quaternion,
            ))
            for quaternion, offset in zip(history, offsets)
        ])
    return result
