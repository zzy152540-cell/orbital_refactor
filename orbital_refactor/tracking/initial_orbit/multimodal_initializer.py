from __future__ import annotations

from collections import defaultdict

import numpy as np

from .data_contracts import IODObservation


def multimodal_state_guess(observations: tuple[IODObservation, ...]):
    """Create a J2000 state guess from paired LOS and range/range-rate data."""

    valid = tuple(item for item in observations if item.valid_flag)
    grouped = defaultdict(list)
    for item in valid:
        grouped[(item.observer_id, item.timestamp)].append(item)
    position_samples = []
    range_rate_rows = []
    for (_observer_id, timestamp), values in grouped.items():
        line_items = [item for item in values if item.modality == "LOS"]
        radar_items = [item for item in values if item.modality == "RADAR"]
        if not line_items or not radar_items:
            continue
        line = _combine_lines(line_items)
        radar = radar_items[0]
        rho, rho_dot = radar.measurement
        observer = radar.observer_state_eci
        position = observer[:3] + rho * line
        angular_variance = float(np.mean([
            np.trace(item.covariance) / 2.0 for item in line_items
        ]))
        range_variance = float(radar.covariance[0, 0])
        covariance = (
            range_variance * np.outer(line, line)
            + rho * rho * angular_variance * (np.eye(3) - np.outer(line, line))
        )
        covariance += np.eye(3) * 1e-6
        position_samples.append((float(timestamp), position, covariance))
        range_rate_rows.append((
            line,
            float(rho_dot + line @ observer[3:]),
            float(radar.covariance[1, 1]),
        ))
    if len(position_samples) < 2:
        raise ValueError("IOD requires paired LOS and radar data at two or more epochs.")
    epoch = min(item[0] for item in position_samples)
    design = []
    values = []
    for timestamp, position, covariance in position_samples:
        whitening = _inverse_sqrt(covariance)
        dt = timestamp - epoch
        design.append(whitening @ np.hstack((np.eye(3), dt * np.eye(3))))
        values.append(whitening @ position)
    state, *_ = np.linalg.lstsq(np.vstack(design), np.concatenate(values), rcond=None)
    if range_rate_rows:
        velocity_design = []
        velocity_values = []
        for line, value, variance in range_rate_rows:
            scale = np.sqrt(variance)
            velocity_design.append(line / scale)
            velocity_values.append(value / scale)
        regularization = 1.0 / 10.0
        velocity_design.extend(regularization * np.eye(3))
        velocity_values.extend(regularization * state[3:])
        state[3:], *_ = np.linalg.lstsq(
            np.asarray(velocity_design), np.asarray(velocity_values), rcond=None,
        )
    return epoch, state, {
        "pairedPositionCount": len(position_samples),
        "rangeRateConstraintCount": len(range_rate_rows),
        "observerCount": len({item.observer_id for item in valid}),
    }


def _combine_lines(items):
    weighted = np.zeros(3)
    for item in items:
        variance = float(np.trace(item.covariance) / 2.0)
        weighted += item.measurement / variance
    norm = np.linalg.norm(weighted)
    if norm <= 0.0:
        raise ValueError("LOS observations cancel and cannot define a direction.")
    return weighted / norm


def _inverse_sqrt(covariance):
    values, vectors = np.linalg.eigh(covariance)
    return vectors @ np.diag(1.0 / np.sqrt(np.maximum(values, 1e-15))) @ vectors.T
