from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from brain_inspired.orbital_phase_adapter import OrbitalPlaneFrame
from brain_inspired.orbital_radial_state import OrbitalRadialConfig, OrbitalRadialState
from orbital_core.dynamics import rk4_step_absolute

Array = np.ndarray


@dataclass(frozen=True)
class OrbitalRadialHistory:
    node_id: str
    timestamps: Array
    source_displacement: Array
    predicted_displacement: Array
    anchored_displacement: Array
    prediction_residual: Array
    anchored_residual: Array
    bump_concentration: Array
    bump_width: Array
    saturated_at_boundary: Array
    valid: Array
    cue_applied: Array
    anchor_gain: Array
    anchor_age: Array
    reference_radius: float


def run_orbital_radial_states(
    *, timestamps: Array, posterior_state_history_by_node: Mapping[str, Array],
    frame_by_node: Mapping[str, OrbitalPlaneFrame] | None = None,
    anchor_mask_by_node: Mapping[str, Array] | None = None,
    anchor_confidence_by_node: Mapping[str, Array] | None = None,
    radial_rate_bias_by_node: Mapping[str, float] | None = None,
    config: OrbitalRadialConfig | None = None,
) -> dict[str, OrbitalRadialHistory]:
    """Run causal per-node radial states over stored posterior histories."""
    times = np.asarray(timestamps, dtype=float).reshape(-1)
    if (times.size == 0 or np.any(~np.isfinite(times))
            or (times.size > 1 and np.any(np.diff(times) <= 0.0))):
        raise ValueError("timestamps must be finite, nonempty, and increasing.")
    states = {str(node): np.asarray(values, dtype=float)
              for node, values in posterior_state_history_by_node.items()}
    if not states:
        raise ValueError("At least one posterior state history is required.")
    for values in states.values():
        if values.shape != (times.size, 6) or np.any(~np.isfinite(values)):
            raise ValueError("Posterior state histories must have shape (N, 6).")
    frames = dict(frame_by_node or {})
    if set(frames) - set(states):
        raise ValueError("Frames contain unknown nodes.")

    results = {}
    for node_id, history in states.items():
        frame = frames.get(node_id, OrbitalPlaneFrame.from_state_eci(history[0]))
        mask = _node_series(anchor_mask_by_node, node_id, times.size,
                            default=False, dtype=bool)
        confidence = _node_series(anchor_confidence_by_node, node_id, times.size,
                                  default=1.0, dtype=float)
        if np.any(~np.isfinite(confidence)) or np.any(
            (confidence < 0.0) | (confidence > 1.0)
        ):
            raise ValueError("Anchor confidence values must lie in [0, 1].")
        rate_bias = float((radial_rate_bias_by_node or {}).get(node_id, 0.0))
        if not np.isfinite(rate_bias):
            raise ValueError("Per-node radial-rate biases must be finite.")
        results[node_id] = _run_node(
            node_id=node_id, times=times, posterior=history, frame=frame,
            anchor_mask=mask, confidence=confidence,
            radial_rate_bias=rate_bias, config=config,
        )
    return results


def _run_node(
    *, node_id, times, posterior, frame, anchor_mask, confidence,
    radial_rate_bias, config,
):
    radial = OrbitalRadialState(node_id=node_id, frame=frame, config=config)
    first = radial.initialize_from_state(timestamp=times[0], state_eci=posterior[0])
    source = [first.source_displacement]
    predicted = [first.decoded_displacement]
    anchored = [first.decoded_displacement]
    prediction_residual = [first.displacement_residual]
    anchored_residual = [first.displacement_residual]
    concentration = [first.bump_concentration]
    width = [first.bump_width]
    saturated = [first.saturated_at_boundary]
    valid = [first.valid]
    cues = [False]
    gains = [0.0]
    ages = [first.anchor_age]
    for index in range(1, times.size):
        delta_time = float(times[index] - times[index - 1])
        prior = rk4_step_absolute(posterior[index - 1], delta_time)
        prediction = radial.predict_from_state(
            timestamp=times[index], predicted_state_eci=prior,
            radial_rate_bias=radial_rate_bias,
        )
        endpoint = radial.anchor_from_state(
            timestamp=times[index], posterior_state_eci=posterior[index],
            confidence=confidence[index], trusted=bool(anchor_mask[index]),
        )
        source.append(endpoint.source_displacement)
        predicted.append(prediction.decoded_displacement)
        anchored.append(endpoint.decoded_displacement)
        prediction_residual.append(prediction.displacement_residual)
        anchored_residual.append(endpoint.displacement_residual)
        concentration.append(endpoint.bump_concentration)
        width.append(endpoint.bump_width)
        saturated.append(endpoint.saturated_at_boundary)
        valid.append(endpoint.valid)
        cues.append(endpoint.cue_applied)
        gains.append(endpoint.anchor_gain)
        ages.append(endpoint.anchor_age)
    return OrbitalRadialHistory(
        node_id=node_id, timestamps=times.copy(),
        source_displacement=np.asarray(source),
        predicted_displacement=np.asarray(predicted),
        anchored_displacement=np.asarray(anchored),
        prediction_residual=np.asarray(prediction_residual),
        anchored_residual=np.asarray(anchored_residual),
        bump_concentration=np.asarray(concentration), bump_width=np.asarray(width),
        saturated_at_boundary=np.asarray(saturated, dtype=bool),
        valid=np.asarray(valid, dtype=bool), cue_applied=np.asarray(cues, dtype=bool),
        anchor_gain=np.asarray(gains), anchor_age=np.asarray(ages),
        reference_radius=radial.reference_radius,
    )


def _node_series(mapping, node_id, size, *, default, dtype):
    values = (np.full(size, default, dtype=dtype)
              if mapping is None or node_id not in mapping
              else np.asarray(mapping[node_id], dtype=dtype))
    if values.shape != (size,):
        raise ValueError("Per-node radial controls must have shape (N,).")
    return values
