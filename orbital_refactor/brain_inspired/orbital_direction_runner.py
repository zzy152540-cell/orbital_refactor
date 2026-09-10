from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from brain_inspired.orbital_direction_state import (
    OrbitalDirectionConfig,
    OrbitalDirectionState,
    _circular_difference,
)
from brain_inspired.orbital_phase_adapter import (
    OrbitalPlaneFrame,
    extract_orbital_phase_state,
)
from orbital_core.dynamics import rk4_step_absolute

Array = np.ndarray


@dataclass(frozen=True)
class OrbitalDirectionHistory:
    node_id: str
    timestamps: Array
    source_phase: Array
    predicted_phase: Array
    anchored_phase: Array
    prediction_residual: Array
    anchored_residual: Array
    bump_concentration: Array
    bump_width: Array
    valid: Array
    cue_applied: Array
    anchor_gain: Array
    anchor_age: Array
    neural_activity: Array | None = None


def run_orbital_direction_states(
    *, timestamps: Array, posterior_state_history_by_node: Mapping[str, Array],
    frame_by_node: Mapping[str, OrbitalPlaneFrame] | None = None,
    anchor_mask_by_node: Mapping[str, Array] | None = None,
    anchor_confidence_by_node: Mapping[str, Array] | None = None,
    phase_rate_bias_by_node: Mapping[str, float] | None = None,
    config: OrbitalDirectionConfig | None = None,
    retain_activity: bool = False,
) -> dict[str, OrbitalDirectionHistory]:
    """Run causal per-node direction states over stored posterior histories.

    At epoch k, prediction uses only posterior k-1 propagated to k.  Posterior
    k is visible only to the optional endpoint anchoring operation.
    """
    times = np.asarray(timestamps, dtype=float).reshape(-1)
    if (
        times.size == 0 or np.any(~np.isfinite(times))
        or (times.size > 1 and np.any(np.diff(times) <= 0.0))
    ):
        raise ValueError("timestamps must be finite, nonempty, and increasing.")
    states = {
        str(node): np.asarray(values, dtype=float)
        for node, values in posterior_state_history_by_node.items()
    }
    if not states:
        raise ValueError("At least one posterior state history is required.")
    for values in states.values():
        if values.shape != (times.size, 6) or np.any(~np.isfinite(values)):
            raise ValueError(
                "Posterior state histories must have shape (N, 6)."
            )
    frames = dict(frame_by_node or {})
    unknown_frames = set(frames) - set(states)
    if unknown_frames:
        raise ValueError(f"Frames contain unknown nodes: {sorted(unknown_frames)}")

    results = {}
    for node_id, history in states.items():
        frame = (
            frames[node_id]
            if node_id in frames
            else OrbitalPlaneFrame.from_state_eci(history[0])
        )
        anchor_mask = _node_series(
            anchor_mask_by_node, node_id, times.size, default=False,
            dtype=bool,
        )
        confidence = _node_series(
            anchor_confidence_by_node, node_id, times.size, default=1.0,
            dtype=float,
        )
        if np.any(~np.isfinite(confidence)) or np.any(
            (confidence < 0.0) | (confidence > 1.0)
        ):
            raise ValueError("Anchor confidence values must lie in [0, 1].")
        rate_bias = float((phase_rate_bias_by_node or {}).get(node_id, 0.0))
        if not np.isfinite(rate_bias):
            raise ValueError("Per-node phase-rate biases must be finite.")
        results[node_id] = _run_node(
            node_id=node_id, times=times, posterior=history, frame=frame,
            anchor_mask=anchor_mask, confidence=confidence, config=config,
            phase_rate_bias=rate_bias,
            retain_activity=retain_activity,
        )
    return results


def _run_node(
    *, node_id, times, posterior, frame, anchor_mask, confidence, config,
    phase_rate_bias, retain_activity,
):
    direction = OrbitalDirectionState(
        node_id=node_id, frame=frame, config=config,
    )
    first = direction.initialize_from_state(
        timestamp=times[0], state_eci=posterior[0],
    )
    source_phase = [_phase(times[0], posterior[0], frame)]
    predicted_phase = [first.decoded_phase]
    anchored_phase = [first.decoded_phase]
    prediction_residual = [
        _circular_difference(first.decoded_phase, source_phase[0])
    ]
    anchored_residual = list(prediction_residual)
    concentration = [first.bump_concentration]
    width = [first.bump_width]
    valid = [first.valid]
    cue_applied = [False]
    anchor_gain = [0.0]
    anchor_age = [first.anchor_age]
    neural_activity = [first.neural_activity.copy()] if retain_activity else None

    for index in range(1, times.size):
        delta_time = float(times[index] - times[index - 1])
        prior_state = rk4_step_absolute(posterior[index - 1], delta_time)
        predicted = direction.predict_from_state(
            timestamp=times[index], predicted_state_eci=prior_state,
            phase_rate_bias=phase_rate_bias,
        )
        source = _phase(times[index], posterior[index], frame)
        anchored = direction.anchor_from_state(
            timestamp=times[index], posterior_state_eci=posterior[index],
            confidence=confidence[index], trusted=bool(anchor_mask[index]),
        )
        source_phase.append(source)
        predicted_phase.append(predicted.decoded_phase)
        anchored_phase.append(anchored.decoded_phase)
        prediction_residual.append(
            _circular_difference(predicted.decoded_phase, source)
        )
        anchored_residual.append(
            _circular_difference(anchored.decoded_phase, source)
        )
        concentration.append(anchored.bump_concentration)
        width.append(anchored.bump_width)
        valid.append(anchored.valid)
        cue_applied.append(anchored.cue_applied)
        anchor_gain.append(anchored.anchor_gain)
        anchor_age.append(anchored.anchor_age)
        if neural_activity is not None:
            neural_activity.append(anchored.neural_activity.copy())
    return OrbitalDirectionHistory(
        node_id=node_id, timestamps=times.copy(),
        source_phase=np.asarray(source_phase),
        predicted_phase=np.asarray(predicted_phase),
        anchored_phase=np.asarray(anchored_phase),
        prediction_residual=np.asarray(prediction_residual),
        anchored_residual=np.asarray(anchored_residual),
        bump_concentration=np.asarray(concentration),
        bump_width=np.asarray(width), valid=np.asarray(valid, dtype=bool),
        cue_applied=np.asarray(cue_applied, dtype=bool),
        anchor_gain=np.asarray(anchor_gain), anchor_age=np.asarray(anchor_age),
        neural_activity=(
            None if neural_activity is None else np.asarray(neural_activity)
        ),
    )


def _phase(timestamp, state, frame):
    return extract_orbital_phase_state(
        timestamp=timestamp, state_eci=state, frame=frame,
    ).argument_of_latitude


def _node_series(mapping, node_id, size, *, default, dtype):
    values = (
        np.full(size, default, dtype=dtype)
        if mapping is None or node_id not in mapping
        else np.asarray(mapping[node_id], dtype=dtype)
    )
    if values.shape != (size,):
        raise ValueError("Per-node direction controls must have shape (N,).")
    return values
