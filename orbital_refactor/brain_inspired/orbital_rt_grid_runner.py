from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from brain_inspired.orbital_rt_grid_state import OrbitalRTGridConfig, OrbitalRTGridState
from orbital_core.dynamics import rk4_step_absolute

Array = np.ndarray


@dataclass(frozen=True)
class OrbitalRTGridHistory:
    node_id: str
    timestamps: Array
    source_rt: Array
    predicted_rt: Array
    anchored_rt: Array
    prediction_residual_rt: Array
    anchored_residual_rt: Array
    valid: Array
    saturated_at_boundary: Array
    cue_applied: Array
    anchor_gain: Array
    anchor_age: Array
    rate_correction_rt: Array
    bias_update_applied: Array
    anchor_innovation_norm: Array
    anchor_rejected: Array
    anchor_rejection_reason: Array
    reference_origin_rt: Array
    reference_rebased: Array
    reference_rebase_count: Array


def run_orbital_rt_grid_states(
    *, timestamps: Array,
    posterior_state_history_by_node: Mapping[str, Array],
    reference_state_history_by_node: Mapping[str, Array],
    anchor_state_history_by_node: Mapping[str, Array] | None = None,
    anchor_mask_by_node: Mapping[str, Array] | None = None,
    anchor_confidence_by_node: Mapping[str, Array] | None = None,
    rate_bias_rt_by_node: Mapping[str, Array] | None = None,
    config: OrbitalRTGridConfig | None = None,
) -> dict[str, OrbitalRTGridHistory]:
    """Run causal independent-axis RT grids around explicit references."""
    times = np.asarray(timestamps, dtype=float).reshape(-1)
    if (times.size == 0 or np.any(~np.isfinite(times))
            or (times.size > 1 and np.any(np.diff(times) <= 0.0))):
        raise ValueError("timestamps must be finite, nonempty, and increasing.")
    posterior = _histories(posterior_state_history_by_node, times.size)
    reference = _histories(reference_state_history_by_node, times.size)
    if set(posterior) != set(reference):
        raise ValueError("Posterior and reference histories must cover identical nodes.")
    anchors = (posterior if anchor_state_history_by_node is None else
               _histories(anchor_state_history_by_node, times.size))
    if set(anchors) != set(posterior):
        raise ValueError("Anchor and posterior histories must cover identical nodes.")
    result = {}
    for node, states in posterior.items():
        mask = _series(anchor_mask_by_node, node, times.size, False, bool)
        confidence = _series(anchor_confidence_by_node, node, times.size, 1.0, float)
        if np.any(~np.isfinite(confidence)) or np.any(
            (confidence < 0.0) | (confidence > 1.0)
        ):
            raise ValueError("Anchor confidence values must lie in [0, 1].")
        bias = _rate_bias_series(rate_bias_rt_by_node, node, times.size)
        result[node] = _run_node(
            node=node, times=times, posterior=states,
            anchors=anchors[node], reference=reference[node],
            mask=mask, confidence=confidence,
            bias=bias, config=config,
        )
    return result


def _run_node(*, node, times, posterior, anchors, reference, mask, confidence,
              bias, config):
    grid = OrbitalRTGridState(node_id=node, config=config)
    first = grid.initialize(
        timestamp=times[0], state_eci=posterior[0],
        reference_state_eci=reference[0],
    )
    source = [first.source_rt]
    predicted = [first.decoded_rt]
    anchored = [first.decoded_rt]
    prediction_residual = [first.residual_rt]
    anchored_residual = [first.residual_rt]
    valid = [first.valid]
    saturated = [first.saturated_at_boundary]
    cues = [False]
    gains = [0.0]
    ages = [first.anchor_age]
    corrections = [first.rate_correction_rt]
    bias_updates = [first.bias_update_applied]
    innovations = [first.anchor_innovation_norm]
    rejected = [first.anchor_rejected]
    rejection_reasons = [first.anchor_rejection_reason]
    reference_origins = [first.reference_origin_rt]
    reference_rebased = [first.reference_rebased]
    reference_rebase_counts = [first.reference_rebase_count]
    for index in range(1, times.size):
        dt = float(times[index] - times[index - 1])
        prior = rk4_step_absolute(posterior[index - 1], dt)
        prediction = grid.predict(
            timestamp=times[index], predicted_state_eci=prior,
            reference_state_eci=reference[index], rate_bias_rt=bias[index],
        )
        endpoint = grid.anchor(
            timestamp=times[index], posterior_state_eci=anchors[index],
            reference_state_eci=reference[index],
            confidence=confidence[index], trusted=bool(mask[index]),
        )
        source.append(endpoint.source_rt)
        predicted.append(prediction.decoded_rt)
        anchored.append(endpoint.decoded_rt)
        prediction_residual.append(prediction.residual_rt)
        anchored_residual.append(endpoint.residual_rt)
        valid.append(endpoint.valid)
        saturated.append(endpoint.saturated_at_boundary)
        cues.append(endpoint.cue_applied)
        gains.append(endpoint.anchor_gain)
        ages.append(endpoint.anchor_age)
        corrections.append(endpoint.rate_correction_rt)
        bias_updates.append(endpoint.bias_update_applied)
        innovations.append(endpoint.anchor_innovation_norm)
        rejected.append(endpoint.anchor_rejected)
        rejection_reasons.append(endpoint.anchor_rejection_reason)
        reference_origins.append(endpoint.reference_origin_rt)
        reference_rebased.append(
            prediction.reference_rebased | endpoint.reference_rebased
        )
        reference_rebase_counts.append(endpoint.reference_rebase_count)
    return OrbitalRTGridHistory(
        node_id=node, timestamps=times.copy(), source_rt=np.asarray(source),
        predicted_rt=np.asarray(predicted), anchored_rt=np.asarray(anchored),
        prediction_residual_rt=np.asarray(prediction_residual),
        anchored_residual_rt=np.asarray(anchored_residual),
        valid=np.asarray(valid, dtype=bool),
        saturated_at_boundary=np.asarray(saturated, dtype=bool),
        cue_applied=np.asarray(cues, dtype=bool), anchor_gain=np.asarray(gains),
        anchor_age=np.asarray(ages),
        rate_correction_rt=np.asarray(corrections),
        bias_update_applied=np.asarray(bias_updates, dtype=bool),
        anchor_innovation_norm=np.asarray(innovations),
        anchor_rejected=np.asarray(rejected, dtype=bool),
        anchor_rejection_reason=np.asarray(rejection_reasons, dtype=str),
        reference_origin_rt=np.asarray(reference_origins),
        reference_rebased=np.asarray(reference_rebased, dtype=bool),
        reference_rebase_count=np.asarray(reference_rebase_counts, dtype=int),
    )


def _histories(mapping, size):
    result = {str(node): np.asarray(values, dtype=float)
              for node, values in mapping.items()}
    if not result:
        raise ValueError("At least one state history is required.")
    if any(values.shape != (size, 6) or np.any(~np.isfinite(values))
           for values in result.values()):
        raise ValueError("State histories must have shape (N, 6).")
    return result


def _series(mapping, node, size, default, dtype):
    values = (np.full(size, default, dtype=dtype)
              if mapping is None or node not in mapping
              else np.asarray(mapping[node], dtype=dtype))
    if values.shape != (size,):
        raise ValueError("Per-node RT controls must have shape (N,).")
    return values


def _rate_bias_series(mapping, node, size):
    values = np.asarray(
        np.zeros(2) if mapping is None or node not in mapping else mapping[node],
        dtype=float,
    )
    if values.shape == (2,):
        values = np.broadcast_to(values, (size, 2)).copy()
    if values.shape != (size, 2) or np.any(~np.isfinite(values)):
        raise ValueError(
            "Per-node RT rate bias must be a finite 2-vector or an (N, 2) series."
        )
    return values
