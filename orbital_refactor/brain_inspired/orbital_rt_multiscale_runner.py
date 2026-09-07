from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from brain_inspired.orbital_rt_grid_runner import (
    _histories,
    _rate_bias_series,
    _series,
)
from brain_inspired.orbital_rt_multiscale_state import (
    OrbitalRTMultiScaleConfig,
    OrbitalRTMultiScaleState,
)
from orbital_core.dynamics import rk4_step_absolute

Array = np.ndarray


@dataclass(frozen=True)
class OrbitalRTMultiScaleHistory:
    node_id: str
    timestamps: Array
    source_rt: Array
    decoded_rt: Array
    residual_rt: Array
    coarse_center_rt: Array
    fine_residual_rt: Array
    cross_scale_disagreement_rt: Array
    coarse_cell_changed: Array
    valid: Array
    saturated_at_boundary: Array
    cue_applied: Array
    anchor_rejected: Array


def run_orbital_rt_multiscale_states(
    *, timestamps: Array,
    posterior_state_history_by_node: Mapping[str, Array],
    reference_state_history_by_node: Mapping[str, Array],
    anchor_state_history_by_node: Mapping[str, Array] | None = None,
    anchor_mask_by_node: Mapping[str, Array] | None = None,
    anchor_confidence_by_node: Mapping[str, Array] | None = None,
    rate_bias_rt_by_node: Mapping[str, Array] | None = None,
    config: OrbitalRTMultiScaleConfig | None = None,
):
    times = np.asarray(timestamps, dtype=float).reshape(-1)
    if (times.size == 0 or np.any(~np.isfinite(times))
            or (times.size > 1 and np.any(np.diff(times) <= 0.0))):
        raise ValueError("timestamps must be finite, nonempty, and increasing.")
    posterior = _histories(posterior_state_history_by_node, times.size)
    reference = _histories(reference_state_history_by_node, times.size)
    anchors = (posterior if anchor_state_history_by_node is None else
               _histories(anchor_state_history_by_node, times.size))
    if set(posterior) != set(reference) or set(posterior) != set(anchors):
        raise ValueError(
            "Posterior, reference, and anchor histories must cover identical nodes."
        )
    result = {}
    for node, states in posterior.items():
        result[node] = _run_node(
            node=node, times=times, posterior=states, reference=reference[node],
            anchors=anchors[node],
            mask=_series(anchor_mask_by_node, node, times.size, False, bool),
            confidence=_series(
                anchor_confidence_by_node, node, times.size, 1.0, float,
            ),
            bias=_rate_bias_series(rate_bias_rt_by_node, node, times.size),
            config=config,
        )
    return result


def _run_node(*, node, times, posterior, reference, anchors, mask, confidence,
              bias, config):
    state = OrbitalRTMultiScaleState(node_id=node, config=config)
    snapshots = [state.initialize(
        timestamp=times[0], state_eci=posterior[0],
        reference_state_eci=reference[0],
    )]
    for index in range(1, times.size):
        dt = float(times[index] - times[index - 1])
        prior = rk4_step_absolute(posterior[index - 1], dt)
        state.predict(
            timestamp=times[index], predicted_state_eci=prior,
            reference_state_eci=reference[index], rate_bias_rt=bias[index],
        )
        snapshots.append(state.anchor(
            timestamp=times[index], posterior_state_eci=anchors[index],
            reference_state_eci=reference[index], confidence=confidence[index],
            trusted=bool(mask[index]),
        ))
    return OrbitalRTMultiScaleHistory(
        node_id=node, timestamps=times.copy(),
        source_rt=np.asarray([item.source_rt for item in snapshots]),
        decoded_rt=np.asarray([item.decoded_rt for item in snapshots]),
        residual_rt=np.asarray([item.residual_rt for item in snapshots]),
        coarse_center_rt=np.asarray([
            item.coarse_center_rt for item in snapshots
        ]),
        fine_residual_rt=np.asarray([
            item.fine_residual_rt for item in snapshots
        ]),
        cross_scale_disagreement_rt=np.asarray([
            item.cross_scale_disagreement_rt for item in snapshots
        ]),
        coarse_cell_changed=np.asarray([
            item.coarse_cell_changed for item in snapshots
        ], dtype=bool),
        valid=np.asarray([item.valid for item in snapshots], dtype=bool),
        saturated_at_boundary=np.asarray([
            item.saturated_at_boundary for item in snapshots
        ], dtype=bool),
        cue_applied=np.asarray([
            item.cue_applied for item in snapshots
        ], dtype=bool),
        anchor_rejected=np.asarray([
            item.anchor_rejected for item in snapshots
        ], dtype=bool),
    )
