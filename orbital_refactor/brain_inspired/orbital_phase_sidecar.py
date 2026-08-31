from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brain_inspired.orbital_phase_adapter import (
    OrbitalPlaneFrame,
    extract_orbital_phase_state,
)
from brain_inspired.passive_phase_observer import PassiveRingCANNObserver
from brain_inspired.ring_cann import RingCANNConfig

Array = np.ndarray


@dataclass(frozen=True)
class OrbitalPhaseSidecarHistory:
    timestamps: Array
    source_phase: Array
    source_phase_rate: Array
    decoded_phase: Array
    phase_residual: Array
    bump_concentration: Array
    bump_width: Array
    cross_track_position: Array
    cue_applied: Array
    valid: Array
    source_id: str | None = None


def run_orbital_phase_sidecar(
    *, timestamps: Array, state_history_eci: Array, frame: OrbitalPlaneFrame,
    cue_interval_samples: int | None = None,
    source_id: str | None = None,
    cann_config: RingCANNConfig | None = None,
) -> OrbitalPhaseSidecarHistory:
    """Observe an ECI state history without changing its estimator trajectory."""

    times = np.asarray(timestamps, dtype=float).reshape(-1)
    states = np.asarray(state_history_eci, dtype=float)
    if times.size == 0 or np.any(~np.isfinite(times)):
        raise ValueError("timestamps must be finite and nonempty.")
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError("timestamps must be strictly increasing.")
    if states.shape != (times.size, 6) or np.any(~np.isfinite(states)):
        raise ValueError("state_history_eci must be finite with shape (N, 6).")
    if cue_interval_samples is not None and cue_interval_samples < 1:
        raise ValueError("cue_interval_samples must be positive or None.")

    phases = tuple(
        extract_orbital_phase_state(
            timestamp=timestamp, state_eci=states[index], frame=frame,
            source_id=source_id,
        )
        for index, timestamp in enumerate(times)
    )
    observer = PassiveRingCANNObserver(cann_config or RingCANNConfig())
    outputs = [observer.initialize(
        phase=phases[0].argument_of_latitude,
        timestamp=phases[0].timestamp, source_id=source_id,
    )]
    cue_flags = [False]
    for index, phase_state in enumerate(phases[1:], start=1):
        use_hint = (
            cue_interval_samples is not None
            and index % cue_interval_samples == 0
        )
        outputs.append(observer.update(
            phase_state.as_periodic_input(use_phase_hint=use_hint)
        ))
        cue_flags.append(use_hint)
    source_phase = np.asarray([
        phase.argument_of_latitude for phase in phases
    ])
    decoded = np.asarray([output.decoded_phase for output in outputs])
    residual = (decoded - source_phase + np.pi) % (2.0 * np.pi) - np.pi
    return OrbitalPhaseSidecarHistory(
        timestamps=times.copy(), source_phase=source_phase,
        source_phase_rate=np.asarray([
            phase.argument_of_latitude_rate for phase in phases
        ]),
        decoded_phase=decoded, phase_residual=residual,
        bump_concentration=np.asarray([
            output.bump_concentration for output in outputs
        ]),
        bump_width=np.asarray([output.bump_width for output in outputs]),
        cross_track_position=np.asarray([
            phase.cross_track_position for phase in phases
        ]),
        cue_applied=np.asarray(cue_flags, dtype=bool),
        valid=np.asarray([output.valid for output in outputs], dtype=bool),
        source_id=source_id,
    )
