"""Controlled close-crossing and long-occlusion identity acceptance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbital_core.dynamics import rk4_step_absolute
from tracking import CooperativeTrackIdentityResolver, TargetNodeReport


@dataclass(frozen=True)
class IdentityReacquisitionAcceptance:
    seed: int
    observer_count: int
    epoch_count: int
    minimum_target_separation_m: float
    identity_switch_count: int
    false_cooperative_birth_count: int
    reacquired_cooperative_target_count: int
    final_cooperative_target_count: int
    passed: bool


def run_multitarget_identity_reacquisition_acceptance(
    *,
    seed: int = 0,
    observer_count: int = 3,
    duration: float = 400.0,
    dt: float = 10.0,
    outage_start: float = 170.0,
    outage_end: float = 230.0,
) -> IdentityReacquisitionAcceptance:
    """Cross two close targets while every observer is simultaneously blind."""

    if observer_count < 2:
        raise ValueError("observer_count must be at least two.")
    if dt <= 0.0 or duration <= 0.0:
        raise ValueError("duration and dt must be positive.")
    if not 0.0 < outage_start <= outage_end < duration:
        raise ValueError("The outage must lie strictly inside the scenario.")
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt)
    rng = np.random.default_rng(seed)
    covariance = np.diag([400.0, 400.0, 400.0, 0.04, 0.04, 0.04])
    truth = {
        "alpha": np.array([7.0e6, -1_000.0, 0.0, 0.0, 7_505.0, 0.0]),
        "beta": np.array([7.0e6, 1_000.0, 0.0, 0.0, 7_495.0, 0.0]),
    }
    resolver = CooperativeTrackIdentityResolver(
        position_gate_m=20_000.0,
        velocity_gate_mps=50.0,
        minimum_match_margin=0.02,
        max_reacquisition_gap_seconds=duration,
    )
    expected_by_physical_target: dict[str, str] = {}
    identity_switch_count = 0
    created_ids = set()
    reacquired_ids = set()
    minimum_separation = np.inf
    retired_for_outage = False

    for epoch_index, timestamp in enumerate(timestamps):
        if epoch_index:
            truth = {
                target_id: rk4_step_absolute(state, dt)
                for target_id, state in truth.items()
            }
        minimum_separation = min(
            minimum_separation,
            float(np.linalg.norm(truth["alpha"][:3] - truth["beta"][:3])),
        )
        in_outage = outage_start <= timestamp <= outage_end
        if in_outage and not retired_for_outage:
            for node_index in range(1, observer_count + 1):
                observer_id = f"observer-{node_index}"
                for target_id in sorted(truth):
                    resolver.retire_local_track(
                        observer_id=observer_id,
                        local_target_id=f"{target_id}-pre-{node_index}",
                        local_track_id=f"track-{target_id}-pre-{node_index}",
                    )
            retired_for_outage = True
        if in_outage:
            continue

        phase = "post" if timestamp > outage_end else "pre"
        reports = []
        physical_by_local_key = {}
        for node_index in range(1, observer_count + 1):
            observer_id = f"observer-{node_index}"
            for target_id, state in sorted(truth.items()):
                local_target_id = f"{target_id}-{phase}-{node_index}"
                local_track_id = f"track-{target_id}-{phase}-{node_index}"
                noisy_state = state + np.hstack((
                    rng.normal(0.0, 20.0, size=3),
                    rng.normal(0.0, 0.05, size=3),
                ))
                reports.append(TargetNodeReport(
                    observer_id=observer_id,
                    target_id=local_target_id,
                    track_id=local_track_id,
                    timestamp=float(timestamp),
                    state_eci=noisy_state,
                    covariance_eci=covariance,
                    quality_score=0.9,
                ))
                physical_by_local_key[(observer_id, local_target_id)] = target_id

        resolution = resolver.resolve(reports)
        created_ids.update(resolution.created_cooperative_target_ids)
        reacquired_ids.update(resolution.reacquired_cooperative_target_ids)
        for assignment in resolution.assignments:
            physical_target = physical_by_local_key[
                (assignment.observer_id, assignment.local_target_id)
            ]
            if physical_target not in expected_by_physical_target:
                expected_by_physical_target[physical_target] = (
                    assignment.cooperative_target_id
                )
            elif (
                assignment.cooperative_target_id
                != expected_by_physical_target[physical_target]
            ):
                identity_switch_count += 1

    false_births = max(0, len(created_ids) - len(truth))
    final_target_count = len(resolver.cooperative_target_ids)
    passed = bool(
        identity_switch_count == 0
        and false_births == 0
        and len(reacquired_ids) == len(truth)
        and final_target_count == len(truth)
    )
    return IdentityReacquisitionAcceptance(
        seed=int(seed),
        observer_count=int(observer_count),
        epoch_count=len(timestamps),
        minimum_target_separation_m=float(minimum_separation),
        identity_switch_count=identity_switch_count,
        false_cooperative_birth_count=false_births,
        reacquired_cooperative_target_count=len(reacquired_ids),
        final_cooperative_target_count=final_target_count,
        passed=passed,
    )
