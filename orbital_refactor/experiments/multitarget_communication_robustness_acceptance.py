"""Multi-seed communication robustness acceptance for multi-target tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .multitarget_multimodal_dynamic_visibility import (
    MultitargetMultimodalVisibilityResult,
    run_multitarget_multimodal_dynamic_visibility,
)


@dataclass(frozen=True)
class CommunicationRobustnessAcceptance:
    seeds: tuple[int, ...]
    runs: tuple[MultitargetMultimodalVisibilityResult, ...]
    minimum_final_target_retention_fraction: float
    minimum_final_consensus_availability_fraction: float
    maximum_position_error_m: float
    position_error_p95_m: float
    maximum_velocity_error_mps: float
    velocity_error_p95_mps: float
    maximum_consecutive_missing_epochs: int
    maximum_outage_recovery_delay_seconds: float
    total_rejected_message_count: int
    total_covariance_psd_failure_count: int
    mean_elapsed_seconds: float
    passed: bool


def run_multitarget_communication_robustness_acceptance(
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    duration: float = 600.0,
    dt: float = 10.0,
    observer_count: int = 5,
    communication_delay_seconds: float = 10.0,
    packet_loss_rate: float = 0.2,
    outage_node_id: str = "observer-5",
    outage_start: float = 200.0,
    outage_end: float = 250.0,
    topology_switch_time: float = 300.0,
) -> CommunicationRobustnessAcceptance:
    selected_seeds = tuple(int(value) for value in seeds)
    if not selected_seeds:
        raise ValueError("At least one random seed is required.")
    runs = tuple(
        run_multitarget_multimodal_dynamic_visibility(
            duration=duration,
            dt=dt,
            random_seed=seed,
            observer_count=observer_count,
            communication_delay_seconds=communication_delay_seconds,
            packet_loss_rate=packet_loss_rate,
            outage_node_id=outage_node_id,
            outage_start=outage_start,
            outage_end=outage_end,
            topology="fully_connected",
            topology_switch_time=topology_switch_time,
            topology_after_switch="chain",
        )
        for seed in selected_seeds
    )
    position_errors = np.asarray([
        result.maximum_position_error_m for result in runs
    ])
    velocity_errors = np.asarray([
        result.maximum_velocity_error_mps for result in runs
    ])
    recovery_delays = [
        result.outage_recovery_delay_seconds
        for result in runs
        if result.outage_recovery_delay_seconds is not None
    ]
    minimum_retention = min(
        result.final_target_retention_fraction for result in runs
    )
    minimum_consensus_availability = min(
        result.final_consensus_availability_fraction for result in runs
    )
    maximum_position_error = float(np.max(position_errors))
    maximum_velocity_error = float(np.max(velocity_errors))
    psd_failures = sum(
        result.covariance_psd_failure_count for result in runs
    )
    maximum_recovery_delay = (
        float(max(recovery_delays)) if recovery_delays else float("inf")
    )
    passed = bool(
        minimum_retention == 1.0
        and minimum_consensus_availability >= 0.8
        and maximum_position_error < 10_000.0
        and maximum_velocity_error < 50.0
        and psd_failures == 0
        and len(recovery_delays) == len(runs)
        and maximum_recovery_delay <= 2.0 * dt
    )
    return CommunicationRobustnessAcceptance(
        seeds=selected_seeds,
        runs=runs,
        minimum_final_target_retention_fraction=minimum_retention,
        minimum_final_consensus_availability_fraction=(
            minimum_consensus_availability
        ),
        maximum_position_error_m=maximum_position_error,
        position_error_p95_m=float(np.percentile(position_errors, 95.0)),
        maximum_velocity_error_mps=maximum_velocity_error,
        velocity_error_p95_mps=float(np.percentile(velocity_errors, 95.0)),
        maximum_consecutive_missing_epochs=max(
            result.maximum_consecutive_missing_epochs for result in runs
        ),
        maximum_outage_recovery_delay_seconds=maximum_recovery_delay,
        total_rejected_message_count=sum(
            result.rejected_message_count for result in runs
        ),
        total_covariance_psd_failure_count=psd_failures,
        mean_elapsed_seconds=float(np.mean([
            result.elapsed_seconds for result in runs
        ])),
        passed=passed,
    )
