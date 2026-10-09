from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.multitarget_communication_robustness_acceptance import (
    run_multitarget_communication_robustness_acceptance,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multi-seed multi-target communication acceptance."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--dt", type=float, default=10.0)
    parser.add_argument("--observers", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run_multitarget_communication_robustness_acceptance(
        seeds=tuple(args.seeds),
        duration=args.duration,
        dt=args.dt,
        observer_count=args.observers,
        outage_node_id=f"observer-{args.observers}",
        outage_start=args.duration / 3.0,
        outage_end=args.duration * 5.0 / 12.0,
        topology_switch_time=args.duration / 2.0,
    )
    payload = {
        "seeds": list(result.seeds),
        "minimumFinalTargetRetentionFraction": (
            result.minimum_final_target_retention_fraction
        ),
        "minimumFinalConsensusAvailabilityFraction": (
            result.minimum_final_consensus_availability_fraction
        ),
        "maximumPositionErrorM": result.maximum_position_error_m,
        "positionErrorP95M": result.position_error_p95_m,
        "maximumVelocityErrorMps": result.maximum_velocity_error_mps,
        "velocityErrorP95Mps": result.velocity_error_p95_mps,
        "maximumConsecutiveMissingEpochs": (
            result.maximum_consecutive_missing_epochs
        ),
        "maximumOutageRecoveryDelaySeconds": (
            result.maximum_outage_recovery_delay_seconds
        ),
        "totalRejectedMessageCount": result.total_rejected_message_count,
        "totalCovariancePsdFailureCount": (
            result.total_covariance_psd_failure_count
        ),
        "meanElapsedSeconds": result.mean_elapsed_seconds,
        "passed": result.passed,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
