from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.multitarget_multimodal_dynamic_visibility import (
    run_multitarget_multimodal_dynamic_visibility,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run distributed multi-target dynamic-visibility acceptance."
    )
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--dt", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--observers", type=int, default=3)
    parser.add_argument("--communication-delay", type=float, default=0.0)
    parser.add_argument("--packet-loss", type=float, default=0.0)
    parser.add_argument("--outage-node")
    parser.add_argument("--outage-start", type=float)
    parser.add_argument("--outage-end", type=float)
    parser.add_argument(
        "--topology", choices=("fully_connected", "chain", "ring"),
        default="fully_connected",
    )
    parser.add_argument("--topology-switch-time", type=float)
    parser.add_argument(
        "--topology-after-switch",
        choices=("fully_connected", "chain", "ring"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run_multitarget_multimodal_dynamic_visibility(
        duration=args.duration,
        dt=args.dt,
        random_seed=args.seed,
        observer_count=args.observers,
        communication_delay_seconds=args.communication_delay,
        packet_loss_rate=args.packet_loss,
        outage_node_id=args.outage_node,
        outage_start=args.outage_start,
        outage_end=args.outage_end,
        topology=args.topology,
        topology_switch_time=args.topology_switch_time,
        topology_after_switch=args.topology_after_switch,
    )
    payload = {
        "durationSeconds": result.timestamps[-1] - result.timestamps[0],
        "sampleIntervalSeconds": args.dt,
        "observerIds": list(result.observer_ids),
        "targetIds": list(result.target_ids),
        "opportunityCountByModality": result.opportunity_count_by_modality,
        "visibleCountByModality": result.visible_count_by_modality,
        "visibilitySwitchCountByModality": result.visibility_switch_count_by_modality,
        "validMeasurementCountByModality": result.valid_measurement_count_by_modality,
        "feedbackApplicationCount": result.feedback_application_count,
        "maximumConsensusContributorCount": max(
            result.consensus_contributor_count_by_epoch_node_target.values(),
            default=0,
        ),
        "maximumFinalPositionErrorM": result.maximum_final_position_error_m,
        "maximumFinalVelocityErrorMps": result.maximum_final_velocity_error_mps,
        "covariancePsdFailureCount": result.covariance_psd_failure_count,
        "receivedMessageCount": result.received_message_count,
        "rejectedMessageCount": result.rejected_message_count,
        "recoveredSourceEventCount": result.recovered_source_event_count,
        "missingTargetEstimateCount": result.missing_target_estimate_count,
        "outageEpochCount": result.outage_epoch_count,
        "pendingMessageCountFinal": result.pending_message_count_final,
        "maximumPositionErrorM": result.maximum_position_error_m,
        "maximumVelocityErrorMps": result.maximum_velocity_error_mps,
        "topologySwitchCount": result.topology_switch_count,
        "finalTargetRetentionFraction": result.final_target_retention_fraction,
        "finalConsensusAvailabilityFraction": (
            result.final_consensus_availability_fraction
        ),
        "maximumConsecutiveMissingEpochs": (
            result.maximum_consecutive_missing_epochs
        ),
        "outageRecoveryDelaySeconds": result.outage_recovery_delay_seconds,
        "elapsedSeconds": result.elapsed_seconds,
        "meanSecondsPerEpoch": result.elapsed_seconds / len(result.timestamps),
        "finalErrorsByNodeTarget": [
            {
                "observerId": observer_id,
                "targetId": target_id,
                "positionErrorM": position_error,
                "velocityErrorMps": result.final_velocity_error_mps_by_node_target[
                    (observer_id, target_id)
                ],
            }
            for (observer_id, target_id), position_error in sorted(
                result.final_position_error_m_by_node_target.items()
            )
        ],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
