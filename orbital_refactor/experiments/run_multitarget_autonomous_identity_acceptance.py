from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.multitarget_autonomous_identity_acceptance import (
    run_multitarget_autonomous_identity_acceptance,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run raw autonomous multi-target identity acceptance."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--observers", type=int, default=3)
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--dt", type=float, default=10.0)
    parser.add_argument("--outage-start", type=float, default=170.0)
    parser.add_argument("--outage-end", type=float, default=230.0)
    parser.add_argument("--image-size", type=int, default=17)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run_multitarget_autonomous_identity_acceptance(
        seed=args.seed,
        observer_count=args.observers,
        duration=args.duration,
        dt=args.dt,
        outage_start=args.outage_start,
        outage_end=args.outage_end,
        image_size=args.image_size,
    )
    payload = {
        "seed": result.seed,
        "observerCount": result.observer_count,
        "epochCount": result.epoch_count,
        "minimumTargetSeparationM": result.minimum_target_separation_m,
        "initialBirthCount": result.initial_birth_count,
        "postOutageBirthCount": result.post_outage_birth_count,
        "identitySwitchCount": result.identity_switch_count,
        "falseCooperativeBirthCount": result.false_cooperative_birth_count,
        "reacquiredCooperativeTargetCount": (
            result.reacquired_cooperative_target_count
        ),
        "finalTrackingTargetCount": result.final_tracking_target_count,
        "finalConsensusTargetCount": result.final_consensus_target_count,
        "maximumFinalPositionErrorM": result.maximum_final_position_error_m,
        "passed": result.passed,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
