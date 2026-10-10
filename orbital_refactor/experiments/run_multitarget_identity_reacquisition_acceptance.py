from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.multitarget_identity_reacquisition_acceptance import (
    run_multitarget_identity_reacquisition_acceptance,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run close-crossing multi-target identity acceptance."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--observers", type=int, default=3)
    parser.add_argument("--duration", type=float, default=400.0)
    parser.add_argument("--dt", type=float, default=10.0)
    parser.add_argument("--outage-start", type=float, default=170.0)
    parser.add_argument("--outage-end", type=float, default=230.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run_multitarget_identity_reacquisition_acceptance(
        seed=args.seed,
        observer_count=args.observers,
        duration=args.duration,
        dt=args.dt,
        outage_start=args.outage_start,
        outage_end=args.outage_end,
    )
    payload = {
        "seed": result.seed,
        "observerCount": result.observer_count,
        "epochCount": result.epoch_count,
        "minimumTargetSeparationM": result.minimum_target_separation_m,
        "identitySwitchCount": result.identity_switch_count,
        "falseCooperativeBirthCount": result.false_cooperative_birth_count,
        "reacquiredCooperativeTargetCount": (
            result.reacquired_cooperative_target_count
        ),
        "finalCooperativeTargetCount": result.final_cooperative_target_count,
        "passed": result.passed,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
