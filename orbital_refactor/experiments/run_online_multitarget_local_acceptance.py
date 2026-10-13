"""Run the paired single-host online multi-target acceptance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .online_multitarget_local_acceptance import (
    OnlineAcceptanceConfig,
    run_online_multitarget_local_acceptance,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observers", type=int, default=5)
    parser.add_argument("--targets", type=int, default=2)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=17)
    parser.add_argument(
        "--cann-profile", choices=("realtime", "research"),
        default="realtime",
    )
    parser.add_argument("--maximum-final-error-m", type=float, default=2_000.0)
    parser.add_argument("--deadline-seconds", type=float, default=1.0)
    parser.add_argument("--dropout-start", type=float)
    parser.add_argument("--dropout-end", type=float)
    parser.add_argument(
        "--dropout-modality",
        choices=("RADAR", "INFRARED", "OPTICAL", "ALL"),
    )
    parser.add_argument("--dropout-target-index", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main():
    args = build_parser().parse_args()
    result = run_online_multitarget_local_acceptance(OnlineAcceptanceConfig(
        observer_count=args.observers,
        target_count=args.targets,
        duration_seconds=args.duration,
        sample_interval_seconds=args.dt,
        seed=args.seed,
        image_size=args.image_size,
        cann_profile=args.cann_profile,
        maximum_final_position_error_m=args.maximum_final_error_m,
        deadline_seconds=args.deadline_seconds,
        dropout_start_seconds=args.dropout_start,
        dropout_end_seconds=args.dropout_end,
        dropout_modality=args.dropout_modality,
        dropout_target_index=args.dropout_target_index,
    ))
    payload = result.to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
