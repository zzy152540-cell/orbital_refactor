from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from experiments.bilateral_distributed_discovery_pilot import (
    run_bilateral_distributed_discovery_pilot,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--identity-exchange-hops", type=int, default=4)
    parser.add_argument("--discovery-exchange-hops", type=int, default=4)
    parser.add_argument("--state-exchange-hops", type=int, default=1)
    parser.add_argument("--no-cann", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/external_acceptance/"
            "bilateral_distributed_discovery_pilot.json"
        ),
    )
    args = parser.parse_args()
    result = run_bilateral_distributed_discovery_pilot(
        duration_seconds=args.duration,
        sample_interval_seconds=args.dt,
        seed=args.seed,
        identity_exchange_hops=args.identity_exchange_hops,
        discovery_exchange_hops=args.discovery_exchange_hops,
        state_exchange_hops=args.state_exchange_hops,
        enable_cann=not args.no_cann,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for name, value in asdict(result).items():
        print(f"{name}={value}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
