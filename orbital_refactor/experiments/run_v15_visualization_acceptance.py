from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from visualization.acceptance import compare_visualization_algorithm_payloads


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare paired visualization recordings for non-intrusion."
    )
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    args = parser.parse_args()
    result = compare_visualization_algorithm_payloads(
        args.baseline, args.candidate,
    )
    print(json.dumps(asdict(result), indent=2))
    if not result.algorithm_payloads_identical:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
