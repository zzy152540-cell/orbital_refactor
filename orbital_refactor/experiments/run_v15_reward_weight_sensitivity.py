from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.reward_weight_sensitivity import (
    generate_reward_weight_scan_figure,
    read_policy_diagnostic_csv,
    scan_reward_weight_sensitivity,
    write_reward_weight_scan_csv,
)


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(description="Scan V15 reward weights offline.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    arguments = parser.parse_args(argv)
    summary = scan_reward_weight_sensitivity(
        read_policy_diagnostic_csv(arguments.input)
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_reward_weight_scan_csv(summary, arguments.csv)
    generate_reward_weight_scan_figure(summary, arguments.figure)
    print(arguments.output)
    return arguments.output


if __name__ == "__main__":
    main()
