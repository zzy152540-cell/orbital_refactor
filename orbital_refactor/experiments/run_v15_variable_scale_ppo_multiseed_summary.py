from __future__ import annotations

import argparse
from pathlib import Path

from experiments.variable_scale_ppo_multiseed import (
    generate_variable_scale_ppo_multiseed_visualization,
    write_variable_scale_ppo_multiseed_summary,
)


def main(argv=None) -> tuple[Path, Path]:
    parser = argparse.ArgumentParser(
        description="Aggregate and visualize frozen variable-scale PPO seeds."
    )
    parser.add_argument("--summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    arguments = parser.parse_args(argv)
    output = write_variable_scale_ppo_multiseed_summary(
        tuple(arguments.summaries), arguments.output,
    )
    figure = generate_variable_scale_ppo_multiseed_visualization(
        output, arguments.figure,
    )
    print(output)
    print(figure)
    return output, figure


if __name__ == "__main__":
    main()
