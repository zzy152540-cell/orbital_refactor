from __future__ import annotations

import argparse
from pathlib import Path

from experiments.variable_scale_ppo_visualization import (
    generate_variable_scale_ppo_training_visualization,
)


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(
        description="Visualize variable-scale PPO training and evaluation metrics."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    output = generate_variable_scale_ppo_training_visualization(
        arguments.summary, arguments.output,
    )
    print(output)
    return output


if __name__ == "__main__":
    main()
