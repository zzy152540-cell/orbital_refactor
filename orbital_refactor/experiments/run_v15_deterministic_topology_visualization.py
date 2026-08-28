from __future__ import annotations

import argparse
from pathlib import Path

from experiments.deterministic_topology_visualization import (
    generate_deterministic_topology_visualization,
)


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(
        description="Visualize paired V15 deterministic topology policies."
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    output = generate_deterministic_topology_visualization(
        arguments.csv, arguments.output,
    )
    print(output)
    return output


if __name__ == "__main__":
    main()
