"""Run the V14 four-architecture three-satellite comparison."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v14_comparison import (
    build_v14_comparison_case,
    export_v14_comparison,
    run_v14_comparison,
)


def main() -> None:
    result = run_v14_comparison(build_v14_comparison_case())
    paths = export_v14_comparison(result, PROJECT_ROOT / "results" / "v14_comparison")
    for algorithm, metrics in result.metrics_by_algorithm.items():
        print(
            f"{algorithm:28s} "
            f"position={metrics.fleet_position_rmse:10.3f} m  "
            f"velocity={metrics.fleet_velocity_rmse:10.6f} m/s  "
            f"NEES={metrics.mean_nees:10.3f}"
        )
    print(f"JSON: {paths['json']}")
    print(f"CSV:  {paths['csv']}")


if __name__ == "__main__":
    main()
