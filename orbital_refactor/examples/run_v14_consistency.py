"""Run the V14 range-only information-sharing consistency diagnosis."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v14_consistency import (
    run_v14_network_schmidt_monte_carlo,
    run_v14_range_consistency_monte_carlo,
)


def main() -> None:
    result = run_v14_range_consistency_monte_carlo()
    for strategy, summary in result.summary_by_strategy.items():
        print(
            f"{strategy:24s} "
            f"pos={summary.mean_position_rmse:8.3f} m  "
            f"vel={summary.mean_velocity_rmse:9.6f} m/s  "
            f"NEES={summary.mean_nees:9.3f}  "
            f"NEES95={summary.mean_nees_95_coverage:6.3f}  "
            f"NIS={summary.mean_nis:7.3f}  "
            f"NIS95={summary.mean_nis_95_coverage:6.3f}"
        )
    print("\nThree-satellite chain")
    network = run_v14_network_schmidt_monte_carlo()
    for strategy, summary in network.summary_by_strategy.items():
        print(
            f"{strategy:24s} "
            f"pos={summary.mean_position_rmse:8.3f} m  "
            f"vel={summary.mean_velocity_rmse:9.6f} m/s  "
            f"NEES={summary.mean_nees:9.3f}  "
            f"NEES95={summary.mean_nees_95_coverage:6.3f}  "
            f"NIS={summary.mean_nis:7.3f}  "
            f"NIS95={summary.mean_nis_95_coverage:6.3f}"
        )


if __name__ == "__main__":
    main()
