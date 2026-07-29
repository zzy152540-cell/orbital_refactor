"""Monte Carlo validation for v13.4 attitude-aware BODY-angle fusion."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.run_v13_4_attitude_coupling import run_comparison

METHODS = (
    "truth_attitude",
    "mekf_with_covariance",
    "mekf_without_covariance",
)
METRICS = (
    "position_rmse",
    "velocity_rmse",
    "mean_angle_nis",
    "mean_orbit_nees",
)


def run_monte_carlo(
    seeds,
    *,
    duration: float = 120.0,
    step: float = 2.0,
):
    """Return raw method rows and aggregate rows for the supplied seeds."""

    raw_rows = []
    comparisons = {}
    for seed in (int(value) for value in seeds):
        try:
            comparison = run_comparison(
                duration=duration,
                step=step,
                random_seed=seed,
            )
        except Exception as exc:
            for method in METHODS:
                raw_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        **{metric: np.nan for metric in METRICS},
                        "mean_attitude_error_deg": np.nan,
                        "failed": True,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            continue
        comparisons[seed] = comparison
        for method in METHODS:
            metrics = getattr(comparison, method)
            raw_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    **{metric: float(getattr(metrics, metric)) for metric in METRICS},
                    "mean_attitude_error_deg": comparison.mean_attitude_error_deg,
                    "failed": False,
                    "error": "",
                }
            )
    return raw_rows, _aggregate(raw_rows, comparisons)


def _aggregate(raw_rows, comparisons):
    summary_rows = []
    for method in METHODS:
        method_rows = [row for row in raw_rows if row["method"] == method]
        successful = [row for row in method_rows if not row["failed"]]
        summary = {
            "method": method,
            "runs": len(method_rows),
            "successful_runs": len(successful),
            "failure_rate": (
                1.0 - len(successful) / len(method_rows) if method_rows else np.nan
            ),
        }
        for metric in (*METRICS, "mean_attitude_error_deg"):
            values = np.asarray(
                [row[metric] for row in successful],
                dtype=float,
            )
            for statistic, value in _statistics(values).items():
                summary[f"{metric}_{statistic}"] = value
        summary_rows.append(summary)

    paired = list(comparisons.values())
    summary_rows.append(
        {
            "method": "paired_covariance_effect",
            "runs": len(raw_rows) // len(METHODS),
            "successful_runs": len(paired),
            "failure_rate": (
                1.0 - len(paired) / (len(raw_rows) // len(METHODS))
                if raw_rows
                else np.nan
            ),
            "position_rmse_win_rate": _win_rate(
                paired, "position_rmse", lower_is_better=True
            ),
            "velocity_rmse_win_rate": _win_rate(
                paired, "velocity_rmse", lower_is_better=True
            ),
            "angle_nis_consistency_win_rate": _nis_consistency_win_rate(paired),
        }
    )
    return summary_rows


def _statistics(values):
    if values.size == 0:
        return {
            key: np.nan for key in ("mean", "std", "p05", "p50", "p95")
        }
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "p05": float(np.percentile(values, 5.0)),
        "p50": float(np.percentile(values, 50.0)),
        "p95": float(np.percentile(values, 95.0)),
    }


def _win_rate(comparisons, metric, *, lower_is_better):
    if not comparisons:
        return np.nan
    wins = [
        getattr(item.mekf_with_covariance, metric)
        < getattr(item.mekf_without_covariance, metric)
        for item in comparisons
    ]
    if not lower_is_better:
        wins = [not value for value in wins]
    return float(np.mean(wins))


def _nis_consistency_win_rate(comparisons):
    if not comparisons:
        return np.nan
    return float(
        np.mean(
            [
                abs(item.mekf_with_covariance.mean_angle_nis - 2.0)
                < abs(item.mekf_without_covariance.mean_angle_nis - 2.0)
                for item in comparisons
            ]
        )
    )


def export_results(raw_rows, summary_rows, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output.with_name(f"{output.stem}_summary{output.suffix}")
    _write_csv(output, raw_rows)
    _write_csv(summary_path, summary_rows)
    return output, summary_path


def _write_csv(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=20260729)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--step", type=float, default=2.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/v13_4_attitude_monte_carlo.csv"),
    )
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
    seeds = range(args.base_seed, args.base_seed + args.seeds)
    raw_rows, summary_rows = run_monte_carlo(
        seeds,
        duration=args.duration,
        step=args.step,
    )
    raw_path, summary_path = export_results(raw_rows, summary_rows, args.output)
    paired = next(
        row for row in summary_rows if row["method"] == "paired_covariance_effect"
    )
    print(f"Raw results: {raw_path}")
    print(f"Summary: {summary_path}")
    print(
        "Attitude-covariance paired win rates: "
        f"position={paired['position_rmse_win_rate']:.3f}, "
        f"velocity={paired['velocity_rmse_win_rate']:.3f}, "
        f"NIS consistency={paired['angle_nis_consistency_win_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
