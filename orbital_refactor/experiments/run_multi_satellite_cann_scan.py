from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.multi_satellite_cann_comparison import (
    run_multi_satellite_cann_comparison,
)


DEFAULT_DROPOUTS = {
    "sat_01": {"RADAR": ((600.0, 700.0),)},
    "sat_02": {"INFRARED": ((700.0, 800.0),)},
    "sat_03": {"OPTICAL": ((1300.0, 1400.0),)},
}


def run_recovery_scan(*, seeds, output_path, duration=1800.0, dt=2.0):
    """Run and atomically persist each completed recovery-fault seed."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = _load_existing(output, duration=duration, dt=dt)
    rows = payload["results_by_seed"]
    for seed in seeds:
        key = str(int(seed))
        if key in rows:
            continue
        result = run_multi_satellite_cann_comparison(
            duration=duration, dt=dt, seed=int(seed),
            dropout_windows_by_node=DEFAULT_DROPOUTS,
            recovery_fault_samples=2,
            include_confirmation_baseline=True,
        )
        rows[key] = result["summary"]
        payload["aggregate"] = _aggregate(rows)
        _atomic_write_json(output, payload)
    return payload


def _load_existing(path, *, duration, dt):
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("duration") != duration or payload.get("dt") != dt:
            raise ValueError("Existing scan configuration does not match.")
        return payload
    return {
        "duration": float(duration), "dt": float(dt),
        "scenario": "three_node_multimodal_recovery_faults",
        "dropout_windows_by_node": DEFAULT_DROPOUTS,
        "recovery_fault_samples": 2,
        "results_by_seed": {}, "aggregate": {},
    }


def _aggregate(rows):
    values = list(rows.values())
    fields = (
        "baseline_position_rmse_m", "cann_position_rmse_m",
        "confirmation_position_rmse_m",
        "cann_minus_confirmation_position_rmse_m",
        "position_change_m", "baseline_velocity_rmse_mps",
        "cann_velocity_rmse_mps", "velocity_change_mps",
    )
    result = {"completed_seed_count": len(values)}
    for field in fields:
        samples = np.asarray([row[field] for row in values], dtype=float)
        result[field] = {
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples)),
            "minimum": float(np.min(samples)),
            "maximum": float(np.max(samples)),
        }
    return result


def _atomic_write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--duration", type=float, default=1800.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/cann/multi_satellite_recovery_scan.json"),
    )
    arguments = parser.parse_args()
    payload = run_recovery_scan(
        seeds=arguments.seeds, output_path=arguments.output,
        duration=arguments.duration, dt=arguments.dt,
    )
    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
