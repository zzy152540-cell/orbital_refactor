from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SCENARIOS = (
    ("none", "No outage"),
    ("ir", "IR outage"),
    ("opt", "Optical outage"),
    ("rad", "Radar outage"),
    ("all", "All-modal outage"),
)


def generate_comparison(results_root: Path, output_dir: Path) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = []
    for key, label in SCENARIOS:
        path = results_root / f"single_satellite_outage_{key}" / "summary.json"
        values = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"scenario": key, "label": label, **values})

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "comparison.csv"
    fields = (
        "scenario", "label", "position_rmse_m", "position_rmse_outage_m",
        "position_rmse_recovery_m", "velocity_rmse_mps",
        "velocity_rmse_outage_mps", "velocity_rmse_recovery_mps",
        "cann_phase_rmse_deg", "cann_max_phase_error_deg",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["label"] for row in rows]
    x = np.arange(len(rows))
    width = 0.25
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    position_fields = (
        ("position_rmse_m", "Full duration"),
        ("position_rmse_outage_m", "600-1200 s window"),
        ("position_rmse_recovery_m", "After 1200 s"),
    )
    for offset, (field, legend) in zip((-width, 0.0, width), position_fields):
        bars = axes[0].bar(
            x + offset, [row[field] for row in rows], width, label=legend,
        )
        axes[0].bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    axes[0].set_ylabel("Position RMSE (m)")
    axes[0].set_title("Single-satellite three-modal filter: outage sensitivity")
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)

    velocity_fields = (
        ("velocity_rmse_mps", "Full duration"),
        ("velocity_rmse_outage_mps", "600-1200 s window"),
        ("velocity_rmse_recovery_mps", "After 1200 s"),
    )
    for offset, (field, legend) in zip((-width, 0.0, width), velocity_fields):
        bars = axes[1].bar(
            x + offset, [row[field] for row in rows], width, label=legend,
        )
        axes[1].bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    axes[1].set_ylabel("Velocity RMSE (m/s)")
    axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)

    phase = [row["cann_phase_rmse_deg"] for row in rows]
    bars = axes[2].bar(x, phase, width=0.55, color="tab:orange")
    axes[2].bar_label(bars, fmt="%.5f", padding=2, fontsize=9)
    axes[2].set_ylabel("CANN phase RMSE (deg)")
    axes[2].set_ylim(0.0, max(phase) * 1.18)
    axes[2].grid(axis="y", alpha=0.3)

    for axis in axes:
        axis.set_xticks(x, labels)
    fig.tight_layout()
    figure_path = output_dir / "overview.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return figure_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results/cann"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/cann/single_satellite_outage_comparison"),
    )
    args = parser.parse_args()
    figure, csv_path = generate_comparison(args.results_root, args.output_dir)
    print({"figure": str(figure), "csv": str(csv_path)})


if __name__ == "__main__":
    main()
