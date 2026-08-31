from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SCENARIOS = (
    ("none", "No outage"), ("ir", "IR outage"),
    ("opt", "Optical outage"), ("rad", "Radar outage"),
    ("all", "All-modal outage"),
)


def generate_comparison(results_root: Path, output_dir: Path) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = []
    for key, label in SCENARIOS:
        cann = json.loads((
            results_root / f"single_satellite_outage_{key}" / "summary.json"
        ).read_text(encoding="utf-8"))
        original = json.loads((
            results_root / f"single_satellite_original_{key}" / "summary.json"
        ).read_text(encoding="utf-8"))
        rows.append({
            "scenario": key, "label": label,
            "original_position_rmse_m": original["position_rmse_m"],
            "cann_position_rmse_m": cann["position_rmse_m"],
            "original_outage_rmse_m": original["position_rmse_outage_m"],
            "cann_outage_rmse_m": cann["position_rmse_outage_m"],
            "original_velocity_rmse_mps": original["velocity_rmse_mps"],
            "cann_velocity_rmse_mps": cann["velocity_rmse_mps"],
            "position_delta_m": cann["position_rmse_m"] - original["position_rmse_m"],
            "outage_delta_m": cann["position_rmse_outage_m"] - original["position_rmse_outage_m"],
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    x = np.arange(len(rows)); width = 0.36
    labels = [row["label"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for axis, original_field, cann_field, ylabel, title in (
        (axes[0], "original_position_rmse_m", "cann_position_rmse_m",
         "Full-duration position RMSE (m)", "Main filter output"),
        (axes[1], "original_outage_rmse_m", "cann_outage_rmse_m",
         "600-1200 s position RMSE (m)", "Outage-window output"),
    ):
        original_bars = axis.bar(
            x - width / 2, [row[original_field] for row in rows], width,
            label="Original Federated-CI", color="tab:blue",
        )
        cann_bars = axis.bar(
            x + width / 2, [row[cann_field] for row in rows], width,
            label="Federated-CI + passive CANN", color="tab:orange",
        )
        axis.bar_label(original_bars, fmt="%.2f", padding=2, fontsize=8)
        axis.bar_label(cann_bars, fmt="%.2f", padding=2, fontsize=8)
        axis.set_ylabel(ylabel); axis.set_title(title)
        axis.grid(axis="y", alpha=.3); axis.legend()
    axes[1].set_xticks(x, labels)
    fig.suptitle("Passive CANN does not alter single-satellite filter accuracy")
    fig.tight_layout()
    figure_path = output_dir / "overview.png"
    fig.savefig(figure_path, dpi=180); plt.close(fig)
    return figure_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results/cann"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/cann/cann_vs_original_filter"),
    )
    args = parser.parse_args()
    print(dict(zip(
        ("figure", "csv"),
        map(str, generate_comparison(args.results_root, args.output_dir)),
    )))


if __name__ == "__main__":
    main()
