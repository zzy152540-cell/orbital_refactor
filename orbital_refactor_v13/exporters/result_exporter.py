from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if is_dataclass(obj):
            return asdict(obj)
        return super().default(obj)


def export_history_npz(history: Any, output_path: str | Path) -> Path:
    """Export the common state/covariance/diagnostic histories to compressed NPZ."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = _history_attr(history, "state_history", "fused_state_history")
    covariance = _history_attr(history, "covariance_history", "fused_covariance_history")
    acceleration = _history_attr(history, "acceleration_history", "fused_acceleration_history")
    payload: dict[str, Any] = {
        "timestamps": np.asarray(history.timestamps),
        "state_estimate": np.asarray(state),
        "covariance": np.asarray(covariance),
        "acceleration_estimate": np.asarray(acceleration),
    }
    for name, values in getattr(history, "nis_history", {}).items():
        payload[f"nis_{name}"] = np.asarray(values)
    for name, values in getattr(history, "gate_history", {}).items():
        payload[f"gate_{name}"] = np.asarray(values)
    if hasattr(history, "ci_weight_history"):
        modalities = sorted({key for row in history.ci_weight_history if row for key in row})
        for modality in modalities:
            payload[f"weight_{modality}"] = np.array([
                np.nan if row is None else row.get(modality, 0.0)
                for row in history.ci_weight_history
            ])
    np.savez_compressed(path, **payload)
    return path


def export_history_csv(history: Any, output_path: str | Path) -> Path:
    """Export one row per epoch with position, velocity, acceleration and diagnostics."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = np.asarray(_history_attr(history, "state_history", "fused_state_history"))
    acceleration = np.asarray(_history_attr(history, "acceleration_history", "fused_acceleration_history"))
    nis_history = getattr(history, "nis_history", {})
    gate_history = getattr(history, "gate_history", {})
    modalities = sorted(nis_history)
    weight_modalities = sorted({
        key for row in getattr(history, "ci_weight_history", []) if row for key in row
    })
    header = ["timestamp", "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"]
    header += [f"nis_{name}" for name in modalities]
    header += [f"gate_{name}" for name in modalities]
    header += [f"weight_{name}" for name in weight_modalities]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for index, timestamp in enumerate(history.timestamps):
            row = [float(timestamp), *state[index].tolist(), *acceleration[index].tolist()]
            row += [float(nis_history[name][index]) for name in modalities]
            row += [bool(gate_history[name][index]) for name in modalities]
            weights = getattr(history, "ci_weight_history", [])
            current = weights[index] if index < len(weights) else None
            row += [np.nan if current is None else float(current.get(name, 0.0)) for name in weight_modalities]
            writer.writerow(row)
    return path


def export_module_output_json(module_output: Any, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(module_output, file, cls=NumpyJSONEncoder, ensure_ascii=False, indent=2)
    return path


def export_run_bundle(history: Any, module_output: Any, output_directory: str | Path, stem: str = "state_awareness") -> dict[str, Path]:
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "csv": export_history_csv(history, directory / f"{stem}.csv"),
        "npz": export_history_npz(history, directory / f"{stem}.npz"),
        "json": export_module_output_json(module_output, directory / f"{stem}.json"),
    }


def _history_attr(history: Any, *names: str) -> Any:
    for name in names:
        if hasattr(history, name):
            return getattr(history, name)
    raise AttributeError(f"History object does not expose any of: {names}")
