from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def load_json(json_path: str | Path) -> Any:
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_numpy_safe(x: Any, dtype=float) -> np.ndarray:
    return np.asarray(x, dtype=dtype)


def build_time_axis(p_sim: Dict[str, Any], n: int) -> np.ndarray:
    cam_step = float(p_sim.get("cam_step", 1.0))
    return np.arange(n, dtype=float) * cam_step
