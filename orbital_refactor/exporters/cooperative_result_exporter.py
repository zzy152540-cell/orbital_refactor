from __future__ import annotations
import json
from pathlib import Path

def save_cooperative_summary(result, output_dir="results/cooperative"):
    out=Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data={
        "position_rmse": float(result.metrics.cooperative_position_rmse),
        "velocity_rmse": float(result.metrics.cooperative_velocity_rmse),
        "local_position_rmse": result.metrics.local_position_rmse,
        "local_velocity_rmse": result.metrics.local_velocity_rmse,
    }
    (out/"rmse.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    hist=result.cooperative_history
    (out/"node_history.json").write_text(
        json.dumps({
            "active": hist.active_node_history,
            "received": hist.received_node_history
        }, indent=2), encoding="utf-8")
