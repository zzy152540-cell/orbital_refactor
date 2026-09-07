from __future__ import annotations

import numpy as np

from cooperative.shadow_quality_feedback import (
    ShadowQualityFeedbackConfig,
    apply_shadow_quality_feedback,
)
from experiments.walker_modality_shadow_quality_audit import (
    run_walker_modality_shadow_quality_audit,
)
from experiments.walker_raw_sensor_comparison import _run


def run_walker_shadow_feedback_comparison(
    *, duration=20.0, dt=2.0, seed=0,
    feedback_config=None,
):
    """Two-pass counterfactual; quality from pass one affects only pass two."""
    audit = run_walker_modality_shadow_quality_audit(
        duration=duration, dt=dt, seed=seed,
    )
    source = audit["observations"]
    disabled = apply_shadow_quality_feedback(
        source, audit["records"], config=ShadowQualityFeedbackConfig(),
    )
    if not all(left is right for left, right in zip(source, disabled)):
        raise RuntimeError("Disabled shadow feedback changed an observation.")
    baseline = audit["baseline_history"]
    selected = feedback_config or ShadowQualityFeedbackConfig(
        enabled=True, maximum_covariance_inflation=4.0,
    )
    adjusted = apply_shadow_quality_feedback(
        source, audit["records"], config=selected,
    )
    feedback = _run(audit["case"], adjusted)
    return {
        "baseline": _filter_metrics(baseline, audit["case"]["truth"]),
        "feedback": _filter_metrics(feedback, audit["case"]["truth"]),
        "inflation": {
            "mean": float(np.mean([
                item.metadata["shadow_covariance_inflation"]
                for item in adjusted
            ])),
            "maximum": float(np.max([
                item.metadata["shadow_covariance_inflation"]
                for item in adjusted
            ])),
            "rejected_count": int(sum(
                item.metadata["shadow_feedback_rejected"] for item in adjusted
            )),
        },
        "disabled_identity_preserved": True,
    }


def _filter_metrics(history, truth):
    position_error = np.concatenate([
        np.linalg.norm(
            history.active_state_history_by_node[node][:, :3]
            - truth[node][:, :3], axis=1,
        )
        for node in history.node_ids
    ])
    nis = np.asarray([
        value
        for node in history.node_ids
        for epoch in history.nis_history_by_node[node]
        for value in epoch.values()
        if np.isfinite(value)
    ])
    return {
        "position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
        "maximum_position_error_m": float(np.max(position_error)),
        "mean_nis": float(np.mean(nis)) if nis.size else float("nan"),
        "nis_count": int(nis.size),
    }


if __name__ == "__main__":
    print(run_walker_shadow_feedback_comparison())
