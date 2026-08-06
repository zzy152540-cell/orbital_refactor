from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cooperative.multi_sat_pipeline import CooperativePipelineResult
from scenarios.multi_satellite_scenario import CooperativeScenario


def plot_cooperative_results(
    *,
    scenario: CooperativeScenario,
    result: CooperativePipelineResult,
    save_dir: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot local and cooperative filtering results.

    Plots:
        1. Position-error norm
        2. Velocity-error norm
        3. CI node weights
        4. Cooperative XYZ position-error components
        5. Cooperative XYZ velocity-error components
    """
    timestamps = np.asarray(scenario.timestamps, dtype=float)
    truth = np.asarray(
        scenario.target_trajectory.state_history_eci,
        dtype=float,
    )
    cooperative = np.asarray(
        result.cooperative_history.state_history_eci,
        dtype=float,
    )

    if truth.shape != cooperative.shape:
        raise ValueError(
            "Truth and cooperative histories must have the same shape."
        )

    output_dir = None
    if save_dir is not None:
        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    _plot_position_error_norm(
        timestamps=timestamps,
        truth=truth,
        local_histories=result.local_absolute_state_history_by_node,
        cooperative=cooperative,
        output_dir=output_dir,
    )

    _plot_velocity_error_norm(
        timestamps=timestamps,
        truth=truth,
        local_histories=result.local_absolute_state_history_by_node,
        cooperative=cooperative,
        output_dir=output_dir,
    )

    _plot_ci_weights(
        timestamps=timestamps,
        weight_history=result.cooperative_history.node_weight_history,
        output_dir=output_dir,
    )

    _plot_cooperative_position_components(
        timestamps=timestamps,
        truth=truth,
        cooperative=cooperative,
        output_dir=output_dir,
    )

    _plot_cooperative_velocity_components(
        timestamps=timestamps,
        truth=truth,
        cooperative=cooperative,
        output_dir=output_dir,
    )

    if show:
        plt.show()
    else:
        plt.close("all")


def _plot_position_error_norm(
    *,
    timestamps: np.ndarray,
    truth: np.ndarray,
    local_histories: dict[str, np.ndarray],
    cooperative: np.ndarray,
    output_dir: Path | None,
) -> None:
    plt.figure(figsize=(10, 6))

    for node_id, state_history in local_histories.items():
        state_history = np.asarray(state_history, dtype=float)
        error_norm = np.linalg.norm(
            state_history[:, :3] - truth[:, :3],
            axis=1,
        )
        plt.plot(
            timestamps,
            error_norm,
            label=f"{node_id} local",
        )

    cooperative_error = np.linalg.norm(
        cooperative[:, :3] - truth[:, :3],
        axis=1,
    )
    plt.plot(
        timestamps,
        cooperative_error,
        linewidth=2.5,
        label="Cooperative CI",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Position error norm (m)")
    plt.title("Local and Cooperative Position Errors")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    _save_figure(output_dir, "position_error_norm.png")


def _plot_velocity_error_norm(
    *,
    timestamps: np.ndarray,
    truth: np.ndarray,
    local_histories: dict[str, np.ndarray],
    cooperative: np.ndarray,
    output_dir: Path | None,
) -> None:
    plt.figure(figsize=(10, 6))

    for node_id, state_history in local_histories.items():
        state_history = np.asarray(state_history, dtype=float)
        error_norm = np.linalg.norm(
            state_history[:, 3:] - truth[:, 3:],
            axis=1,
        )
        plt.plot(
            timestamps,
            error_norm,
            label=f"{node_id} local",
        )

    cooperative_error = np.linalg.norm(
        cooperative[:, 3:] - truth[:, 3:],
        axis=1,
    )
    plt.plot(
        timestamps,
        cooperative_error,
        linewidth=2.5,
        label="Cooperative CI",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Velocity error norm (m/s)")
    plt.title("Local and Cooperative Velocity Errors")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    _save_figure(output_dir, "velocity_error_norm.png")


def _plot_ci_weights(
    *,
    timestamps: np.ndarray,
    weight_history: list[dict[str, float]],
    output_dir: Path | None,
) -> None:
    node_ids = sorted(
        {
            node_id
            for weights in weight_history
            for node_id in weights
        }
    )

    plt.figure(figsize=(10, 6))

    for node_id in node_ids:
        values = np.array(
            [
                weights.get(node_id, np.nan)
                for weights in weight_history
            ],
            dtype=float,
        )
        plt.plot(
            timestamps,
            values,
            label=node_id,
        )

    plt.xlabel("Time (s)")
    plt.ylabel("CI weight")
    plt.title("Cooperative CI Node Weights")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    _save_figure(output_dir, "ci_node_weights.png")


def _plot_cooperative_position_components(
    *,
    timestamps: np.ndarray,
    truth: np.ndarray,
    cooperative: np.ndarray,
    output_dir: Path | None,
) -> None:
    error = cooperative[:, :3] - truth[:, :3]
    labels = ("X error", "Y error", "Z error")

    plt.figure(figsize=(10, 6))

    for index, label in enumerate(labels):
        plt.plot(
            timestamps,
            error[:, index],
            label=label,
        )

    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Position error (m)")
    plt.title("Cooperative Position-Error Components")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    _save_figure(
        output_dir,
        "cooperative_position_components.png",
    )


def _plot_cooperative_velocity_components(
    *,
    timestamps: np.ndarray,
    truth: np.ndarray,
    cooperative: np.ndarray,
    output_dir: Path | None,
) -> None:
    error = cooperative[:, 3:] - truth[:, 3:]
    labels = ("Vx error", "Vy error", "Vz error")

    plt.figure(figsize=(10, 6))

    for index, label in enumerate(labels):
        plt.plot(
            timestamps,
            error[:, index],
            label=label,
        )

    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity error (m/s)")
    plt.title("Cooperative Velocity-Error Components")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    _save_figure(
        output_dir,
        "cooperative_velocity_components.png",
    )


def _save_figure(
    output_dir: Path | None,
    filename: str,
) -> None:
    if output_dir is not None:
        plt.savefig(
            output_dir / filename,
            dpi=200,
            bbox_inches="tight",
        )