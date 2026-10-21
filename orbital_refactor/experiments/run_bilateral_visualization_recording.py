"""Generate and optionally open a bilateral 15-observer/15-target replay."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.bilateral_multitarget_frontend import (
    BilateralFrontendConfig,
    _run_bilateral_online_arm,
    build_bilateral_raw_frontend,
)
from interfaces.data_objects import AbsolutePositionObservation, RawSensorFrame
from visualization.cann_history_adapter import build_cann_visualization_history
from visualization.data_contract import (
    VisualEdge,
    VisualNavigationState,
    VisualNodeState,
    VisualObservation,
    VisualizationFrame,
)
from visualization.recording import VisualizationRecordingWriter


def generate_bilateral_visualization_recording(
    output: str | Path,
    *,
    duration: float = 20.0,
    dt: float = 1.0,
    seed: int = 0,
    include_cann: bool = True,
    cann_node_limit: int = 3,
    retain_raw_frames: bool = True,
    progress_callback=None,
) -> Path:
    """Run the accepted bilateral frontend and export immutable GUI frames."""

    def progress(stage, fraction):
        if progress_callback is not None:
            progress_callback(stage, float(fraction), 0.0)

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Visualization recording already exists: {output}")
    config = BilateralFrontendConfig(
        duration_seconds=float(duration),
        sample_interval_seconds=float(dt),
        random_seed=int(seed),
        retain_raw_frames=bool(retain_raw_frames),
        use_local_unified_multimodal_measurement_update=True,
    )
    progress("bilateral_frontend_started", 0.02)
    frontend = build_bilateral_raw_frontend(config)
    progress("bilateral_frontend_complete", 0.35)
    captured = []
    _run_bilateral_online_arm(
        frontend,
        enable_cann=True,
        enable_local_unified=True,
        epoch_callback=captured.append,
    )
    progress("bilateral_tracking_complete", 0.65)
    if len(captured) != len(frontend.truth.timestamps):
        raise RuntimeError("Bilateral visualization capture is incomplete.")

    friendly_ids = frontend.truth.manifest.friendly_observer_ids
    hostile_ids = frontend.truth.manifest.hostile_target_ids
    cann_history = _hostile_target_cann_history(
        frontend, captured,
        node_limit=cann_node_limit,
    ) if include_cann else None
    progress("bilateral_cann_complete" if include_cann else "cann_disabled", 0.82)
    configured_edges = _friendly_topology_edges(frontend)
    frames = []
    for payload in captured:
        index = int(payload["epoch_index"])
        timestamp = float(payload["timestamp"])
        nodes = _visual_nodes(
            frontend,
            payload,
            friendly_ids=friendly_ids,
            hostile_ids=hostile_ids,
        )
        observations = _visual_observations(frontend, payload["epoch_messages"])
        observation_edges = tuple(
            VisualEdge(
                observer_id,
                target_id,
                "OBSERVATION_LINK",
                modality="MULTIMODAL",
                directed=True,
            )
            for observer_id, target_id in sorted({
                (item.observer_id, item.target_id) for item in observations
            })
        )
        tracked = tuple(
            node for node in nodes
            if node.status == "HOSTILE_TARGET_TRACKED"
        )
        errors = np.asarray([
            np.linalg.norm(node.estimate_state[:3] - node.truth_state[:3])
            for node in tracked
        ], dtype=float)
        frames.append(VisualizationFrame(
            scenario_id="bilateral-15-friendly-15-hostile",
            run_id=f"seed-{seed}-joint-multimodal-cann",
            timestamp=timestamp,
            epoch_index=index,
            nodes=nodes,
            edges=(*configured_edges, *observation_edges),
            observations=observations,
            navigation=(
                _default_navigation(friendly_ids, hostile_ids)
                if cann_history is None
                else cann_history.navigation_by_epoch[index]
            ),
            cann=(
                () if cann_history is None
                else cann_history.cann_by_epoch[index]
            ),
            metadata={
                "fleet_position_rmse_m": (
                    float(np.sqrt(np.mean(np.square(errors))))
                    if errors.size else 0.0
                ),
                "friendly_observer_count": len(friendly_ids),
                "hostile_target_count": len(hostile_ids),
                "tracked_hostile_target_count": len(tracked),
                "node_role_by_id": {
                    **{node_id: "FRIENDLY_OBSERVER" for node_id in friendly_ids},
                    **{node_id: "HOSTILE_TARGET" for node_id in hostile_ids},
                },
                "truth_usage": "DISPLAY_AND_SCORING_ONLY",
                "topology_version": "friendly-ring-v1",
                "transmitted_message_count": len(payload["epoch_messages"]),
                "delivered_message_count": len(payload["epoch_messages"]),
                "dropped_message_count": 0,
                "delayed_message_count": 0,
                "resynchronization_count": 0,
            },
        ))
    with VisualizationRecordingWriter(output) as writer:
        for frame in frames:
            writer.append(frame)
    progress("recording_complete", 1.0)
    return output


def _visual_nodes(frontend, payload, *, friendly_ids, hostile_ids):
    index = int(payload["epoch_index"])
    tracks = {
        track.estimate.track_id: track
        for track in payload["tracker_update"].tracks_by_target.values()
        if track.lifecycle.value != "TERMINATED"
    }
    samples = payload["identity_epoch"].samples_by_truth
    nodes = []
    for node_id in friendly_ids:
        truth = frontend.truth.friendly_truth_state_history_by_node[node_id][index]
        covariance = np.eye(6) * 1e-6
        nodes.append(VisualNodeState(
            node_id=node_id,
            estimate_state=truth,
            truth_state=truth,
            covariance_diagonal=np.diag(covariance),
            covariance=covariance,
            status="FRIENDLY_OBSERVER",
        ))
    for target_id in hostile_ids:
        truth = frontend.truth.hostile_truth_state_history_by_target[target_id][index]
        sample = samples.get(target_id)
        track = None if sample is None else tracks.get(sample.track_id)
        if track is None:
            estimate = truth
            covariance = np.eye(6) * 1e12
            valid = False
            status = "HOSTILE_TARGET_UNINITIALIZED"
        else:
            estimate = track.estimate.state_eci
            covariance = track.estimate.covariance_eci
            valid = True
            status = "HOSTILE_TARGET_TRACKED"
        nodes.append(VisualNodeState(
            node_id=target_id,
            estimate_state=estimate,
            truth_state=truth,
            covariance_diagonal=np.diag(covariance),
            covariance=covariance,
            valid=valid,
            status=status,
        ))
    return tuple(nodes)


def _visual_observations(frontend, messages):
    values = []
    for message in messages:
        target_id = frontend.truth_target_by_detection_group[
            message.detection_group_id
        ]
        raw = frontend.raw_frames_by_detection_group.get(
            message.detection_group_id, {}
        ).get(message.modality)
        raw_data, raw_kind = _raw_visualization_payload(raw, message.modality)
        values.append(VisualObservation(
            observer_id=message.observer_id,
            target_id=target_id,
            modality=message.modality,
            measurement=message.measurement,
            covariance_diagonal=np.diag(message.covariance),
            visible=True,
            valid=message.valid_flag,
            processing_status="AVAILABLE_TO_TRACKER",
            frame=(
                "RANGE_RANGE_RATE" if message.modality == "RADAR"
                else "BODY_AZ_EL"
            ),
            raw_sensor_data=raw_data,
            raw_data_kind=raw_kind,
            metadata={
                "message_id": message.message_id,
                "detection_group_id": message.detection_group_id,
                "truth_identity_usage": "DISPLAY_ONLY",
            },
        ))
    return tuple(values)


def _raw_visualization_payload(raw, modality):
    if raw is None:
        return None, None
    if isinstance(raw, RawSensorFrame):
        return raw.visualization_payload()
    if modality == "RADAR" and hasattr(raw, "power"):
        return np.asarray(raw.power), "RANGE_DOPPLER_POWER"
    if modality in {"INFRARED", "OPTICAL"} and hasattr(raw, "image"):
        return np.asarray(raw.image), f"{modality}_GRAYSCALE_IMAGE"
    raise TypeError(
        f"Unsupported raw {modality} visualization frame: {type(raw).__name__}"
    )


def _friendly_topology_edges(frontend):
    topology = frontend.truth.manifest.friendly_topology
    pairs = {
        tuple(sorted((node_id, neighbor)))
        for node_id in topology.node_ids
        for neighbor in topology.neighbors(node_id)
    }
    return tuple(
        VisualEdge(left, right, "ACTIVE_TOPOLOGY")
        for left, right in sorted(pairs)
    )


def _hostile_target_cann_history(frontend, captured, *, node_limit):
    friendly_ids = frontend.truth.manifest.friendly_observer_ids
    hostile_ids = frontend.truth.manifest.hostile_target_ids
    selected_count = (
        len(hostile_ids) if int(node_limit) <= 0
        else min(int(node_limit), len(hostile_ids))
    )
    selected = tuple(hostile_ids[:selected_count])
    times = np.asarray(frontend.truth.timestamps, dtype=float)
    posterior_by_epoch = tuple(
        {
            target_id: _tracked_target_posterior(payload, target_id)
            for target_id in selected
        }
        for payload in captured
    )
    navigation = [list(_target_cann_navigation_defaults(
        friendly_ids, hostile_ids, selected, posterior,
    )) for posterior in posterior_by_epoch]
    cann = [[] for _ in times]
    available_indices = tuple(
        index for index, posterior in enumerate(posterior_by_epoch)
        if all(posterior[target_id] is not None for target_id in selected)
    )
    for start, stop in _contiguous_runs(available_indices):
        segment = slice(start, stop)
        segment_times = times[segment]
        states = {
            target_id: np.stack([
                posterior_by_epoch[index][target_id].state_eci
                for index in range(start, stop)
            ])
            for target_id in selected
        }
        covariance = {
            target_id: np.stack([
                posterior_by_epoch[index][target_id].covariance_eci
                for index in range(start, stop)
            ])
            for target_id in selected
        }
        history = SimpleNamespace(
            timestamps=segment_times,
            node_ids=selected,
            active_state_history_by_node=states,
            active_covariance_history_by_node=covariance,
        )
        posterior_cues = tuple(
            AbsolutePositionObservation(
                timestamp=float(timestamp),
                satellite_id=target_id,
                measurement_eci=states[target_id][local_index, :3],
                covariance=covariance[target_id][local_index, :3, :3],
                confidence=1.0,
                valid_flag=True,
                source_type="TARGET_FILTER_POSTERIOR",
            )
            for local_index, timestamp in enumerate(segment_times)
            for target_id in selected
        )
        base = build_cann_visualization_history(
            history=history,
            initial_state_by_node={
                target_id: states[target_id][0] for target_id in selected
            },
            absolute_position_observations=posterior_cues,
            cann_node_ids=selected,
        )
        for local_index, global_index in enumerate(range(start, stop)):
            by_node = {
                item.node_id: replace(
                    item,
                    absolute_navigation_available=False,
                    last_absolute_navigation_timestamp=None,
                    status="TARGET_FILTER_POSTERIOR_CANN",
                )
                for item in base.navigation_by_epoch[local_index]
            }
            navigation[global_index] = [
                by_node.get(item.node_id, item)
                for item in navigation[global_index]
            ]
            cann[global_index].extend(
                replace(item, metrics={
                    **item.metrics,
                    "input_state_source": "TARGET_FILTER_POSTERIOR",
                })
                for item in base.cann_by_epoch[local_index]
            )
    return SimpleNamespace(
        navigation_by_epoch=tuple(tuple(epoch) for epoch in navigation),
        cann_by_epoch=tuple(tuple(epoch) for epoch in cann),
    )


def _tracked_target_posterior(payload, target_id):
    sample = payload["identity_epoch"].samples_by_truth.get(target_id)
    if sample is None:
        return None
    track = next((
        item for item in payload["tracker_update"].tracks_by_target.values()
        if item.estimate.track_id == sample.track_id
        and item.lifecycle.value != "TERMINATED"
    ), None)
    return None if track is None else track.estimate


def _contiguous_runs(indices):
    if not indices:
        return ()
    runs = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            runs.append((start, previous + 1))
            start = index
        previous = index
    runs.append((start, previous + 1))
    return tuple(runs)


def _target_cann_navigation_defaults(
    friendly_ids, hostile_ids, selected, posterior,
):
    return tuple(
        VisualNavigationState(
            node_id=node_id,
            absolute_navigation_available=(node_id in friendly_ids),
            status=(
                "KNOWN_FRIENDLY_STATE"
                if node_id in friendly_ids else
                "TARGET_FILTER_UNINITIALIZED"
                if node_id in selected and posterior[node_id] is None else
                "TARGET_CANN_PENDING"
                if node_id in selected else
                "TARGET_CANN_NOT_RECORDED"
            ),
        )
        for node_id in (*friendly_ids, *hostile_ids)
    )


def _default_navigation(friendly_ids, hostile_ids):
    return tuple(
        VisualNavigationState(
            node_id=node_id,
            absolute_navigation_available=(node_id in friendly_ids),
            status=(
                "KNOWN_FRIENDLY_STATE" if node_id in friendly_ids
                else "HOSTILE_TARGET_NOT_SELF_NAVIGATING"
            ),
        )
        for node_id in (*friendly_ids, *hostile_ids)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/visualization_recordings/bilateral30_20s_seed0"),
    )
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-cann", action="store_true")
    parser.add_argument(
        "--cann-node-limit", type=int, default=3,
        help="maximum number of hostile target posterior CANN histories",
    )
    parser.add_argument("--no-raw-frames", action="store_true")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    output = generate_bilateral_visualization_recording(
        args.output,
        duration=args.duration,
        dt=args.dt,
        seed=args.seed,
        include_cann=not args.no_cann,
        cann_node_limit=args.cann_node_limit,
        retain_raw_frames=not args.no_raw_frames,
    )
    print(output)
    if args.open:
        from experiments.run_v15_visualization_window import configure_qt_runtime
        configure_qt_runtime()
        from visualization.replay_window import run_replay_window
        raise SystemExit(run_replay_window(output))


if __name__ == "__main__":
    main()
