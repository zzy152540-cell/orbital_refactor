from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from cooperative.cooperative_update import CooperativeUpdateResult, update_local_state
from cooperative.multi_neighbor_schmidt import (
    initialize_multi_neighbor_schmidt,
    multi_neighbor_schmidt_predict,
    multi_neighbor_schmidt_update,
)
from interfaces.data_objects import ObservationMessage, StateMessage, TargetEstimate
from orbital_core.dynamics import (
    make_process_noise,
    numerical_jacobian_discrete,
    rk4_step_absolute,
)


@dataclass(frozen=True)
class ObservationRoutingDecision:
    receiver_id: str
    role: str
    update_target_id: str | None
    nuisance_target_id: str | None


@dataclass(frozen=True)
class ThirdPartyObservationUpdateResult:
    routing: ObservationRoutingDecision
    target_estimate_by_id: dict[str, TargetEstimate]
    update: CooperativeUpdateResult


@dataclass(frozen=True)
class ThirdPartyTrackHistory:
    receiver_id: str
    timestamps: np.ndarray
    state_history_by_target: dict[str, np.ndarray]
    covariance_history_by_target: dict[str, np.ndarray]
    nis_history: list[dict[str, float]]
    used_observation_ids: tuple[str, ...]


def classify_observation_receiver(
    receiver_id: str,
    observation: ObservationMessage,
) -> ObservationRoutingDecision:
    """Classify how a receiver may consume one directed physical observation."""

    receiver = str(receiver_id)
    observer = str(observation.observer_id)
    target = str(observation.target_id)
    if observer == target:
        raise ValueError("Observation endpoints must describe different satellites.")
    if receiver == observer:
        return ObservationRoutingDecision(
            receiver, "observer_self_state", observer, target
        )
    if receiver == target:
        return ObservationRoutingDecision(
            receiver, "target_self_state", target, observer
        )
    return ObservationRoutingDecision(
        receiver, "third_party_target_track", target, observer
    )


def apply_third_party_observation(
    *,
    receiver_id: str,
    target_estimate_by_id: Mapping[str, TargetEstimate],
    observation: ObservationMessage,
) -> ThirdPartyObservationUpdateResult:
    """Update only the receiver's track of the observation's directed target.

    The observer track is treated as an uncertain nuisance state. This is an
    intentionally local approximation: it does not claim that the two remote
    target tracks remain independent after the update.
    """

    routing = classify_observation_receiver(receiver_id, observation)
    if routing.role != "third_party_target_track":
        raise ValueError("This API accepts only a third-party observation receiver.")
    receiver = str(receiver_id)
    tracks = {str(key): value for key, value in target_estimate_by_id.items()}
    update_target = str(routing.update_target_id)
    nuisance_target = str(routing.nuisance_target_id)
    missing = {update_target, nuisance_target} - set(tracks)
    if missing:
        raise ValueError(f"Receiver is missing required target tracks: {sorted(missing)}")
    local_track = tracks[update_target]
    nuisance_track = tracks[nuisance_target]
    for target_id, track in (
        (update_target, local_track), (nuisance_target, nuisance_track),
    ):
        if str(track.estimator_node_id) != receiver:
            raise ValueError("Every target track must belong to the receiver estimator.")
        if str(track.target_node_id) != target_id:
            raise ValueError("Target-track mapping key does not match its target ID.")
        if not np.isclose(float(track.timestamp), float(observation.timestamp)):
            raise ValueError("Third-party target tracks must align to observation time.")
    nuisance_message = StateMessage(
        source_node_id=receiver,
        target_node_id=nuisance_target,
        timestamp=float(nuisance_track.timestamp),
        state_estimate=np.asarray(nuisance_track.state_estimate, dtype=float).copy(),
        covariance=np.asarray(nuisance_track.covariance, dtype=float).copy(),
        quality_score=float(nuisance_track.quality_score),
        valid_flag=bool(nuisance_track.valid_flag),
        information_ids=tuple(nuisance_track.information_ids),
    )
    quaternion = (
        observation.metadata.get("quaternion_i2b_wxyz")
        if str(observation.frame).upper() == "BODY" else None
    )
    update = update_local_state(
        local_estimate=local_track,
        neighbor_state=nuisance_message,
        observation=observation,
        quaternion_i2b_wxyz=quaternion,
    )
    updated_tracks = dict(tracks)
    updated_tracks[update_target] = update.estimate
    return ThirdPartyObservationUpdateResult(routing, updated_tracks, update)


def run_third_party_target_track_filter(
    *,
    receiver_id: str,
    timestamps: np.ndarray,
    initial_target_estimate_by_id: Mapping[str, TargetEstimate],
    observation_messages: tuple[ObservationMessage, ...] | list[ObservationMessage],
    process_noise_acceleration: float = 1e-8,
) -> ThirdPartyTrackHistory:
    """Propagate a receiver's target bank and apply source-time observations."""

    times = np.asarray(timestamps, dtype=float).reshape(-1)
    if times.size == 0 or (times.size > 1 and not np.all(np.diff(times) > 0.0)):
        raise ValueError("timestamps must be nonempty and strictly increasing.")
    receiver = str(receiver_id)
    tracks = {
        str(target_id): estimate
        for target_id, estimate in initial_target_estimate_by_id.items()
    }
    if not tracks:
        raise ValueError("At least one target track is required.")
    for target_id, estimate in tracks.items():
        if str(estimate.estimator_node_id) != receiver:
            raise ValueError("Every initial target track must belong to receiver_id.")
        if str(estimate.target_node_id) != target_id:
            raise ValueError("Initial target-track key does not match target ID.")
        if not np.isclose(float(estimate.timestamp), float(times[0])):
            raise ValueError("Initial target tracks must start at the first timestamp.")
    grouped = {float(timestamp): [] for timestamp in times}
    seen_message_ids = set()
    for observation in observation_messages:
        timestamp = float(observation.timestamp)
        if timestamp not in grouped:
            raise ValueError("Observation timestamp is not in timestamps.")
        if observation.message_id in seen_message_ids:
            raise ValueError("Observation message IDs must be unique.")
        seen_message_ids.add(observation.message_id)
        if str(observation.observer_id) == receiver or str(observation.target_id) == receiver:
            raise ValueError("Target-track runner accepts only third-party observations.")
        grouped[timestamp].append(observation)
    state_history = {
        target: np.zeros((times.size, 6), dtype=float) for target in tracks
    }
    covariance_history = {
        target: np.zeros((times.size, 6, 6), dtype=float) for target in tracks
    }
    nis_history = []
    for index, timestamp in enumerate(times):
        if index > 0:
            dt = float(timestamp - times[index - 1])
            propagated = {}
            for target_id, estimate in tracks.items():
                transition = numerical_jacobian_discrete(
                    lambda value: rk4_step_absolute(value, dt),
                    estimate.state_estimate,
                )
                covariance = (
                    transition @ estimate.covariance @ transition.T
                    + make_process_noise(dt, process_noise_acceleration)
                )
                propagated[target_id] = TargetEstimate(
                    estimator_node_id=receiver, target_node_id=target_id,
                    timestamp=float(timestamp),
                    state_estimate=rk4_step_absolute(estimate.state_estimate, dt),
                    covariance=0.5 * (covariance + covariance.T),
                    quality_score=float(estimate.quality_score),
                    valid_flag=bool(estimate.valid_flag),
                    information_ids=tuple(estimate.information_ids),
                )
            tracks = propagated
        epoch_nis = {}
        for observation in sorted(
            grouped[float(timestamp)], key=lambda item: item.information_id
        ):
            result = apply_third_party_observation(
                receiver_id=receiver,
                target_estimate_by_id=tracks,
                observation=observation,
            )
            tracks = result.target_estimate_by_id
            epoch_nis[observation.information_id] = result.update.nis
        nis_history.append(epoch_nis)
        for target_id, estimate in tracks.items():
            state_history[target_id][index] = estimate.state_estimate
            covariance_history[target_id][index] = estimate.covariance
    used_ids = tuple(dict.fromkeys(
        information_id
        for estimate in tracks.values()
        for information_id in estimate.information_ids
    ))
    return ThirdPartyTrackHistory(
        receiver, times.copy(), state_history, covariance_history,
        nis_history, used_ids,
    )


def run_third_party_schmidt_pair_filter(
    *, receiver_id: str, active_target_id: str, consider_target_id: str,
    timestamps: np.ndarray,
    initial_target_estimate_by_id: Mapping[str, TargetEstimate],
    observation_messages: tuple[ObservationMessage, ...] | list[ObservationMessage],
    process_noise_acceleration: float = 1e-8,
) -> ThirdPartyTrackHistory:
    """Track one remote target with a persistent remote consider target."""

    times = np.asarray(timestamps, dtype=float).reshape(-1)
    if times.size == 0 or (times.size > 1 and not np.all(np.diff(times) > 0.0)):
        raise ValueError("timestamps must be nonempty and strictly increasing.")
    receiver = str(receiver_id)
    active_id = str(active_target_id)
    consider_id = str(consider_target_id)
    if active_id == consider_id:
        raise ValueError("Active and consider targets must be different.")
    tracks = {str(key): value for key, value in initial_target_estimate_by_id.items()}
    if set(tracks) != {active_id, consider_id}:
        raise ValueError("Schmidt pair requires exactly the active and consider tracks.")
    for target_id, estimate in tracks.items():
        if str(estimate.estimator_node_id) != receiver:
            raise ValueError("Every target track must belong to receiver_id.")
        if str(estimate.target_node_id) != target_id:
            raise ValueError("Target-track key does not match target ID.")
    state = initialize_multi_neighbor_schmidt(
        timestamp=float(times[0]), active_node_id=active_id,
        active_state=tracks[active_id].state_estimate,
        active_covariance=tracks[active_id].covariance,
        neighbor_state_by_id={consider_id: tracks[consider_id].state_estimate},
        neighbor_covariance_by_id={consider_id: tracks[consider_id].covariance},
    )
    grouped = {float(timestamp): [] for timestamp in times}
    for observation in observation_messages:
        timestamp = float(observation.timestamp)
        if timestamp not in grouped:
            raise ValueError("Observation timestamp is not in timestamps.")
        if {str(observation.observer_id), str(observation.target_id)} != {
            active_id, consider_id
        }:
            raise ValueError("Every observation must connect the Schmidt pair.")
        grouped[timestamp].append(observation)
    state_history = {
        active_id: np.zeros((times.size, 6)),
        consider_id: np.zeros((times.size, 6)),
    }
    covariance_history = {
        active_id: np.zeros((times.size, 6, 6)),
        consider_id: np.zeros((times.size, 6, 6)),
    }
    nis_history = []
    for index, timestamp in enumerate(times):
        if index > 0:
            state = multi_neighbor_schmidt_predict(
                state, float(timestamp),
                process_noise_acceleration=process_noise_acceleration,
            )
        epoch_nis = {}
        for observation in sorted(
            grouped[float(timestamp)], key=lambda item: item.information_id
        ):
            quaternion = (
                observation.metadata.get("quaternion_i2b_wxyz")
                if str(observation.frame).upper() == "BODY" else None
            )
            update = multi_neighbor_schmidt_update(
                state, observation, quaternion_i2b_wxyz=quaternion,
            )
            state = update.state
            epoch_nis[observation.information_id] = update.nis
        state_history[active_id][index] = state.active_state
        state_history[consider_id][index] = state.neighbor_state_by_id[consider_id]
        covariance_history[active_id][index] = state.active_covariance
        covariance_history[consider_id][index] = state.neighbor_covariance(consider_id)
        nis_history.append(epoch_nis)
    return ThirdPartyTrackHistory(
        receiver, times.copy(), state_history, covariance_history,
        nis_history, tuple(state.information_ids),
    )
