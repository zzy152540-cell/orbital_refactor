"""Dynamic observer join/leave acceptance for cooperative target identity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brain_inspired import build_multitarget_brain_sidecar
from cooperative.topology import chain_topology, fully_connected_topology
from orbital_core.constants import R_EARTH
from orbital_core.coordinates import build_rtn_quaternion
from orbital_core.dynamics import propagate_absolute_orbit
from orbital_core.measurements import (
    measure_relative_range,
    measure_relative_range_rate,
)
from orbital_core.orbit_elements import keplerian_to_eci
from tracking import (
    CooperativeTrackIdentityResolver,
    DistributedTargetConsensus,
    DistributedTargetNodeNetwork,
    OnlineMultiTargetTracker,
    TargetInitialState,
    UnlabeledIODObservation,
)


@dataclass(frozen=True)
class DynamicMembershipAcceptance:
    initial_node_ids: tuple[str, ...]
    joined_node_ids: tuple[str, ...]
    removed_node_ids: tuple[str, ...]
    initial_cooperative_target_count: int
    final_cooperative_target_count: int
    identity_switch_count: int
    false_cooperative_birth_count: int
    departed_contributor_count: int
    final_target_count: int
    brain_target_count: int
    brain_valid_fraction: float
    brain_direction_neuron_count: int
    brain_place_heatmap_shape: tuple[int, int]
    brain_place_normalization_error: float
    passed: bool


def run_multitarget_dynamic_membership_acceptance(
) -> DynamicMembershipAcceptance:
    """Replace the identity-origin node, then leave only the joining node."""

    timestamps = np.array([0.0, 10.0, 20.0, 30.0])
    scene_id = "multitarget-dynamic-membership-acceptance"
    target_initials = {
        "physical-01": _state(0.5, 55.0, 700e3),
        "physical-02": _state(3.5, 55.3, 705e3),
    }
    observer_initials = {
        "observer-a": _state(-1.0, 54.7, 698e3),
        "observer-b": _state(-0.5, 54.9, 702e3),
        "observer-c": _state(-1.5, 55.1, 704e3),
    }
    target_histories = {
        key: propagate_absolute_orbit(value, timestamps)
        for key, value in target_initials.items()
    }
    observer_histories = {
        key: propagate_absolute_orbit(value, timestamps)
        for key, value in observer_initials.items()
    }
    local_names = {
        "observer-a": ("alpha", "beta"),
        "observer-b": ("red", "blue"),
        "observer-c": ("green", "gold"),
    }
    trackers = {
        observer_id: OnlineMultiTargetTracker(
            scene_id=scene_id,
            initial_states={
                local_target_id: TargetInitialState(
                    target_id=local_target_id,
                    track_id=f"track-{observer_id}-{local_target_id}",
                    timestamp=0.0,
                    state_eci=target_initials[physical_target_id],
                    covariance_eci=np.diag(
                        [100.0**2] * 3 + [0.5**2] * 3
                    ),
                )
                for local_target_id, physical_target_id in zip(
                    local_names[observer_id], target_initials, strict=True,
                )
            },
        )
        for observer_id in observer_initials
    }
    resolver = CooperativeTrackIdentityResolver(
        position_gate_m=100_000.0,
        velocity_gate_mps=500.0,
    )
    consensus = DistributedTargetConsensus(
        scene_id=scene_id,
        topology=fully_connected_topology(("observer-a", "observer-b")),
    )
    network = DistributedTargetNodeNetwork(
        trackers_by_node={
            key: trackers[key] for key in ("observer-a", "observer-b")
        },
        consensus=consensus,
        identity_resolver=resolver,
    )
    _prime_nodes(
        network,
        node_ids=("observer-a", "observer-b"),
        timestamp=0.0,
        epoch_index=0,
        observer_histories=observer_histories,
    )
    trackers["observer-c"].step(
        timestamp=0.0,
        observations=(),
        observer_states_eci={
            "observer-c": observer_histories["observer-c"][0],
        },
        q_eci2pri_by_observer={
            "observer-c": build_rtn_quaternion(
                observer_histories["observer-c"][0]
            ),
        },
    )
    first = _step_nodes(
        network,
        node_ids=("observer-a", "observer-b"),
        timestamp=10.0,
        epoch_index=1,
        observer_histories=observer_histories,
        target_histories=target_histories,
        local_names=local_names,
    )
    expected_identity = _identity_by_physical_target(
        first, local_names, ("observer-a", "observer-b"),
    )
    initial_cooperative_ids = set(expected_identity.values())

    first_change = network.update_membership(
        trackers_by_node={
            key: trackers[key] for key in ("observer-b", "observer-c")
        },
        topology=fully_connected_topology(("observer-b", "observer-c")),
    )
    joined = _step_nodes(
        network,
        node_ids=("observer-b", "observer-c"),
        timestamp=20.0,
        epoch_index=2,
        observer_histories=observer_histories,
        target_histories=target_histories,
        local_names=local_names,
    )
    joined_identity = _identity_by_physical_target(
        joined, local_names, ("observer-b", "observer-c"),
    )

    second_change = network.update_membership(
        trackers_by_node={"observer-c": trackers["observer-c"]},
        topology=chain_topology(("observer-c",)),
    )
    final = _step_nodes(
        network,
        node_ids=("observer-c",),
        timestamp=30.0,
        epoch_index=3,
        observer_histories=observer_histories,
        target_histories=target_histories,
        local_names=local_names,
    )
    final_identity = _identity_by_physical_target(
        final, local_names, ("observer-c",),
    )
    identity_switches = sum(
        candidate.get(target_id) != cooperative_target_id
        for candidate in (joined_identity, final_identity)
        for target_id, cooperative_target_id in expected_identity.items()
    )
    all_created = {
        *first.identity_resolution.created_cooperative_target_ids,
        *joined.identity_resolution.created_cooperative_target_ids,
        *final.identity_resolution.created_cooperative_target_ids,
    }
    contributors = {
        contributor
        for estimates in final.consensus.estimates_by_node.values()
        for estimate in estimates.values()
        for contributor in estimate.contributing_observer_ids
    }
    departed_contributors = contributors & {"observer-a", "observer-b"}
    final_target_count = min(
        len(values) for values in final.consensus.estimates_by_node.values()
    )
    final_cooperative_ids = set(resolver.cooperative_target_ids)
    false_births = len(all_created - initial_cooperative_ids)
    brain = _build_brain_sidecar(
        updates=(first, joined, final),
        timestamps=timestamps[1:],
        cooperative_target_ids=initial_cooperative_ids,
    )
    brain_snapshots = {
        target_id: brain.snapshot(target_id, -1)
        for target_id in sorted(initial_cooperative_ids)
    }
    brain_valid = np.concatenate([
        history.valid for history in brain.navigation_by_target.values()
    ])
    place_normalization_error = max(
        abs(float(np.sum(snapshot.place_heatmap_3x3)) - 1.0)
        for snapshot in brain_snapshots.values()
    )
    direction_neuron_counts = {
        snapshot.direction_ring.size for snapshot in brain_snapshots.values()
    }
    heatmap_shapes = {
        snapshot.place_heatmap_3x3.shape for snapshot in brain_snapshots.values()
    }
    passed = bool(
        first_change.added_node_ids == ("observer-c",)
        and first_change.removed_node_ids == ("observer-a",)
        and second_change.removed_node_ids == ("observer-b",)
        and identity_switches == 0
        and false_births == 0
        and not departed_contributors
        and final_cooperative_ids == initial_cooperative_ids
        and final_target_count == len(target_initials)
        and set(brain.navigation_by_target) == initial_cooperative_ids
        and np.all(brain_valid)
        and direction_neuron_counts == {180}
        and heatmap_shapes == {(3, 3)}
        and place_normalization_error < 1.0e-12
    )
    return DynamicMembershipAcceptance(
        initial_node_ids=("observer-a", "observer-b"),
        joined_node_ids=first_change.added_node_ids,
        removed_node_ids=tuple(sorted({
            *first_change.removed_node_ids,
            *second_change.removed_node_ids,
        })),
        initial_cooperative_target_count=len(initial_cooperative_ids),
        final_cooperative_target_count=len(final_cooperative_ids),
        identity_switch_count=int(identity_switches),
        false_cooperative_birth_count=false_births,
        departed_contributor_count=len(departed_contributors),
        final_target_count=final_target_count,
        brain_target_count=len(brain.navigation_by_target),
        brain_valid_fraction=float(np.mean(brain_valid)),
        brain_direction_neuron_count=next(iter(direction_neuron_counts)),
        brain_place_heatmap_shape=next(iter(heatmap_shapes)),
        brain_place_normalization_error=place_normalization_error,
        passed=passed,
    )


def _build_brain_sidecar(*, updates, timestamps, cooperative_target_ids):
    state_history_by_target = {}
    for target_id in sorted(cooperative_target_ids):
        states = []
        for update in updates:
            candidates = [
                estimates[target_id].state_eci
                for estimates in update.consensus.estimates_by_node.values()
                if target_id in estimates
            ]
            if not candidates:
                raise AssertionError(
                    f"Cooperative target {target_id!r} is missing from an epoch."
                )
            states.append(candidates[0])
        state_history_by_target[target_id] = np.asarray(states)
    return build_multitarget_brain_sidecar(
        timestamps=np.asarray(timestamps, dtype=float),
        state_history_by_target=state_history_by_target,
    )


def _prime_nodes(network, *, node_ids, timestamp, epoch_index, observer_histories):
    states = {
        node_id: observer_histories[node_id][epoch_index]
        for node_id in node_ids
    }
    network.step(
        timestamp=timestamp,
        observations_by_node={node_id: () for node_id in node_ids},
        observer_states_eci=states,
        q_eci2pri_by_observer={
            node_id: build_rtn_quaternion(state)
            for node_id, state in states.items()
        },
    )


def _step_nodes(
    network,
    *,
    node_ids,
    timestamp,
    epoch_index,
    observer_histories,
    target_histories,
    local_names,
):
    states = {
        node_id: observer_histories[node_id][epoch_index]
        for node_id in node_ids
    }
    observations = {
        node_id: tuple(
            observation
            for target_index, target_history in enumerate(
                target_histories.values()
            )
            for observation in _observations(
                node_id,
                timestamp,
                states[node_id],
                target_history[epoch_index],
                group=f"{node_id}-{epoch_index}-{target_index}",
            )
        )
        for node_id in node_ids
    }
    return network.step(
        timestamp=timestamp,
        observations_by_node=observations,
        observer_states_eci=states,
        q_eci2pri_by_observer={
            node_id: build_rtn_quaternion(state)
            for node_id, state in states.items()
        },
    )


def _identity_by_physical_target(update, local_names, node_ids):
    physical_ids = ("physical-01", "physical-02")
    physical_by_local = {
        (node_id, local_target_id): physical_target_id
        for node_id in node_ids
        for local_target_id, physical_target_id in zip(
            local_names[node_id], physical_ids, strict=True,
        )
    }
    result = {}
    for assignment in update.identity_resolution.assignments:
        physical_target_id = physical_by_local[
            (assignment.observer_id, assignment.local_target_id)
        ]
        previous = result.setdefault(
            physical_target_id, assignment.cooperative_target_id,
        )
        if previous != assignment.cooperative_target_id:
            raise AssertionError("One physical target received competing identities.")
    return result


def _observations(observer_id, timestamp, observer, target, *, group):
    relative = target[:3] - observer[:3]
    return (
        UnlabeledIODObservation(
            timestamp=timestamp,
            observer_id=observer_id,
            modality="LOS",
            observer_state_eci=observer,
            measurement=relative / np.linalg.norm(relative),
            covariance=np.diag([2e-5, 2e-5]) ** 2,
            detection_group_id=group,
            metadata={"sourceModality": "INFRARED"},
        ),
        UnlabeledIODObservation(
            timestamp=timestamp,
            observer_id=observer_id,
            modality="RADAR",
            observer_state_eci=observer,
            measurement=np.array([
                measure_relative_range(observer, target),
                measure_relative_range_rate(observer, target),
            ]),
            covariance=np.diag([10.0, 0.05]) ** 2,
            detection_group_id=group,
        ),
    )


def _state(anomaly_deg, inclination_deg, altitude_m):
    return keplerian_to_eci(
        R_EARTH + altitude_m,
        0.001,
        np.deg2rad(inclination_deg),
        np.deg2rad(15.0),
        0.0,
        np.deg2rad(anomaly_deg),
    )
