import numpy as np

from cooperative.topology import chain_topology
from orbital_core.constants import R_EARTH
from orbital_core.coordinates import build_rtn_quaternion
from orbital_core.dynamics import propagate_absolute_orbit
from orbital_core.measurements import (
    measure_relative_range,
    measure_relative_range_rate,
)
from orbital_core.orbit_elements import keplerian_to_eci
from tracking import (
    AutonomousIODManager,
    DistributedTargetConsensus,
    DistributedTargetNodeNetwork,
    GlobalTargetEstimate,
    IODBufferConfig,
    MultiTargetAssociationPipeline,
    MultiTargetIODManager,
    MultiTargetIODUpdate,
    NeighborTrackIdentityConsensus,
    OnlineMultiTargetTracker,
    TargetInitialState,
    TargetNodeReport,
    UnlabeledIODObservation,
)


def _report(node, local_target, *, offset=0.0, timestamp=10.0):
    return TargetNodeReport(
        observer_id=node,
        target_id=local_target,
        track_id=f"track-{local_target}",
        timestamp=timestamp,
        state_eci=np.array([
            7.0e6 + offset, 100.0, -50.0, 0.0, 7500.0, 1.0,
        ]),
        covariance_eci=np.diag([4.0, 4.0, 4.0, 0.04, 0.04, 0.04]),
        quality_score=0.9,
        valid_flag=True,
        used_measurement_ids=(f"measurement-{node}-{timestamp}",),
    )


def _orbit_state(anomaly_deg, inclination_deg=55.0, altitude_m=700e3):
    return keplerian_to_eci(
        R_EARTH + altitude_m,
        0.001,
        np.deg2rad(inclination_deg),
        np.deg2rad(15.0),
        0.0,
        np.deg2rad(anomaly_deg),
    )


def _observations(observer_id, timestamp, observer, target):
    relative = target[:3] - observer[:3]
    group = f"{observer_id}-{timestamp}"
    return (
        UnlabeledIODObservation(
            timestamp=timestamp,
            observer_id=observer_id,
            modality="LOS",
            observer_state_eci=observer,
            measurement=relative / np.linalg.norm(relative),
            covariance=np.diag([2e-5, 2e-5]) ** 2,
            detection_group_id=group,
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


def test_neighbors_reconcile_independently_created_local_tracks():
    resolver = NeighborTrackIdentityConsensus(
        topology=chain_topology(("node-a", "node-b")),
        position_gate_m=100.0,
        velocity_gate_mps=10.0,
    )

    result = resolver.resolve((
        _report("node-a", "local-a", offset=0.0),
        _report("node-b", "local-b", offset=2.0),
    ))

    assert len({report.target_id for report in result.reports}) == 1
    assert len({report.track_id for report in result.reports}) == 1
    assignment = next(
        item for item in result.assignments if item.observer_id == "node-b"
    )
    cooperative_id = assignment.cooperative_target_id
    estimate = GlobalTargetEstimate(
        target_id=cooperative_id,
        track_id=assignment.cooperative_track_id,
        timestamp=10.0,
        state_eci=np.zeros(6),
        covariance_eci=np.eye(6),
        contributing_observer_ids=("node-a", "node-b"),
        node_weights={"node-a": 0.5, "node-b": 0.5},
        valid_flag=True,
    )
    localized = resolver.localize_estimates(
        observer_id="node-b", estimates={cooperative_id: estimate},
    )
    assert set(localized) == {"local-b"}
    assert localized["local-b"].track_id == "track-local-b"


def test_identity_propagates_across_two_hop_chain():
    resolver = NeighborTrackIdentityConsensus(
        topology=chain_topology(("node-a", "node-b", "node-c")),
        position_gate_m=100.0,
        velocity_gate_mps=10.0,
        exchange_hops=2,
    )

    result = resolver.resolve((
        _report("node-a", "local-a", offset=0.0),
        _report("node-b", "local-b", offset=2.0),
        _report("node-c", "local-c", offset=4.0),
    ))

    assert len({report.target_id for report in result.reports}) == 1
    identities = resolver.cooperative_target_ids_by_node()
    assert len({values[0] for values in identities.values()}) == 1


def test_well_separated_targets_are_not_merged():
    resolver = NeighborTrackIdentityConsensus(
        topology=chain_topology(("node-a", "node-b")),
        position_gate_m=100.0,
        velocity_gate_mps=10.0,
    )

    result = resolver.resolve((
        _report("node-a", "local-a", offset=0.0),
        _report("node-b", "local-b", offset=1000.0),
    ))

    assert len({report.target_id for report in result.reports}) == 2


def test_retired_local_track_reacquires_persistent_cooperative_identity():
    resolver = NeighborTrackIdentityConsensus(
        topology=chain_topology(("node-a", "node-b")),
        position_gate_m=100.0,
        velocity_gate_mps=10.0,
    )
    first = resolver.resolve((
        _report("node-a", "local-a", offset=0.0),
        _report("node-b", "local-b", offset=2.0),
    ))
    cooperative_id = first.reports[0].target_id
    assert resolver.retire_local_track(
        observer_id="node-b",
        local_target_id="local-b",
        local_track_id="track-local-b",
    )

    second = resolver.resolve((
        _report("node-a", "local-a", offset=10.0),
        _report("node-b", "local-b-new", offset=12.0),
    ))

    node_b = next(
        report for report in second.reports if report.observer_id == "node-b"
    )
    assert node_b.target_id == cooperative_id
    assert second.reacquired_cooperative_target_ids == (cooperative_id,)


def test_independent_node_filters_use_neighbor_identity_before_consensus_ci():
    times = np.array([0.0, 10.0, 20.0])
    target_history = propagate_absolute_orbit(_orbit_state(0.5), times)
    observer_histories = {
        "node-a": propagate_absolute_orbit(
            _orbit_state(-1.0, 54.7, 698e3), times,
        ),
        "node-b": propagate_absolute_orbit(
            _orbit_state(-0.5, 54.9, 702e3), times,
        ),
    }
    local_names = {"node-a": "local-a", "node-b": "local-b"}
    trackers = {
        node_id: OnlineMultiTargetTracker(
            scene_id="neighbor-identity-network",
            initial_states={
                local_name: TargetInitialState(
                    target_id=local_name,
                    track_id=f"track-{local_name}",
                    timestamp=0.0,
                    state_eci=target_history[0],
                    covariance_eci=np.diag([
                        100.0, 100.0, 100.0, 1.0, 1.0, 1.0,
                    ]),
                ),
            },
        )
        for node_id, local_name in local_names.items()
    }
    topology = chain_topology(tuple(trackers))
    network = DistributedTargetNodeNetwork(
        trackers_by_node=trackers,
        consensus=DistributedTargetConsensus(
            scene_id="neighbor-identity-network", topology=topology,
        ),
        identity_resolver=NeighborTrackIdentityConsensus(
            topology=topology,
            position_gate_m=100_000.0,
            velocity_gate_mps=500.0,
        ),
    )

    update = None
    for index, timestamp in enumerate(times):
        states = {
            node_id: history[index]
            for node_id, history in observer_histories.items()
        }
        update = network.step(
            timestamp=float(timestamp),
            observations_by_node={
                node_id: _observations(
                    node_id, float(timestamp), state, target_history[index],
                )
                for node_id, state in states.items()
            },
            observer_states_eci=states,
            q_eci2pri_by_observer={
                node_id: build_rtn_quaternion(state)
                for node_id, state in states.items()
            },
        )

    assert update is not None
    identities = {
        target_id
        for estimates in update.consensus.estimates_by_node.values()
        for target_id in estimates
    }
    assert len(identities) == 1
    for estimates in update.consensus.estimates_by_node.values():
        estimate = next(iter(estimates.values()))
        assert estimate.contributing_observer_ids == ("node-a", "node-b")


def test_each_node_performs_its_own_autonomous_iod_without_shared_initializer():
    times = np.array([0.0, 10.0, 20.0, 30.0])
    target_history = propagate_absolute_orbit(_orbit_state(0.5), times)
    observer_histories = {
        "node-a": propagate_absolute_orbit(
            _orbit_state(-1.0, 54.7, 698e3), times,
        ),
        "node-b": propagate_absolute_orbit(
            _orbit_state(-0.5, 54.9, 702e3), times,
        ),
    }
    trackers = {
        node_id: OnlineMultiTargetTracker(
            scene_id="autonomous-neighbor-identity",
            association_pipeline=MultiTargetAssociationPipeline(
                autonomous_iod_manager=AutonomousIODManager(
                    iod_manager=MultiTargetIODManager(
                        buffer_config=IODBufferConfig(
                            minimum_epochs=3,
                            minimum_observers=1,
                            minimum_time_span_seconds=20.0,
                            maximum_time_span_seconds=30.0,
                        ),
                    ),
                ),
            ),
        )
        for node_id in observer_histories
    }
    topology = chain_topology(tuple(trackers))
    network = DistributedTargetNodeNetwork(
        trackers_by_node=trackers,
        consensus=DistributedTargetConsensus(
            scene_id="autonomous-neighbor-identity", topology=topology,
        ),
        identity_resolver=NeighborTrackIdentityConsensus(
            topology=topology,
            position_gate_m=100_000.0,
            velocity_gate_mps=500.0,
        ),
    )

    updates = []
    for index, timestamp in enumerate(times):
        states = {
            node_id: history[index]
            for node_id, history in observer_histories.items()
        }
        updates.append(network.step(
            timestamp=float(timestamp),
            observations_by_node={
                node_id: _observations(
                    node_id, float(timestamp), state, target_history[index],
                )
                for node_id, state in states.items()
            },
            observer_states_eci=states,
            q_eci2pri_by_observer={
                node_id: build_rtn_quaternion(state)
                for node_id, state in states.items()
            },
        ))

    assert {
        node_id: sum(
            len(update.tracking_by_node[node_id].initialized_states_by_target)
            for update in updates
        )
        for node_id in trackers
    } == {"node-a": 1, "node-b": 1}
    assert trackers["node-a"] is not trackers["node-b"]


def test_discovery_channel_does_not_replace_node_local_tracking_input():
    class RecordingIOD:
        def __init__(self):
            self.received = None

        def ingest(self, observations):
            self.received = tuple(observations)
            return MultiTargetIODUpdate({}, {}, ())

    observer = _orbit_state(-1.0)
    target = _orbit_state(0.5)
    local = _observations("node-a", 0.0, observer, target)
    neighbor = _observations("node-b", 0.0, observer, target)
    iod = RecordingIOD()
    pipeline = MultiTargetAssociationPipeline(
        autonomous_iod_manager=iod,
    )

    update = pipeline.step(
        local, {}, discovery_observations=neighbor,
    )

    assert iod.received == neighbor
    assert update.association.unassigned_observation_indices == (0, 1)


def test_discovery_channel_excludes_observations_explained_by_active_tracks():
    class RecordingIOD:
        def __init__(self):
            self.received = None

        def ingest(self, observations):
            self.received = tuple(observations)
            return MultiTargetIODUpdate({}, {}, ())

    observer = _orbit_state(-1.0)
    target = _orbit_state(0.5)
    discovery = _observations("node-b", 0.0, observer, target)
    iod = RecordingIOD()
    pipeline = MultiTargetAssociationPipeline(
        autonomous_iod_manager=iod,
        minimum_cost_margin=0.0,
    )
    active = TargetInitialState(
        target_id="known-target",
        track_id="known-track",
        timestamp=0.0,
        state_eci=target,
        covariance_eci=np.diag([100.0, 100.0, 100.0, 1.0, 1.0, 1.0]),
    )

    pipeline.step((), {active.target_id: active}, discovery_observations=discovery)

    assert iod.received == ()
