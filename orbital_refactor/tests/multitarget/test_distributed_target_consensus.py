import numpy as np

from cooperative.message_transport import MessageChannel
from cooperative.topology import chain_topology
from tracking import DistributedTargetConsensus, TargetNodeReport


def _report(
    observer, target, timestamp, *, offset=0.0, information_id=None, valid=True,
):
    return TargetNodeReport(
        observer_id=observer,
        target_id=target,
        track_id=f"track-{target}",
        timestamp=timestamp,
        state_eci=np.array([7.0e6 + offset, 100.0, -50.0, 0.0, 7500.0, 1.0]),
        covariance_eci=np.diag([4.0, 4.0, 4.0, 0.04, 0.04, 0.04]),
        quality_score=0.9,
        valid_flag=valid,
        used_measurement_ids=(
            information_id or f"measurement-{observer}-{target}-{timestamp}",
        ),
    )


def test_three_observers_two_targets_fuse_only_across_topology_neighbors():
    consensus = DistributedTargetConsensus(
        scene_id="scene",
        topology=chain_topology(("observer_a", "observer_b", "observer_c")),
    )

    result = consensus.step(
        timestamp=10.0,
        local_reports=[
            _report("observer_a", "target_01", 10.0, offset=0.0),
            _report("observer_b", "target_01", 10.0, offset=2.0),
            _report("observer_c", "target_01", 10.0, offset=4.0),
            _report("observer_a", "target_02", 10.0, offset=1000.0),
            _report("observer_b", "target_02", 10.0, offset=1002.0),
            _report("observer_c", "target_02", 10.0, offset=1004.0),
        ],
    )

    middle = result.estimates_by_node["observer_b"]
    assert set(middle) == {"target_01", "target_02"}
    assert middle["target_01"].contributing_observer_ids == (
        "observer_a", "observer_b", "observer_c",
    )
    assert middle["target_02"].state_eci[0] - middle["target_01"].state_eci[0] > 900.0
    assert result.estimates_by_node["observer_a"][
        "target_01"
    ].contributing_observer_ids == ("observer_a", "observer_b")


def test_delayed_reports_are_buffered_and_aligned_to_current_epoch():
    consensus = DistributedTargetConsensus(
        scene_id="scene",
        topology=chain_topology(("observer_a", "observer_b")),
        message_channel=MessageChannel(
            delay_by_source={"observer_a": 1.0, "observer_b": 1.0},
        ),
    )

    first = consensus.step(
        timestamp=0.0,
        local_reports=[
            _report("observer_a", "target_01", 0.0),
            _report("observer_b", "target_01", 0.0, offset=2.0),
        ],
    )
    assert first.pending_message_count == 2
    assert first.estimates_by_node["observer_a"][
        "target_01"
    ].contributing_observer_ids == ("observer_a",)

    second = consensus.step(
        timestamp=1.0,
        local_reports=[
            _report("observer_a", "target_01", 1.0, offset=7500.0),
            _report("observer_b", "target_01", 1.0, offset=7502.0),
        ],
    )
    assert len(second.received_message_ids_by_node["observer_a"]) == 1
    assert second.estimates_by_node["observer_a"][
        "target_01"
    ].contributing_observer_ids == ("observer_a", "observer_b")
    assert second.pending_message_count == 2


def test_packet_loss_and_information_lineage_prevent_invalid_reuse():
    consensus = DistributedTargetConsensus(
        scene_id="scene",
        topology=chain_topology(("observer_a", "observer_b", "observer_c")),
        message_channel=MessageChannel(
            packet_loss_rate={"observer_c": 1.0},
            random_seed=3,
        ),
    )

    result = consensus.step(
        timestamp=10.0,
        local_reports=[
            _report(
                "observer_a", "target_01", 10.0,
                information_id="shared-physical-observation",
            ),
            _report(
                "observer_b", "target_01", 10.0,
                offset=2.0,
                information_id="shared-physical-observation",
            ),
            _report("observer_c", "target_01", 10.0, offset=4.0),
        ],
    )

    middle = result.estimates_by_node["observer_b"]["target_01"]
    assert middle.contributing_observer_ids == ("observer_b",)
    assert len(result.rejected_message_ids_by_node["observer_b"]) == 1
    assert result.received_message_ids_by_node["observer_b"] == (
        "scene:observer_a:target_01:track-target_01:10.000000000",
    )


def test_invalid_local_and_received_reports_do_not_create_an_estimate():
    consensus = DistributedTargetConsensus(
        scene_id="scene",
        topology=chain_topology(("observer_a", "observer_b")),
    )

    result = consensus.step(
        timestamp=10.0,
        local_reports=[
            _report("observer_a", "target_01", 10.0, valid=False),
            _report("observer_b", "target_01", 10.0, valid=False),
        ],
    )

    assert result.estimates_by_node == {"observer_a": {}, "observer_b": {}}
    assert len(result.rejected_message_ids_by_node["observer_a"]) == 1
    assert len(result.rejected_message_ids_by_node["observer_b"]) == 1


def test_three_by_two_node_outage_and_recovery_preserve_target_identity():
    consensus = DistributedTargetConsensus(
        scene_id="scene",
        topology=chain_topology(("observer_a", "observer_b", "observer_c")),
    )

    def epoch_reports(timestamp, active_nodes):
        return [
            _report(
                observer,
                target,
                timestamp,
                offset=target_offset + observer_offset,
            )
            for observer, observer_offset in (
                ("observer_a", 0.0),
                ("observer_b", 2.0),
                ("observer_c", 4.0),
            )
            if observer in active_nodes
            for target, target_offset in (
                ("target_01", 0.0),
                ("target_02", 1000.0),
            )
        ]

    nominal = consensus.step(
        timestamp=0.0,
        local_reports=epoch_reports(
            0.0, {"observer_a", "observer_b", "observer_c"},
        ),
    )
    outage = consensus.step(
        timestamp=1.0,
        local_reports=epoch_reports(1.0, {"observer_a", "observer_b"}),
    )
    recovered = consensus.step(
        timestamp=2.0,
        local_reports=epoch_reports(
            2.0, {"observer_a", "observer_b", "observer_c"},
        ),
    )

    assert nominal.estimates_by_node["observer_b"][
        "target_01"
    ].contributing_observer_ids == ("observer_a", "observer_b", "observer_c")
    assert outage.estimates_by_node["observer_b"][
        "target_01"
    ].contributing_observer_ids == ("observer_a", "observer_b")
    assert recovered.estimates_by_node["observer_b"][
        "target_01"
    ].contributing_observer_ids == ("observer_a", "observer_b", "observer_c")
    assert outage.available_source_ids_by_node["observer_b"] == ("observer_a",)
    assert recovered.recovered_source_ids_by_node["observer_b"] == ("observer_c",)
    for result in (nominal, outage, recovered):
        middle = result.estimates_by_node["observer_b"]
        assert set(middle) == {"target_01", "target_02"}
        assert middle["target_02"].state_eci[0] - middle["target_01"].state_eci[0] > 900.0


def test_active_source_without_target_report_is_not_a_link_outage():
    consensus = DistributedTargetConsensus(
        scene_id="scene",
        topology=chain_topology(("observer_a", "observer_b")),
    )

    empty = consensus.step(
        timestamp=0.0,
        local_reports=[],
        active_source_ids=("observer_a", "observer_b"),
    )
    reported = consensus.step(
        timestamp=1.0,
        local_reports=[
            _report("observer_a", "target_01", 1.0),
            _report("observer_b", "target_01", 1.0, offset=2.0),
        ],
        active_source_ids=("observer_a", "observer_b"),
    )

    assert empty.recovered_source_ids_by_node == {
        "observer_a": (), "observer_b": (),
    }
    assert reported.recovered_source_ids_by_node == {
        "observer_a": (), "observer_b": (),
    }


def test_information_lineage_propagates_across_multiple_chain_hops():
    node_ids = ("observer_a", "observer_b", "observer_c", "observer_d")
    consensus = DistributedTargetConsensus(
        scene_id="scene",
        topology=chain_topology(node_ids),
    )
    inherited = {node_id: () for node_id in node_ids}
    final = None
    for epoch in range(3):
        reports = [
            TargetNodeReport(
                observer_id=node_id,
                target_id="target_01",
                track_id="track-target_01",
                timestamp=float(epoch),
                state_eci=np.array([
                    7.0e6 + index, 7500.0 * epoch, -50.0,
                    0.0, 7500.0, 1.0,
                ]),
                covariance_eci=np.diag([
                    4.0, 4.0, 4.0, 0.04, 0.04, 0.04,
                ]),
                quality_score=0.9,
                used_measurement_ids=tuple(dict.fromkeys((
                    *inherited[node_id],
                    f"epoch-{epoch}:{node_id}",
                ))),
            )
            for index, node_id in enumerate(node_ids)
        ]
        final = consensus.step(
            timestamp=float(epoch),
            local_reports=reports,
            active_source_ids=node_ids,
        )
        inherited = {
            node_id: final.estimates_by_node[node_id][
                "target_01"
            ].information_ids
            for node_id in node_ids
        }

    assert final is not None
    left_information = set(
        final.estimates_by_node["observer_a"]["target_01"].information_ids
    )
    assert "epoch-0:observer_d" in left_information
    assert "epoch-2:observer_a" in left_information
