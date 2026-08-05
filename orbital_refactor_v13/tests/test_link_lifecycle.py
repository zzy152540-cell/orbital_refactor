import pytest

from cooperative.link_lifecycle import LinkLifecycle, LinkLifecycleState


def test_short_suspension_resumes_same_lineage_when_history_is_available():
    link = LinkLifecycle("receiver", "source", "source->receiver:0")

    suspended = link.suspend(topology_version=1)
    resumed = suspended.resume(topology_version=2, history_available=True)

    assert suspended.state == LinkLifecycleState.SUSPENDED
    assert resumed.state == LinkLifecycleState.ACTIVE
    assert resumed.lineage_id == link.lineage_id
    assert resumed.suspension_count == 1
    assert resumed.accepts(lineage_id=link.lineage_id, topology_version=2)


def test_long_suspension_requires_distinct_resynchronized_lineage():
    link = LinkLifecycle("receiver", "source", "source->receiver:0")
    waiting = link.suspend(topology_version=1).resume(
        topology_version=2, history_available=False,
        unavailable_reason="max_pinned_age_exceeded",
    )

    assert waiting.state == LinkLifecycleState.RESYNC_REQUIRED
    assert not waiting.accepts(
        lineage_id="source->receiver:0", topology_version=2
    )
    with pytest.raises(ValueError, match="distinct lineage"):
        waiting.establish_resynchronized_lineage(
            lineage_id="source->receiver:0"
        )

    restored = waiting.establish_resynchronized_lineage(
        lineage_id="source->receiver:resync:1"
    )
    assert restored.state == LinkLifecycleState.ACTIVE
    assert restored.resynchronization_reason is None
    assert restored.resynchronization_count == 1
    assert restored.accepts(
        lineage_id="source->receiver:resync:1", topology_version=2
    )


def test_topology_versions_are_monotonic_and_new_lineage_is_state_guarded():
    link = LinkLifecycle("receiver", "source", "source->receiver:0")

    with pytest.raises(ValueError, match="newer version"):
        link.suspend(topology_version=0)
    with pytest.raises(ValueError, match="awaiting resync"):
        link.establish_resynchronized_lineage(
            lineage_id="source->receiver:resync:1"
        )
