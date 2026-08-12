import pytest

from experiments.five_node_risk_validation_batches import (
    combine_five_node_risk_validation_batches,
    deserialize_five_node_risk_validation_batch,
    run_five_node_risk_validation_batch,
    serialize_five_node_risk_validation_batch,
)


def _batch(batch_id, seeds):
    return run_five_node_risk_validation_batch(
        batch_id=batch_id, seeds=seeds,
        decision_epochs=(1,), horizon_epochs=(1,),
        packet_loss_by_edge={("sat_01", "sat_05"): 0.4},
        communication_delay_by_edge={("sat_01", "sat_05"): 2.0},
    )


def test_five_node_batches_combine_seed_disjoint_action_records():
    combined = combine_five_node_risk_validation_batches(
        _batch("first", (0,)), _batch("second", (1,))
    )

    assert combined.node_counts == (5,)
    assert combined.seeds == (0, 1)
    assert len(combined.decision_observations) == 2
    assert len(combined.records) == 54


def test_five_node_batches_reject_overlapping_seeds():
    first = _batch("first", (0,))
    second = _batch("second", (0,))
    with pytest.raises(ValueError, match="disjoint"):
        combine_five_node_risk_validation_batches(first, second)


def test_five_node_batch_round_trips_trusted_local_payload():
    batch = _batch("saved", (0,))
    payload = serialize_five_node_risk_validation_batch(batch)

    assert deserialize_five_node_risk_validation_batch(payload) == batch
