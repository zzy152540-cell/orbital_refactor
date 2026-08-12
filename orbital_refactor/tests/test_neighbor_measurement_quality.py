import numpy as np

from cooperative.neighbor_measurement_quality import (
    build_neighbor_link_quality_schedule,
    NeighborMeasurementQualityPolicy,
)


def test_two_level_quality_policy_combines_base_and_link_penalties():
    policy = NeighborMeasurementQualityPolicy(
        base_inflation_by_modality={"RANGE_RATE": 32.0},
        age_grace=2.0,
        age_inflation_per_second=4.0,
        loss_inflation_per_packet=16.0,
        resynchronization_inflation=1000.0,
        maximum_inflation=128.0,
    )

    assert policy.inflation(
        modality="RANGE", age=0.0, consecutive_losses=0,
        resynchronization_required=False,
    ) == 0.0
    assert policy.inflation(
        modality="RANGE", age=10.0, consecutive_losses=5,
        resynchronization_required=True,
    ) == 0.0
    assert policy.inflation(
        modality="RANGE_RATE", age=0.0, consecutive_losses=0,
        resynchronization_required=False,
    ) == 32.0
    assert policy.inflation(
        modality="RANGE_RATE", age=4.0, consecutive_losses=1,
        resynchronization_required=False,
    ) == 56.0
    assert policy.inflation(
        modality="RANGE_RATE", age=4.0, consecutive_losses=1,
        resynchronization_required=True,
    ) == 128.0


def test_quality_policy_rejects_invalid_parameters():
    with np.testing.assert_raises_regex(ValueError, "nonnegative"):
        NeighborMeasurementQualityPolicy(age_grace=-1.0)


def test_link_quality_schedule_uses_only_records_available_by_epoch():
    schedule = build_neighbor_link_quality_schedule(
        receiver_id="sat_01",
        timestamps=(0.0, 2.0, 4.0, 6.0),
        neighbor_ids=("sat_02",),
        message_records=(
            {
                "source_id": "sat_02", "current_timestamp": 0.0,
                "receiver_id": "sat_01",
                "message_timestamp": 0.0,
                "arrival_timestamp": 0.0, "accepted": True,
                "consecutive_losses_before_delivery": 0,
            },
            {
                "source_id": "sat_02", "current_timestamp": 6.0,
                "receiver_id": "sat_01",
                "message_timestamp": 4.0,
                "arrival_timestamp": 6.0, "accepted": True,
                "consecutive_losses_before_delivery": 2,
            },
        ),
    )

    assert schedule[("sat_01", "sat_02", 0.0)].age == 0.0
    assert schedule[("sat_01", "sat_02", 4.0)].age == 4.0
    assert schedule[("sat_01", "sat_02", 4.0)].consecutive_losses == 0
    assert schedule[("sat_01", "sat_02", 6.0)].age == 2.0
    assert schedule[("sat_01", "sat_02", 6.0)].consecutive_losses == 2


def test_link_quality_schedule_does_not_mix_receivers():
    schedule = build_neighbor_link_quality_schedule(
        receiver_id="sat_01",
        timestamps=(0.0, 2.0),
        neighbor_ids=("sat_03",),
        message_records=({
            "receiver_id": "sat_02", "source_id": "sat_03",
            "current_timestamp": 2.0, "message_timestamp": 2.0,
            "accepted": True,
        },),
    )

    assert schedule[("sat_01", "sat_03", 2.0)].age == 2.0
