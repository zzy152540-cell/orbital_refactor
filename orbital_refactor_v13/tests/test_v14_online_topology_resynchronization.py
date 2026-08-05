import numpy as np
from orbital_core.measurement_integrity import (
    INTEGRITY_MODE_PROPORTIONAL_WITH_HARD_GATE,
    MeasurementIntegrityPolicy,
)

from experiments.v14_online_topology_resynchronization import (
    run_v14_online_topology_resynchronization_experiment,
)


def test_three_satellite_online_path_resynchronizes_long_topology_separation():
    result = run_v14_online_topology_resynchronization_experiment(
        seeds=1, duration=12.0, dt=2.0,
        inactive_window=(2.0, 8.0), max_pinned_age=2.0,
    )

    assert result.run_count == 1
    assert result.resynchronization_count == 2
    assert result.rejected_message_count == 0
    assert result.protocol_rejected_message_count == 0
    assert result.minimum_joint_covariance_eigenvalue >= -1e-8
    assert np.isfinite(result.mean_position_rmse)
    assert np.isfinite(result.mean_nees)
    assert ":resync:1" in result.final_lineage_by_directed_link[
        ("sat_a", "sat_b")
    ]
    assert ":resync:1" in result.final_lineage_by_directed_link[
        ("sat_b", "sat_a")
    ]


def test_online_path_resynchronizes_multiple_asynchronous_edges():
    result = run_v14_online_topology_resynchronization_experiment(
        seeds=1, duration=14.0, dt=2.0, max_pinned_age=2.0,
        topology_inactive_windows_by_undirected_edge={
            ("sat_a", "sat_b"): ((2.0, 8.0),),
            ("sat_b", "sat_c"): ((4.0, 10.0),),
        },
    )

    assert result.resynchronization_count == 4
    assert result.rejected_message_count == 0
    assert result.minimum_joint_covariance_eigenvalue >= -1e-8
    for edge in (
        ("sat_a", "sat_b"), ("sat_b", "sat_a"),
        ("sat_b", "sat_c"), ("sat_c", "sat_b"),
    ):
        assert ":resync:1" in result.final_lineage_by_directed_link[edge]


def test_repeated_edge_separations_advance_lineage_generation():
    result = run_v14_online_topology_resynchronization_experiment(
        seeds=1, duration=18.0, dt=2.0, max_pinned_age=2.0,
        topology_inactive_windows_by_undirected_edge={
            ("sat_a", "sat_b"): (
                (2.0, 6.0), (10.0, 14.0),
            ),
        },
    )

    assert result.resynchronization_count == 4
    assert result.rejected_message_count == 0
    assert ":resync:2" in result.final_lineage_by_directed_link[
        ("sat_a", "sat_b")
    ]
    assert ":resync:2" in result.final_lineage_by_directed_link[
        ("sat_b", "sat_a")
    ]


def test_delayed_messages_crossing_topology_change_are_classified_as_stale():
    result = run_v14_online_topology_resynchronization_experiment(
        seeds=1, duration=12.0, dt=2.0, max_pinned_age=2.0,
        communication_delay=4.0,
        topology_inactive_windows_by_undirected_edge={
            ("sat_a", "sat_b"): ((2.0, 8.0),),
        },
    )

    assert result.stale_topology_message_count > 0
    assert result.protocol_rejected_message_count == 0
    assert result.rejected_message_count == (
        result.stale_topology_message_count
    )


def test_online_path_applies_shared_integrity_policy_to_infrared_bias():
    policy = MeasurementIntegrityPolicy(
        mode=INTEGRITY_MODE_PROPORTIONAL_WITH_HARD_GATE,
        inflation_threshold=5.99, maximum_covariance_scale=9.0,
        hard_gate_threshold=9.21,
    )
    result = run_v14_online_topology_resynchronization_experiment(
        seeds=1, duration=12.0, dt=2.0,
        inactive_window=(20.0, 22.0), max_pinned_age=20.0,
        infrared_outlier_bias=(0.02, -0.02),
        infrared_outlier_window=(2.0, 6.0),
        integrity_policy_by_modality={"INFRARED": policy},
    )

    assert result.integrity_status_counts["HARD_REJECTED"] > 0
    assert result.protocol_rejected_message_count == 0
