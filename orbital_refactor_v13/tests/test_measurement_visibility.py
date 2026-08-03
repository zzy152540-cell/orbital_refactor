import numpy as np
import pytest

from orbital_core.constants import R_EARTH
from scenarios.measurement_visibility import (
    MeasurementOpportunity,
    VisibilityConfig,
    VisibilityResult,
    VisibilityTemporalFilterConfig,
    evaluate_inter_satellite_visibility,
    generate_inter_satellite_observation_opportunities,
    summarize_observation_opportunities,
    stabilize_observation_opportunities,
)
from cooperative.topology import chain_topology


ALTITUDE = 700e3
ORBIT_RADIUS = R_EARTH + ALTITUDE


def test_nearby_satellites_have_clear_line_of_sight():
    result = evaluate_inter_satellite_visibility(
        np.array([ORBIT_RADIUS, 0.0, 0.0]),
        np.array([ORBIT_RADIUS, 1000.0, 0.0]),
    )

    assert result.visible
    assert result.reason == "visible"
    assert result.range == pytest.approx(1000.0)
    assert result.earth_clearance == pytest.approx(ALTITUDE)


def test_satellites_on_opposite_sides_of_earth_are_occulted_symmetrically():
    first = np.array([ORBIT_RADIUS, 0.0, 0.0])
    second = -first

    forward = evaluate_inter_satellite_visibility(first, second)
    reverse = evaluate_inter_satellite_visibility(second, first)

    assert not forward.visible
    assert forward.reason == "earth_occulted"
    assert reverse == forward
    assert forward.earth_clearance == pytest.approx(-R_EARTH)


def test_tangent_line_is_treated_as_occulted_at_the_boundary():
    offset = 3e6
    first = np.array([-offset, R_EARTH, 0.0])
    second = np.array([offset, R_EARTH, 0.0])

    result = evaluate_inter_satellite_visibility(first, second)

    assert not result.visible
    assert result.reason == "earth_occulted"
    assert result.earth_clearance == pytest.approx(0.0, abs=1e-8)


def test_maximum_range_is_applied_after_clear_line_of_sight():
    result = evaluate_inter_satellite_visibility(
        np.array([ORBIT_RADIUS, 0.0, 0.0]),
        np.array([ORBIT_RADIUS, 2000.0, 0.0]),
        VisibilityConfig(maximum_range=1500.0),
    )

    assert not result.visible
    assert result.reason == "range_exceeded"


def test_body_fov_uses_positive_x_boresight_and_reports_angle():
    observer = np.array([ORBIT_RADIUS, 0.0, 0.0])
    identity_i2b = np.array([1.0, 0.0, 0.0, 0.0])
    config = VisibilityConfig(field_of_view_half_angle=np.deg2rad(20.0))

    on_axis = evaluate_inter_satellite_visibility(
        observer, observer + np.array([1000.0, 0.0, 0.0]), config,
        quaternion_i2b_wxyz=identity_i2b,
    )
    off_axis = evaluate_inter_satellite_visibility(
        observer, observer + np.array([0.0, 1000.0, 0.0]), config,
        quaternion_i2b_wxyz=identity_i2b,
    )

    assert on_axis.visible
    assert on_axis.off_boresight_angle == pytest.approx(0.0)
    assert not off_axis.visible
    assert off_axis.reason == "outside_fov"
    assert off_axis.off_boresight_angle == pytest.approx(np.pi / 2.0)


def test_fov_requires_attitude_and_valid_half_angle():
    observer = np.array([ORBIT_RADIUS, 0.0, 0.0])
    target = observer + np.array([1000.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="field_of_view_half_angle"):
        VisibilityConfig(field_of_view_half_angle=0.0)
    with pytest.raises(ValueError, match="requires observer quaternion"):
        evaluate_inter_satellite_visibility(
            observer, target,
            VisibilityConfig(field_of_view_half_angle=np.deg2rad(20.0)),
        )


def test_earth_occultation_can_be_disabled_for_control_experiments():
    first = np.array([ORBIT_RADIUS, 0.0, 0.0])
    result = evaluate_inter_satellite_visibility(
        first, -first, VisibilityConfig(earth_occultation=False),
    )

    assert result.visible
    assert result.earth_clearance < 0.0


def test_invalid_geometry_and_configuration_are_rejected():
    same = np.array([ORBIT_RADIUS, 0.0, 0.0])
    result = evaluate_inter_satellite_visibility(same, same)
    assert not result.visible
    assert result.reason == "invalid_geometry"

    inside = evaluate_inter_satellite_visibility(
        np.array([R_EARTH - 1.0, 0.0, 0.0]), same,
    )
    assert not inside.visible
    assert inside.reason == "invalid_geometry"

    with pytest.raises(ValueError, match="maximum_range"):
        VisibilityConfig(maximum_range=0.0)
    with pytest.raises(ValueError, match="3-position or 6-state"):
        evaluate_inter_satellite_visibility(np.zeros(2), same)


def test_opportunity_generator_records_every_directed_edge_and_modality():
    times = np.array([0.0, 2.0])
    topology = chain_topology(["a", "b", "c"])
    truth = {
        "a": np.array([[ORBIT_RADIUS, 0.0, 0.0], [ORBIT_RADIUS, 10.0, 0.0]]),
        "b": np.array([[ORBIT_RADIUS, 1000.0, 0.0], [ORBIT_RADIUS, 1010.0, 0.0]]),
        "c": np.array([[ORBIT_RADIUS, 2000.0, 0.0], [ORBIT_RADIUS, 2010.0, 0.0]]),
    }

    opportunities = generate_inter_satellite_observation_opportunities(
        timestamps=times, truth_state_history_by_node=truth,
        candidate_topology=topology,
        visibility_by_modality={
            "RANGE": VisibilityConfig(maximum_range=1500.0),
            "RANGE_RATE": VisibilityConfig(maximum_range=1500.0),
        },
    )

    assert len(opportunities) == 16
    assert all(item.visibility.visible for item in opportunities)
    assert {item.modality for item in opportunities} == {"RANGE", "RANGE_RATE"}
    first_epoch = [item for item in opportunities if item.timestamp == 0.0]
    assert {(item.observer_id, item.target_id) for item in first_epoch} == {
        ("a", "b"), ("b", "a"), ("b", "c"), ("c", "b"),
    }


def test_opportunity_generator_preserves_invisible_reasons():
    times = np.array([0.0])
    topology = chain_topology(["a", "b"])
    truth = {
        "a": np.array([[ORBIT_RADIUS, 0.0, 0.0]]),
        "b": np.array([[-ORBIT_RADIUS, 0.0, 0.0]]),
    }

    opportunities = generate_inter_satellite_observation_opportunities(
        timestamps=times, truth_state_history_by_node=truth,
        candidate_topology=topology,
        visibility_by_modality={"RANGE": VisibilityConfig()},
    )

    assert len(opportunities) == 2
    assert all(not item.visibility.visible for item in opportunities)
    assert {item.visibility.reason for item in opportunities} == {"earth_occulted"}


def test_opportunity_generator_applies_observer_attitude_history_to_fov():
    times = np.array([0.0])
    topology = chain_topology(["a", "b"])
    truth = {
        "a": np.array([[ORBIT_RADIUS, 0.0, 0.0]]),
        "b": np.array([[ORBIT_RADIUS + 1000.0, 0.0, 0.0]]),
    }
    attitudes = {
        node: np.array([[1.0, 0.0, 0.0, 0.0]]) for node in truth
    }

    opportunities = generate_inter_satellite_observation_opportunities(
        timestamps=times, truth_state_history_by_node=truth,
        candidate_topology=topology,
        visibility_by_modality={
            "AZ_EL": VisibilityConfig(
                field_of_view_half_angle=np.deg2rad(20.0)
            ),
        },
        attitude_history_by_node=attitudes,
    )

    by_observer = {item.observer_id: item for item in opportunities}
    assert by_observer["a"].visibility.visible
    assert by_observer["b"].visibility.reason == "outside_fov"


def test_opportunity_generator_validates_inputs():
    topology = chain_topology(["a", "b"])
    truth = {
        "a": np.array([[ORBIT_RADIUS, 0.0, 0.0]]),
        "b": np.array([[ORBIT_RADIUS, 1000.0, 0.0]]),
    }
    with pytest.raises(ValueError, match="At least one visibility modality"):
        generate_inter_satellite_observation_opportunities(
            timestamps=np.array([0.0]), truth_state_history_by_node=truth,
            candidate_topology=topology, visibility_by_modality={},
        )
    with pytest.raises(ValueError, match="Truth-history keys"):
        generate_inter_satellite_observation_opportunities(
            timestamps=np.array([0.0]), truth_state_history_by_node={"a": truth["a"]},
            candidate_topology=topology,
            visibility_by_modality={"RANGE": VisibilityConfig()},
        )


def test_opportunity_summary_reports_modalities_edges_and_epoch_counts():
    times = np.array([0.0, 2.0])
    topology = chain_topology(["a", "b", "c"])
    truth = {
        "a": np.array([[ORBIT_RADIUS, 0.0, 0.0]] * 2),
        "b": np.array([[ORBIT_RADIUS, 1000.0, 0.0]] * 2),
        "c": np.array([[ORBIT_RADIUS, 2000.0, 0.0]] * 2),
    }
    opportunities = generate_inter_satellite_observation_opportunities(
        timestamps=times, truth_state_history_by_node=truth,
        candidate_topology=topology,
        visibility_by_modality={
            "RANGE": VisibilityConfig(), "RANGE_RATE": VisibilityConfig(),
        },
    )

    summary = summarize_observation_opportunities(opportunities)

    assert summary.overall.opportunity_count == 16
    assert summary.overall.visibility_rate == 1.0
    assert summary.by_modality["RANGE"].opportunity_count == 8
    assert summary.by_directed_edge[("a", "b")].opportunity_count == 4
    assert summary.visible_directed_edge_count_by_timestamp == {0.0: 4, 2.0: 4}
    assert summary.visible_directed_edge_count_by_timestamp_and_modality[
        (0.0, "RANGE")
    ] == 4
    assert all(
        value == 0
        for value in summary.longest_unavailable_epochs_by_edge_and_modality.values()
    )


def test_opportunity_summary_reports_consecutive_range_outage():
    times = np.array([0.0, 2.0, 4.0, 6.0])
    topology = chain_topology(["a", "b"])
    truth = {
        "a": np.array([[ORBIT_RADIUS, 0.0, 0.0]] * 4),
        "b": np.array([
            [ORBIT_RADIUS, 1000.0, 0.0],
            [ORBIT_RADIUS, 2000.0, 0.0],
            [ORBIT_RADIUS, 2500.0, 0.0],
            [ORBIT_RADIUS, 1000.0, 0.0],
        ]),
    }
    opportunities = generate_inter_satellite_observation_opportunities(
        timestamps=times, truth_state_history_by_node=truth,
        candidate_topology=topology,
        visibility_by_modality={"RANGE": VisibilityConfig(maximum_range=1500.0)},
    )

    summary = summarize_observation_opportunities(opportunities)

    assert summary.overall.visibility_rate == 0.5
    assert summary.overall.rejection_counts == {"range_exceeded": 4}
    assert summary.visible_directed_edge_count_by_timestamp == {
        0.0: 2, 2.0: 0, 4.0: 0, 6.0: 2,
    }
    for edge in (("a", "b", "RANGE"), ("b", "a", "RANGE")):
        assert summary.longest_unavailable_epochs_by_edge_and_modality[edge] == 2
        assert summary.longest_unavailable_span_by_edge_and_modality[edge] == 2.0


def test_opportunity_summary_rejects_empty_input():
    with pytest.raises(ValueError, match="At least one measurement opportunity"):
        summarize_observation_opportunities(())


def test_fov_temporal_filter_suppresses_boundary_chatter():
    half_angle = np.deg2rad(5.0)
    angles = np.deg2rad([4.7, 5.1, 4.9, 5.2, 5.3, 4.8, 4.6, 4.4, 4.3])
    raw = tuple(
        MeasurementOpportunity(
            timestamp=float(index), observer_id="a", target_id="b",
            modality="AZ_EL",
            visibility=VisibilityResult(
                visible=angle <= half_angle,
                reason="visible" if angle <= half_angle else "outside_fov",
                range=1000.0, earth_clearance=ALTITUDE,
                off_boresight_angle=float(angle),
            ),
        )
        for index, angle in enumerate(angles)
    )
    limits = {"AZ_EL": VisibilityConfig(field_of_view_half_angle=half_angle)}
    filtered = stabilize_observation_opportunities(
        raw, visibility_by_modality=limits,
        temporal_filter_by_modality={
            "AZ_EL": VisibilityTemporalFilterConfig(
                acquisition_epochs=2, loss_epochs=2,
                fov_hysteresis=np.deg2rad(0.2),
            ),
        },
    )

    raw_switches = sum(
        left.visibility.visible != right.visibility.visible
        for left, right in zip(raw, raw[1:])
    )
    filtered_switches = sum(
        left.visibility.visible != right.visibility.visible
        for left, right in zip(filtered, filtered[1:])
    )
    assert raw_switches == 4
    assert filtered_switches == 1
    assert summarize_observation_opportunities(
        filtered
    ).availability_switch_count_by_edge_and_modality[("a", "b", "AZ_EL")] == 1
    assert [item.visibility.visible for item in filtered] == [
        False, False, False, False, False, False, True, True, True,
    ]


def test_fov_temporal_filter_validates_hysteresis_against_fov():
    opportunity = MeasurementOpportunity(
        timestamp=0.0, observer_id="a", target_id="b", modality="AZ_EL",
        visibility=VisibilityResult(
            True, "visible", 1000.0, ALTITUDE, np.deg2rad(1.0)
        ),
    )
    with pytest.raises(ValueError, match="smaller than the FOV"):
        stabilize_observation_opportunities(
            (opportunity,),
            visibility_by_modality={
                "AZ_EL": VisibilityConfig(
                    field_of_view_half_angle=np.deg2rad(5.0)
                ),
            },
            temporal_filter_by_modality={
                "AZ_EL": VisibilityTemporalFilterConfig(
                    fov_hysteresis=np.deg2rad(5.0)
                ),
            },
        )
