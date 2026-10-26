from experiments.run_bilateral_dynamic_topology_prescan import (
    build_dynamic_topology_schedule,
    run_geometry_scan,
    run_prescan,
)
from orbital_core.constants import R_EARTH
from scenarios.bilateral_multitarget import generate_bilateral_walker_truth
from scenarios.walker_scenario import WalkerDeltaConfig
import numpy as np


def test_dynamic_topology_prescan_is_supplemental_and_reports_ring_visibility():
    result = run_prescan(
        duration=2.0, dt=1.0, maximum_ranges_km=(7000.0,),
    )

    assert result["affectsAcceptanceConfiguration"] is False
    assert result["friendlySatelliteCount"] == 15
    assert result["fixedRing"]["undirectedEdgeCount"] == 15
    scan = result["rangeScans"][0]
    assert scan["epochCount"] == 3
    assert 0.0 <= scan["connectedEpochFraction"] <= 1.0
    assert 0.0 <= scan["ringEdgeVisibilityFraction"] <= 1.0
    assert scan["maximumCandidateNodeDegree"] <= 14


def test_geometry_scan_covers_every_valid_fifteen_satellite_walker_choice():
    rows = run_geometry_scan(
        duration=0.5, dt=0.5, maximum_range_km=10000.0,
    )

    assert len(rows) == 24
    assert {(row["planeCount"], row["phasing"]) for row in rows} == {
        (planes, phasing)
        for planes in (1, 3, 5, 15)
        for phasing in range(planes)
    }


def test_selected_candidate_builds_a_complete_degree_bounded_schedule():
    truth = generate_bilateral_walker_truth(
        duration_seconds=2.0,
        sample_interval_seconds=1.0,
        friendly_config=WalkerDeltaConfig(
            total_satellites=15,
            plane_count=15,
            phasing=1,
            semi_major_axis=R_EARTH + 700e3,
            eccentricity=0.0,
            inclination=np.deg2rad(53.0),
            node_prefix="friendly",
        ),
    )

    schedule, summary = build_dynamic_topology_schedule(truth)

    assert set(schedule) == {0.0, 1.0, 2.0}
    assert summary["edgeCountMin"] == summary["edgeCountMax"] == 14
    assert all(
        max(map(len, topology.adjacency.values())) <= 3
        for topology in schedule.values()
    )

    failed_ids = truth.manifest.friendly_observer_ids[:3]
    fault_schedule, fault_summary = build_dynamic_topology_schedule(
        truth,
        inactive_node_ids_by_timestamp={1.0: failed_ids},
    )
    active_ids = set(truth.manifest.friendly_observer_ids) - set(failed_ids)
    active_edges = {
        tuple(sorted((node_id, neighbor_id)))
        for node_id in active_ids
        for neighbor_id in fault_schedule[1.0].neighbors(node_id)
    }
    assert len(active_edges) == len(active_ids) - 1
    assert all(not fault_schedule[1.0].neighbors(node_id) for node_id in failed_ids)
    assert fault_summary["containsFailureAwareReselection"] is True
