from experiments.external_cooperative_node_ablation import (
    run_external_cooperative_node_ablation,
)


def test_short_targeted_ablation_covers_edges_and_modalities():
    report = run_external_cooperative_node_ablation(
        seeds=(0,), duration=4.0, dt=2.0,
    )

    assert report.node_id == "sat_p03_s01"
    assert report.run_count == 1
    variants = {record.variant for record in report.records}
    assert variants == {
        "full", "without_sat_p02_s01", "without_sat_p04_s01",
        "radar_only", "infrared_only", "optical_only",
    }
