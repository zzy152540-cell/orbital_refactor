from experiments.single_modality_observability_audit import (
    run_single_modality_observability_audit,
)


def test_short_observability_audit_exposes_direction_sensor_overlap():
    report = run_single_modality_observability_audit(
        duration=4.0, dt=2.0, infrared_noise_scales=(1.0,),
    )
    records = {record.modalities: record for record in report.records}

    assert len(records) == 7
    assert report.optical_infrared_row_space_overlap_minimum > 0.999
    assert records["opt+ir"].instantaneous_rank_max == 2
    assert records["opt+rad"].instantaneous_rank_min >= 3
    assert records["opt+ir+rad"].horizon_rank >= records["opt+ir"].horizon_rank
    assert all(record.horizon_condition_number > 0.0 for record in records.values())
