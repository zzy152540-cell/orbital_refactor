from experiments.coupled_ring_line_robustness import (
    evaluate_coupled_ring_line_robustness,
    write_coupled_ring_line_robustness,
)


def main():
    result = evaluate_coupled_ring_line_robustness(
        seeds=range(5), initial_offsets_deg=(-2.0, -1.0, 1.0, 2.0),
        bias_anchor_mode="hybrid_dual",
        minimum_bias_baseline=120.0, line_cue_gain=0.2,
        anchor_agreement_scale_deg_s=0.004,
    )
    print(result["summary"])
    print(write_coupled_ring_line_robustness(
        result, "results/cann/coupled_ring_line_robustness_hybrid_candidate",
    ))


if __name__ == "__main__":
    main()
