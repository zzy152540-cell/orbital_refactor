from experiments.coupled_ring_line_evaluation import (
    evaluate_coupled_ring_line, write_coupled_ring_line_evaluation,
)


def main():
    result = evaluate_coupled_ring_line(
        seeds=range(5), bias_anchor_mode="hybrid_dual",
        minimum_bias_baseline=120.0, line_cue_gain=0.2,
        anchor_agreement_scale_deg_s=0.004,
    )
    print(result["summary"])
    print(write_coupled_ring_line_evaluation(
        result, "results/cann/coupled_ring_line_evaluation_hybrid_candidate",
    ))


if __name__ == "__main__":
    main()
