from experiments.line_cann_decoder_benchmark import (
    run_line_cann_decoder_benchmark, write_line_cann_decoder_benchmark,
)


def main():
    result = run_line_cann_decoder_benchmark()
    print(result["summary"])
    print(write_line_cann_decoder_benchmark(
        result, "results/cann/line_cann_decoder_benchmark",
    ))


if __name__ == "__main__":
    main()
