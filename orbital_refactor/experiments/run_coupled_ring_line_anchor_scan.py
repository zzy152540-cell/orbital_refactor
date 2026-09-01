from experiments.coupled_ring_line_anchor_scan import (
    scan_rolling_anchor_parameters, write_rolling_anchor_parameter_scan,
)


def main():
    result = scan_rolling_anchor_parameters()
    print(result["best"])
    print(write_rolling_anchor_parameter_scan(
        result, "results/cann/coupled_ring_line_anchor_scan",
    ))


if __name__ == "__main__":
    main()
