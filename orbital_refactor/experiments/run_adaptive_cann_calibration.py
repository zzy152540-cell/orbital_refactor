from experiments.adaptive_cann_calibration import (
    calibrate_adaptive_cann,
    write_adaptive_cann_calibration,
)


def main():
    result = calibrate_adaptive_cann()
    print(write_adaptive_cann_calibration(
        result, "results/cann/adaptive_cann_calibration",
    ))
    print(result["summaries"])
    print("best", result["best"])


if __name__ == "__main__":
    main()
