from experiments.angle_tracker_calibration import (
    calibrate_angle_trackers, write_angle_tracker_calibration,
)


def main():
    result = calibrate_angle_trackers()
    print(result["best_pll"])
    print(result["best_circular_kalman"])
    print(write_angle_tracker_calibration(
        result, "results/cann/angle_tracker_calibration",
    ))


if __name__ == "__main__":
    main()
