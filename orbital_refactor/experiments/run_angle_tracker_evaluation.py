from experiments.angle_tracker_evaluation import (
    evaluate_frozen_angle_trackers,
    write_frozen_angle_tracker_evaluation,
)


def main():
    result = evaluate_frozen_angle_trackers()
    print(write_frozen_angle_tracker_evaluation(
        result, "results/cann/angle_tracker_evaluation",
    ))
    for method, metrics in result["summary"].items():
        print(method, metrics)


if __name__ == "__main__":
    main()
