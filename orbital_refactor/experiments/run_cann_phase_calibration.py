from pathlib import Path

from experiments.cann_phase_calibration import (
    run_cann_phase_calibration, write_cann_phase_calibration,
)


def main():
    result = run_cann_phase_calibration()
    paths = write_cann_phase_calibration(
        result, Path("results/cann/phase_calibration"),
    )
    print(result["delay_fit"])
    print({key: str(value) for key, value in paths.items()})


if __name__ == "__main__":
    main()
