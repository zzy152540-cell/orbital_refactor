from pathlib import Path

from experiments.cann_phase_projection_feedback import (
    run_cann_phase_projection_feedback, run_cann_phase_projection_gain_sweep,
    write_feedback_results, write_gain_sweep_results,
)


def main():
    result = run_cann_phase_projection_feedback()
    paths = write_feedback_results(result, Path("results/cann/phase_projection_feedback"))
    print(result["summary"]); print({key: str(value) for key, value in paths.items()})
    sweep = run_cann_phase_projection_gain_sweep()
    sweep_paths = write_gain_sweep_results(
        sweep, Path("results/cann/phase_projection_feedback"),
    )
    print(sweep["rows"]); print({key: str(value) for key, value in sweep_paths.items()})


if __name__ == "__main__":
    main()
