"""Dispatch one packaged background module through the application binary."""

from __future__ import annotations

import importlib
import sys


PACKAGED_WORKER_MODULES = (
    "experiments.run_bilateral_acceptance_control",
    "experiments.run_bilateral_baseline_matrix",
    "experiments.run_bilateral_dropout_prescan",
    "experiments.run_bilateral_node_failure_prescan",
    "experiments.run_bilateral_visualization_recording",
    "experiments.run_interface_conformance_acceptance",
    "experiments.run_multitarget_distributed_node_sync_latency_acceptance",
    "experiments.run_ubuntu_environment_acceptance",
)


def run_worker_module(arguments: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if arguments is None else arguments)
    if not values:
        raise SystemExit("A worker module is required.")
    module, *module_arguments = values
    if module not in PACKAGED_WORKER_MODULES:
        raise SystemExit(f"Unsupported packaged worker module: {module}")
    sys.argv = [module, *module_arguments]
    worker = importlib.import_module(module)
    entrypoint = getattr(worker, "main", None)
    if entrypoint is None:
        raise SystemExit(f"Packaged worker has no main() entrypoint: {module}")
    result = entrypoint()
    return int(result or 0)


def main() -> int:
    return run_worker_module()


if __name__ == "__main__":
    raise SystemExit(main())
