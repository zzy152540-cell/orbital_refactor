"""Dispatch one packaged background module through the application binary."""

from __future__ import annotations

import runpy
import sys


def run_worker_module(arguments: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if arguments is None else arguments)
    if not values:
        raise SystemExit("A worker module is required.")
    module, *module_arguments = values
    if not module.startswith("experiments."):
        raise SystemExit("Only experiments.* worker modules are allowed.")
    sys.argv = [module, *module_arguments]
    runpy.run_module(module, run_name="__main__", alter_sys=False)
    return 0


def main() -> int:
    return run_worker_module()


if __name__ == "__main__":
    raise SystemExit(main())
