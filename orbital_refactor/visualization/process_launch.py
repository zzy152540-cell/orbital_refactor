"""Resolve source and frozen-process commands for background jobs."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Iterable


WORKER_EXECUTABLE_ENV = "ORBITAL_WORKER_EXECUTABLE"


def module_process_command(
    module: str, arguments: Iterable[str],
) -> tuple[str, list[str]]:
    """Return a command that works from source and from a frozen GUI."""

    module = str(module).strip()
    if not module.startswith("experiments."):
        raise ValueError("Background worker modules must belong to experiments.")
    worker = os.environ.get(WORKER_EXECUTABLE_ENV, "").strip()
    values = [str(value) for value in arguments]
    if worker:
        return str(Path(worker)), [module, *values]
    if getattr(sys, "frozen", False):
        return sys.executable, ["--worker-module", module, *values]
    return sys.executable, ["-m", module, *values]
