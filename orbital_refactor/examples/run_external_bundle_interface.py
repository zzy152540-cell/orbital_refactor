"""Minimal command-line integration using only the public bundle interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from interfaces.interface_contracts import InterfaceValidationError
from interfaces.public_api import get_public_api_info, run_module_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", nargs="?", type=Path, help="ModuleInput bundle directory")
    parser.add_argument("response", nargs="?", type=Path, help="new ModuleOutput directory")
    parser.add_argument(
        "--describe", action="store_true", help="print public protocol versions and exit"
    )
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(get_public_api_info(), ensure_ascii=False, indent=2))
        return 0
    if args.request is None or args.response is None:
        parser.error("request and response directories are required unless --describe is used")
    try:
        destination = run_module_bundle(args.request, args.response)
    except InterfaceValidationError as exc:
        print(json.dumps({"ok": False, "error": exc.to_dict()}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, "response": str(destination)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
