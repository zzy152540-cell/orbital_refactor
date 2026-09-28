"""Generate display telemetry aligned with a prepared trajectory data set."""

from __future__ import annotations

import argparse

from exporters.display_telemetry import export_display_telemetry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--functional-truth", required=True)
    parser.add_argument("--background-truth", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--selected-satellite", default=None,
        help="Deprecated compatibility option; all satellites are exported.",
    )
    parser.add_argument("--without-cann", action="store_true")
    args = parser.parse_args()
    path = export_display_telemetry(
        args.trajectory,
        args.functional_truth,
        args.background_truth,
        args.output,
        selected_satellite=args.selected_satellite,
        include_cann=not args.without_cann,
    )
    print(path)


if __name__ == "__main__":
    main()
