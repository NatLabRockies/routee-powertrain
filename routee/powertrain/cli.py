"""
Command line interface for routee-powertrain.

Installed as the ``routee-powertrain`` console script. Currently exposes the v1
to v2 model conversion tool; run ``routee-powertrain --help`` for usage.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from routee.powertrain.io.legacy import MIGRATION_GUIDE_URL, convert_legacy_model

log = logging.getLogger(__name__)


def _add_convert_v1_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "convert-v1",
        help="Convert a legacy v1 .json model file to the v2 archive format.",
        description=(
            "Convert a legacy routee-powertrain v1 .json model to the v2 format. "
            "v1 packed every feature-set estimator into a single file, so one v1 "
            "model usually becomes several v2 models — one per feature set. "
            "v1 files carry no structured vehicle identity, so --make, --model, "
            f"and --year are required. See {MIGRATION_GUIDE_URL}"
        ),
    )
    parser.add_argument("json_path", type=Path, help="Path to legacy .json model.")
    parser.add_argument("output_dir", type=Path, help="Root output directory.")
    parser.add_argument("--make", required=True, help="Vehicle make (e.g. toyota).")
    parser.add_argument(
        "--model",
        required=True,
        help="Vehicle model designation (e.g. camry, golf_1.5tsi). The derived "
        "vehicle_slug is this plus the powertrain family; engine/drivetrain/"
        "trim flags are descriptive metadata only.",
    )
    parser.add_argument("--year", type=int, required=True, help="Model year.")
    parser.add_argument(
        "--variant",
        default="default",
        help=(
            "Model variant (e.g. charge_depleting). Folded into config_slug "
            "as '{rf,ngb,cnn}_{variant}'; not a separate path segment."
        ),
    )
    parser.add_argument("--version", type=int, default=1, help="Model version number.")
    parser.add_argument(
        "--fuel-type",
        default=None,
        help="Fuel type (e.g. GASOLINE, DIESEL, ELECTRICITY, HYDROGEN). "
        "Auto-inferred from powertrain type and model name if not provided.",
    )
    parser.add_argument(
        "--drivetrain",
        default=None,
        help="Drivetrain (e.g. FWD, RWD, AWD, FOURWD).",
    )
    parser.add_argument(
        "--engine", default=None, help="Engine spec (e.g. 4cyl, 2.0tdi, 300kw)."
    )
    parser.add_argument(
        "--trim", default=None, help="Trim level (e.g. sport, le, active)."
    )
    parser.set_defaults(func=_run_convert_v1)


def _run_convert_v1(args: argparse.Namespace) -> int:
    try:
        created = convert_legacy_model(
            args.json_path,
            args.output_dir,
            make=args.make,
            model=args.model,
            year=args.year,
            variant=args.variant,
            fuel_type=args.fuel_type,
            drivetrain=args.drivetrain,
            engine=args.engine,
            trim=args.trim,
            version=args.version,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"Created {len(created)} model(s):")
    for p in created:
        print(f"  {p}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        prog="routee-powertrain",
        description="RouteE-Powertrain command line tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_convert_v1_parser(subparsers)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
