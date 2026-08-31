"""
Command line interface for routee-powertrain.

Installed as the ``routee-powertrain`` console script. Exposes the v1 to v2
model conversion tool and the physical-validation report; run
``routee-powertrain --help`` for usage.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from routee.powertrain.io.legacy import MIGRATION_GUIDE_URL, convert_legacy_model
from routee.powertrain.validation.physics import (
    PhysicsAssumptions,
    PhysicsReport,
    check_model,
)

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


# ---------------------------------------------------------------------------
# validate-physics
# ---------------------------------------------------------------------------


def _add_validate_physics_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "validate-physics",
        help="Check trained models against physical law and report the result.",
        description=(
            "Run a synthetic sweep of links through one or more models and "
            "report which physical predicates hold, plus diagnostics such as "
            "implied drivetrain and regeneration efficiency and flat-ground "
            "economy. Needs no ground-truth data. Models are scored before the "
            "real-world adjustment factor, so the report describes the function "
            "that was learned. Checks needing vehicle mass are skipped, and say "
            "so, when the metadata carries none."
        ),
    )
    parser.add_argument(
        "models",
        nargs="*",
        help="Model paths (directory, .zip, .tar.gz) or registry ids. "
        "Omit when using --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate every model in the registry, subject to the filters below.",
    )
    parser.add_argument("--make", default=None, help="Filter --all by make.")
    parser.add_argument("--model-name", default=None, help="Filter --all by model.")
    parser.add_argument("--year", type=int, default=None, help="Filter --all by year.")
    parser.add_argument(
        "--powertrain-type", default=None, help="Filter --all by powertrain type."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Stop after this many models."
    )
    parser.add_argument(
        "--mass-lbs",
        type=float,
        default=None,
        help="Vehicle mass to assume for models whose metadata carries none. "
        "Enables the mass-dependent checks; recorded in the report as an "
        "assumption rather than a fact.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write the full reports to a JSON file.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write one summary row per model to a CSV file.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print just the cross-model summary table.",
    )
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="Exit non-zero when any model fails a check. Off by default, so "
        "this reports rather than gates.",
    )
    parser.set_defaults(func=_run_validate_physics)


def _resolve_targets(args: argparse.Namespace) -> list:
    """Work out which models to validate, as a list of identifiers."""
    from routee.powertrain.io.load import list_available_models, query_available_models

    targets: list = list(args.models)
    if args.all:
        filters = {
            k: v
            for k, v in (
                ("make", args.make),
                ("model", args.model_name),
                ("year", args.year),
                ("powertrain_type", args.powertrain_type),
            )
            if v is not None
        }
        if filters:
            targets.extend(
                str(info.model_id) for info in query_available_models(**filters)
            )
        else:
            targets.extend(str(model_id) for model_id in list_available_models())
    if args.limit is not None:
        targets = targets[: args.limit]
    return targets


def _summary_row(name: str, key: str, report: PhysicsReport) -> dict:
    row: dict = {
        "model": name,
        "model_key": key,
        "target": report.target,
        "passed": report.passed,
        "mass_source": report.mass_source,
    }
    for check in report.checks:
        row[check.name] = (
            ""
            if check.status == "not_applicable"
            else round(check.violation_rate or 0.0, 4)
        )
    d = report.diagnostics
    row["implied_eta_drive"] = d.implied_eta_drive
    row["implied_eta_regen"] = d.implied_eta_regen
    row["length_invariance"] = d.length_invariance
    return row


def _run_validate_physics(args: argparse.Namespace) -> int:
    import csv as csv_module
    import json as json_module

    from routee.powertrain.io.load import load_model

    if not args.models and not args.all:
        print(
            "error: give at least one model, or --all to sweep the registry",
            file=sys.stderr,
        )
        return 1

    targets = _resolve_targets(args)
    if not targets:
        print("error: no models matched", file=sys.stderr)
        return 1

    assumptions = PhysicsAssumptions()
    reports: dict = {}
    labels: dict = {}
    unloadable: list = []

    for name in targets:
        try:
            model = load_model(name)
        except Exception as e:
            # One unreachable model must not abandon the rest of the sweep.
            unloadable.append((name, f"{type(e).__name__}: {e}"))
            continue
        report = check_model(model, assumptions=assumptions, mass_lbs=args.mass_lbs)
        reports[str(name)] = report
        # A local path can be long and says nothing about which vehicle it is;
        # the model key identifies it compactly however it was addressed.
        labels[str(name)] = model.key.to_path()
        if not args.summary_only:
            print(f"\n#### {name}")
            print(repr(report))

    if reports:
        print("\n" + _format_summary(reports, labels))
    for name, reason in unloadable:
        print(f"could not load {name}: {reason}", file=sys.stderr)

    if args.json:
        args.json.write_text(
            json_module.dumps(
                {k: v.model_dump(mode="json") for k, v in reports.items()}, indent=2
            )
        )
        print(f"wrote {args.json}")
    if args.csv and reports:
        rows = [_summary_row(k, labels.get(k, k), v) for k, v in reports.items()]
        fieldnames: list = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with args.csv.open("w", newline="") as fh:
            writer = csv_module.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")

    if unloadable and not reports:
        return 1
    if args.fail_on_violation and any(not r.passed for r in reports.values()):
        return 2
    return 0


def _format_summary(reports: dict, labels: dict) -> str:
    """A one-row-per-model table of violation rates, checks as columns."""
    check_names: list = []
    for report in reports.values():
        for check in report.checks:
            if check.name not in check_names:
                check_names.append(check.name)

    def abbreviate(name: str) -> str:
        return "".join(part[0] for part in name.split("_")).upper()

    name_width = max([len(labels.get(n, n)) for n in reports] + [5])
    header = f"{'model':<{name_width}}  " + "  ".join(
        f"{abbreviate(n):>5}" for n in check_names
    )
    lines = [header, "-" * len(header)]
    for name, report in reports.items():
        cells = []
        for check_name in check_names:
            check = report.check_map.get(check_name)
            if check is None or check.status == "not_applicable":
                cells.append(f"{'--':>5}")
            else:
                cells.append(f"{100 * (check.violation_rate or 0.0):>4.0f}%")
        lines.append(f"{labels.get(name, name):<{name_width}}  " + "  ".join(cells))
    lines.append("-" * len(header))
    lines.append("  ".join(f"{abbreviate(n)}={n}" for n in check_names))
    failed = sum(1 for r in reports.values() if not r.passed)
    lines.append(
        f"\n{len(reports)} model(s) checked, {failed} with at least one violation."
    )
    lines.append("'--' means the check did not apply (no mass, or no such feature).")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        prog="routee-powertrain",
        description="RouteE-Powertrain command line tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_convert_v1_parser(subparsers)
    _add_validate_physics_parser(subparsers)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
