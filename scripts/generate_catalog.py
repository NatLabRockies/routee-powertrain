#!/usr/bin/env python
"""
Generate a catalog.json index from a directory tree of model .zip archives.

The directory tree should follow the registry path convention:
    <root>/<schema_version>/<make>/<model>/<year>/<trim>/<variant>/v<N>.zip

Usage:
    python scripts/generate_catalog.py /path/to/registry/root
    python scripts/generate_catalog.py /path/to/registry/root --schema-version v2
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from routee.powertrain.io.archive import read_archive_metadata
from routee.powertrain.registry.catalog import Catalog
from routee.powertrain.registry.model_id import ModelId, ModelInfo

# Pattern to extract version number from filename like v1.zip, v2.zip
VERSION_RE = re.compile(r"^v(\d+)\.zip$")


def parse_model_id_from_path(zip_path: Path, schema_root: Path) -> ModelId:
    """
    Derive a ModelId from the filesystem path relative to the schema root.

    Expected structure: <make>/<model>/<year>/<trim>/<variant>/v<N>.zip
    """
    rel = zip_path.relative_to(schema_root)
    parts = list(rel.parts)

    if len(parts) != 6:
        raise ValueError(
            f"Unexpected path depth for {zip_path}. "
            f"Expected <make>/<model>/<year>/<trim>/<variant>/v<N>.zip, "
            f"got {'/'.join(parts)}"
        )

    make, model_name, year_str, trim, variant, filename = parts

    match = VERSION_RE.match(filename)
    if not match:
        raise ValueError(
            f"Filename '{filename}' does not match expected pattern v<N>.zip"
        )

    return ModelId(
        make=make,
        model_name=model_name,
        year=int(year_str),
        trim=trim,
        variant=variant,
        version=int(match.group(1)),
    )


def build_model_info(zip_path: Path, schema_root: Path) -> ModelInfo:
    """Read archive metadata and convert it to a ModelInfo catalog entry."""
    model_id = parse_model_id_from_path(zip_path, schema_root)
    meta = read_archive_metadata(zip_path)

    config = meta["metadata"]["config"]

    # Extract error summaries
    est_errors = meta["errors"]["estimator_errors"]
    error_summary = {}
    for target_name, target_errors in est_errors["error_by_target"].items():
        error_summary[target_name] = {
            k: v for k, v in target_errors.items() if v is not None
        }

    # feature names
    feature_names = [f["name"] for f in config["feature_set"]["features"]]
    target_names = [t["name"] for t in config["target"]["targets"]]

    rel_path = str(zip_path.relative_to(schema_root.parent))

    return ModelInfo(
        model_id=model_id,
        estimator_type=meta["estimator_type"],
        feature_names=feature_names,
        target_names=target_names,
        powertrain_type=config["powertrain_type"],
        errors=error_summary,
        vehicle_description=config["vehicle_description"],
        path=rel_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate catalog.json from a model directory tree"
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory of the model registry",
    )
    parser.add_argument(
        "--schema-version",
        default="v2",
        help="Schema version subdirectory (default: v2)",
    )
    args = parser.parse_args()

    root: Path = args.root
    schema_version: str = args.schema_version
    schema_root = root / schema_version

    if not schema_root.exists():
        print(f"Schema root not found: {schema_root}")
        sys.exit(1)

    zip_files = sorted(schema_root.glob("**/v*.zip"))
    if not zip_files:
        print(f"No model archives found under {schema_root}")
        sys.exit(1)

    models = []
    for zf in zip_files:
        try:
            info = build_model_info(zf, schema_root)
            models.append(info)
            print(f"  indexed: {info.model_id}")
        except Exception as e:
            print(f"  SKIPPED {zf}: {e}")

    catalog = Catalog(schema_version=schema_version, models=models)
    out_path = schema_root / "catalog.json"
    catalog.to_json(out_path)
    print(f"\nWrote catalog with {len(models)} models to {out_path}")


if __name__ == "__main__":
    main()
