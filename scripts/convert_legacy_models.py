#!/usr/bin/env python3
"""
Convert legacy routee-powertrain v1 JSON model files to the v2 directory format.

The old v1 format stored all estimators (one per feature set) in a single JSON
file with base64-encoded model binaries.  The new v2 format stores each
estimator as its own directory containing a ``metadata.json`` and a binary
model file (e.g. ``model.onnx``).

This script converts a single legacy JSON model.  You provide the vehicle
identity (make, model, year, trim, variant) and one v2 model directory is
created for **each** feature-set estimator found in the old file.

Usage
-----
::

    python scripts/convert_legacy_models.py path/to/model.json output_dir/ \\
        --make toyota --model camry --year 2016 --trim 4cyl_2wd

See also ``scripts/convert_nlr_library.py`` for batch-converting the bundled
NLR model library.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes for conversion parameters
# ---------------------------------------------------------------------------


@dataclass
class VehicleIdentity:
    """Vehicle identity fields to stamp into each converted model."""

    make: str
    model: str
    year: int | str
    variant: str = "default"
    fuel_type: Optional[str] = None
    drivetrain: Optional[str] = None
    engine: Optional[str] = None
    trim: Optional[str] = None


# ---------------------------------------------------------------------------
# Estimator type → binary helpers
# ---------------------------------------------------------------------------


def _sanitize_infinities(obj):
    """Recursively replace float('inf') and float('-inf') with None."""
    if isinstance(obj, float) and math.isinf(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_infinities(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_infinities(item) for item in obj]
    return obj


ESTIMATOR_FILE_MAP = {
    "ONNXEstimator": ("model.onnx", "onnx_model"),
    "NGBoostEstimator": ("model.joblib", "ngboost_model"),
    "SmartCoreEstimator": ("model.bin", "smartcore_model"),
    "SKLearnEstimator": ("model.pickle", "rf_regressor"),
}


def _extract_binary(estimator_dict: dict, estimator_type: str) -> bytes:
    """Decode the base64-encoded model binary from the legacy estimator dict."""
    _, key = ESTIMATOR_FILE_MAP[estimator_type]

    raw = estimator_dict.get(key)
    if raw is None:
        raise ValueError(
            f"Expected key '{key}' in estimator dict for type {estimator_type}"
        )

    if isinstance(raw, str):
        return base64.b64decode(raw)
    elif isinstance(raw, dict):
        # SmartCore stores a JSON dict — serialize it to bytes
        return json.dumps(raw).encode("utf-8")
    else:
        raise ValueError(f"Unexpected type for '{key}': {type(raw)}")


# ---------------------------------------------------------------------------
# Core reusable conversion
# ---------------------------------------------------------------------------


def _find_feature_set_for_id(
    feature_sets: List[dict], feature_set_id: str
) -> Optional[dict]:
    """Match a feature_set_id (e.g. 'grade_percent&speed_mph') to the
    corresponding feature set dict from the old config's feature_sets list."""
    for fs in feature_sets:
        names = sorted(f["name"] for f in fs["features"])
        fs_id = "&".join(names)
        if fs_id == feature_set_id:
            return fs
    return None


_DIESEL_NAME_PATTERNS = {"diesel", "tdi", "dci", "328d", "vdi"}


def _infer_fuel_type(
    model_name: str, powertrain_type: str, target_names: List[str]
) -> Optional[str]:
    """Infer fuel type from powertrain type, model name, and target metrics."""
    pt = powertrain_type.upper()
    name = model_name.lower()
    targets = " ".join(t.lower() for t in target_names)

    if pt == "BEV":
        return "ELECTRICITY"
    if pt == "PHEV_EV_MODE":
        return "ELECTRICITY"
    if pt == "PHEV_HEV_MODE":
        return "GASOLINE"

    # Hydrogen fuel cell
    if "kg_h2" in targets or "fuel_cell" in name or name == "mirai":
        return "HYDROGEN"

    # Diesel detection from target metric or model name keywords
    if "gde" in targets:
        return "DIESEL"
    for pattern in _DIESEL_NAME_PATTERNS:
        if pattern in name:
            return "DIESEL"

    if pt == "HEAVY_DUTY":
        return "DIESEL"

    return "GASOLINE"


def convert_legacy_json(
    json_path: Path,
    output_dir: Path,
    identity: VehicleIdentity,
    *,
    version: int = 1,
    schema_version: int = 2,
) -> List[Path]:
    """
    Convert a single legacy v1 JSON model file into v2 model directories.

    Each feature-set estimator in the old model becomes its own v2 model
    directory under *output_dir*.

    Parameters
    ----------
    json_path:
        Path to the legacy ``.json`` model file.
    output_dir:
        Root directory under which model dirs will be created.  The directory
        layout follows the v2 registry convention::

            {output_dir}/v{schema_version}/{make}/{model}/{year}/{trim}/{variant}/{feature_set_id}/v{version}/

    identity:
        Vehicle identification (make / model / year / trim / variant).
    version:
        Model version number (default 1).
    schema_version:
        Schema version to write into metadata (default 2).

    Returns
    -------
    List of created model directory paths.
    """
    with json_path.open("r") as f:
        data = json.load(f)

    old_config: dict = data["metadata"]["config"]
    old_errors: dict = data["errors"]["estimator_errors"]
    all_estimators: dict = data["all_estimators"]
    old_routee_version: str = data["metadata"].get("routee_version", "1.0.0")

    feature_sets: list = old_config["feature_sets"]

    created: List[Path] = []

    for fs_id, est_entry in all_estimators.items():
        estimator_type = est_entry["estimator_constructor_type"]
        estimator_dict = est_entry["estimator"]

        if estimator_type not in ESTIMATOR_FILE_MAP:
            log.warning(
                "Skipping unknown estimator type %r for feature set %r in %s",
                estimator_type,
                fs_id,
                json_path.name,
            )
            continue

        model_filename, _ = ESTIMATOR_FILE_MAP[estimator_type]

        # Extract binary
        model_bytes = _extract_binary(estimator_dict, estimator_type)

        # Find the matching FeatureSet from the old config
        feature_set_dict = _find_feature_set_for_id(feature_sets, fs_id)
        if feature_set_dict is None:
            log.warning(
                "Could not match feature_set_id %r to config feature_sets in %s; "
                "reconstructing from id.",
                fs_id,
                json_path.name,
            )
            # Reconstruct a minimal feature set from the id
            feature_names = fs_id.split("&")
            feature_set_dict = {
                "features": [
                    {
                        "name": name,
                        "units": "unknown",
                        "dtype": "float32",
                        "constraints": {"lower": None, "upper": None},
                    }
                    for name in feature_names
                ]
            }

        # Build new config
        powertrain_type = old_config.get("powertrain_type", "UNDEFINED")
        target_dict = old_config["target"]
        target_names = [t["name"] for t in target_dict.get("targets", [])]

        new_config = {
            "vehicle_description": old_config.get("vehicle_description", ""),
            "powertrain_type": powertrain_type,
            "feature_set": feature_set_dict,
            "distance": old_config["distance"],
            "target": target_dict,
            "make": identity.make,
            "model": identity.model,
            "year": identity.year,
            "predict_method": old_config.get("predict_method", "rate"),
            "test_size": old_config.get("test_size", 0.2),
            "random_seed": old_config.get("random_seed", 42),
            "trip_column": old_config.get("trip_column", "trip_id"),
            "apply_real_world_adjustment": old_config.get(
                "apply_real_world_adjustment", True
            ),
        }

        # Add vehicle attribute fields
        fuel_type = identity.fuel_type
        if fuel_type is None:
            fuel_type = _infer_fuel_type(identity.model, powertrain_type, target_names)
        if fuel_type is not None:
            new_config["fuel_type"] = fuel_type
        if identity.drivetrain is not None:
            new_config["drivetrain"] = identity.drivetrain
        if identity.engine is not None:
            new_config["engine"] = identity.engine
        if identity.trim is not None:
            new_config["trim"] = identity.trim

        # Build errors for this estimator
        estimator_errors_dict = old_errors.get(fs_id)
        if estimator_errors_dict is None:
            log.warning(
                "No errors found for feature_set_id %r in %s; using empty errors.",
                fs_id,
                json_path.name,
            )
            estimator_errors_dict = {
                "feature_set_id": fs_id,
                "error_by_target": {},
            }

        new_errors = {"estimator_errors": estimator_errors_dict}

        # Build metadata.json
        metadata = {
            "schema_version": schema_version,
            "estimator_type": estimator_type,
            "model_file": model_filename,
            "config": new_config,
            "routee_version": old_routee_version,
            "errors": new_errors,
        }

        # Determine output path
        # Sanitize the feature_set_id for use as a directory name
        fs_dir_name = fs_id.replace("&", "_")
        model_dir = (
            output_dir
            / f"v{schema_version}"
            / identity.make
            / identity.model
            / str(identity.year)
            / identity.variant
            / fs_dir_name
            / f"v{version}"
        )

        model_dir.mkdir(parents=True, exist_ok=True)

        # Write metadata.json
        metadata = _sanitize_infinities(metadata)
        metadata_path = model_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

        # Write binary model file
        binary_path = model_dir / model_filename
        binary_path.write_bytes(model_bytes)

        created.append(model_dir)
        log.info("Created model: %s", model_dir)

    return created


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Convert a legacy routee-powertrain v1 JSON model to the v2 format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("json_path", type=Path, help="Path to legacy .json model.")
    parser.add_argument("output_dir", type=Path, help="Root output directory.")
    parser.add_argument("--make", required=True, help="Vehicle make (e.g. toyota).")
    parser.add_argument(
        "--model", required=True, help="Vehicle model (e.g. camry_4cyl_2wd)."
    )
    parser.add_argument("--year", type=int, required=True, help="Model year.")
    parser.add_argument(
        "--variant", default="default", help="Model variant (e.g. charge_depleting)."
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

    args = parser.parse_args()

    identity = VehicleIdentity(
        make=args.make,
        model=args.model,
        year=args.year,
        variant=args.variant,
        fuel_type=args.fuel_type,
        drivetrain=args.drivetrain,
        engine=args.engine,
        trim=args.trim,
    )
    created = convert_legacy_json(
        json_path=args.json_path,
        output_dir=args.output_dir,
        identity=identity,
        version=args.version,
    )
    print(f"Created {len(created)} model(s):")
    for p in created:
        print(f"  {p}")


if __name__ == "__main__":
    main()
