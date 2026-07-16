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

from routee.powertrain.core.digest import stamp_digest
from routee.powertrain.core.metadata import Metadata
from routee.powertrain.core.model import REGISTERED_ESTIMATORS
from routee.powertrain.core.model_config import ModelConfig
from routee.powertrain.registry.model_id import ModelId
from routee.powertrain.validation.errors import ModelErrors

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
}

# Legacy estimator class name -> v2 architecture_tag (matches values emitted by
# routee/powertrain/trainers/*.py). The architecture_tag feeds the derived
# config_slug (via ModelId.from_metadata), so it must match the trainer values.
ARCHITECTURE_TAG_MAP = {
    "ONNXEstimator": "random_forest",  # legacy ONNX exports are all sklearn RFs
    "NGBoostEstimator": "ngboost",
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

            {output_dir}/v{schema_version}/{make}/{vehicle_slug}/{year}/{config_slug}/v{version}/

        Both slugs are derived from the emitted metadata via the package's
        ``ModelId.from_metadata``: the ``vehicle_slug`` as the model name plus
        the coarse powertrain family (e.g. ``camry_ice``), and the
        ``config_slug`` as ``{short_arch}_{variant?}_{feature_hash}``. The
        on-disk path therefore always matches the slugs the registry recomputes
        on load. Distinct feature sets get distinct config slugs automatically;
        the ``variant`` (when not the ``"default"`` sentinel) is stored on the
        config and folded into the config slug.

    identity:
        Vehicle identification. ``model`` should carry the vehicle's commercial
        designation, including whatever distinguishes same-year stablemates
        (e.g. ``golf_1.5tsi`` vs ``golf_2.0tdi``) — the powertrain family is
        the only other field that feeds the derived ``vehicle_slug``.
        ``engine``/``drivetrain``/``trim`` are descriptive, filterable
        metadata, not identity. ``variant`` feeds the derived ``config_slug``;
        the sentinel ``"default"`` is treated as "no variant".
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

    # Treat the "default" placeholder as "no variant" so the common case matches
    # the bundled v2 models (slug ``rf_<hash>`` rather than ``rf_default_<hash>``).
    variant: Optional[str] = identity.variant
    if variant is not None and variant.strip().lower() == "default":
        variant = None

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
        arch_tag = ARCHITECTURE_TAG_MAP[estimator_type]

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
            "variant": variant,
            "test_size": old_config.get("test_size", 0.2),
            "random_seed": old_config.get("random_seed", 42),
            "trip_column": old_config.get("trip_column", "trip_id"),
        }

        # Legacy models stored a boolean flag; map it to the numeric factor.
        # When the flag was False, force no adjustment (1.0); otherwise leave the
        # factor unset so ModelConfig derives it from the powertrain type.
        if old_config.get("apply_real_world_adjustment", True) is False:
            new_config["real_world_adjustment_factor"] = 1.0

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

        # Build errors for this estimator. EstimatorErrors in v2 only holds
        # error_by_target; the old feature_set_id key is stripped.
        estimator_errors_dict = old_errors.get(fs_id)
        if estimator_errors_dict is None:
            log.warning(
                "No errors found for feature_set_id %r in %s; using empty errors.",
                fs_id,
                json_path.name,
            )
            estimator_errors_dict = {"error_by_target": {}}
        else:
            estimator_errors_dict = dict(estimator_errors_dict)
            estimator_errors_dict.pop("feature_set_id", None)

        new_errors = {"estimator_errors": estimator_errors_dict}

        # Build metadata.json. The flat legacy config is decomposed into the
        # grouped v2 sections (vehicle / contract / estimator / training) via
        # Metadata.from_config.
        #
        # Validating through the pydantic ModelConfig + Metadata models keeps the
        # emitted JSON schema-correct (single source of truth). The config_slug —
        # and thus the registry path — is derived from this exact metadata via
        # ModelId.from_metadata, so the on-disk path always matches the slug the
        # registry recomputes (and validates) on load.
        config_obj = ModelConfig.model_validate(new_config)

        # Stamp the self-describing input/output contract onto the estimator so
        # the converted model is required-contract-complete: for ONNX this
        # re-embeds the ordered columns into the binary's metadata_props (and the
        # digest is minted over those embedded bytes). Legacy binaries carry no
        # contract, so we reconstruct the estimator, bind it from the config, and
        # re-serialize. Only the estimator mechanics (lookback/grouping/pad) go
        # into metadata.json's input_spec; the ordered columns live once in the
        # ``contract`` section (and embedded in the binary above).
        estimator_cls = REGISTERED_ESTIMATORS[estimator_type]
        estimator = estimator_cls.from_bytes(model_bytes)
        estimator.bind_io_contract(config_obj)
        model_bytes = estimator.to_bytes()
        input_spec_dict = estimator.input_spec.model_dump(
            mode="json", include={"lookback", "grouping_column", "pad_strategy"}
        )

        metadata_obj = Metadata.from_config(
            config_obj,
            errors=ModelErrors.model_validate(new_errors),
            estimator_type=estimator_type,
            model_file=model_filename,
            architecture_tag=arch_tag,
            input_spec=input_spec_dict,
            routee_version=old_routee_version,
            # Legacy v1 archives don't record when the model was trained, so we
            # leave trained_date null rather than guessing.
            trained_date=None,
        )
        # Mint the instance identity (estimator_sha256 + model_digest) so the
        # converted library needs no separate digest backfill pass. The digest
        # payload excludes the error metrics, so the infinity-sanitization of
        # errors below does not invalidate it.
        stamp_digest(metadata_obj, model_bytes)
        model_id = ModelId.from_metadata(metadata_obj, version)

        model_dir = output_dir / f"v{schema_version}" / model_id.to_path()
        model_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize any infinite error metrics to null before writing.
        metadata_out = _sanitize_infinities(metadata_obj.model_dump(mode="json"))
        (model_dir / "metadata.json").write_text(json.dumps(metadata_out, indent=2))

        # Write binary model file
        (model_dir / model_filename).write_bytes(model_bytes)

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
