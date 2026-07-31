"""
Convert legacy routee-powertrain v1 JSON model files to the v2 archive format.

The v1 format stored every estimator (one per feature set) in a single JSON file
with base64-encoded model binaries. The v2 format stores each estimator as its
own directory containing a ``metadata.json`` and a binary model file (e.g.
``model.onnx``). Because v1 packed several feature sets into one file, converting
a single v1 model produces **several** v2 models — one per feature set.

The v1 format also had no structured vehicle identity (only a free-text
``vehicle_description``), so ``make``, ``model``, and ``year`` must be supplied by
the caller; there is nothing to infer them from.

Examples:

    >>> import routee.powertrain as pt
    >>> paths = pt.convert_legacy_model(
    ...     "MyModel.json", "out/", make="toyota", model="camry", year=2016
    ... )

or from the command line::

    routee-powertrain convert-v1 MyModel.json out/ \\
        --make toyota --model camry --year 2016 --trim 4cyl_2wd
"""

from __future__ import annotations

import base64
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from routee.powertrain.__about__ import MIGRATION_GUIDE_URL
from routee.powertrain.core.metadata import Metadata
from routee.powertrain.core.model_config import ModelConfig
from routee.powertrain.core.provenance import LegacySource, TrainingSource
from routee.powertrain.io.archive import save_model_directory
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

# Estimator types that existed in v1 but have no v2 equivalent. Kept separate
# from "unrecognized" so the failure message can say *why* rather than just
# listing an unknown name.
_REMOVED_ESTIMATORS = {
    "SmartCoreEstimator": (
        "the smartcore/Rust estimator was removed in v2; retrain the model with "
        "SklearnRandomForestTrainer, which exports to ONNX"
    ),
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


def _looks_like_legacy_model_json(data: object) -> bool:
    """True if a parsed JSON document has the shape of a v1 model file."""
    return isinstance(data, dict) and "all_estimators" in data and "metadata" in data


def convert_legacy_json(
    json_path: Union[str, Path],
    output_dir: Union[str, Path],
    identity: VehicleIdentity,
    *,
    provenance_source: Optional[TrainingSource] = None,
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
    provenance_source:
        What produced the training data, recorded in the ``provenance``
        section. Defaults to ``LegacySource(converted_from="v1")`` — v1
        archives record no simulator, pipeline, or dataset information, so
        "converted from v1, origin unknown" is the honest answer. Pass a
        populated source when the caller knows more.
    version:
        Model version number (default 1).
    schema_version:
        Schema version to write into metadata (default 2).

    Returns
    -------
    List of created model directory paths.

    Raises
    ------
    ValueError
        If the file is not a v1 model, or if none of its estimators could be
        converted (e.g. a model saved with the removed smartcore estimator).
    """
    # Imported here rather than at module scope: core.model imports io.archive
    # at import time, so a module-level import would close an import cycle.
    from routee.powertrain.core.model import REGISTERED_ESTIMATORS, Model

    json_path = Path(json_path)
    output_dir = Path(output_dir)

    if provenance_source is None:
        provenance_source = LegacySource(converted_from="v1")

    with json_path.open("r") as f:
        data = json.load(f)

    if not _looks_like_legacy_model_json(data):
        raise ValueError(
            f"{json_path} does not look like a routee-powertrain v1 model file "
            "(expected top-level 'metadata' and 'all_estimators' keys). v2 models "
            "are directories, .zip, or .tar.gz archives — load those with "
            "pt.load_model() directly."
        )

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
    skipped: dict[str, str] = {}

    for fs_id, est_entry in all_estimators.items():
        estimator_type = est_entry["estimator_constructor_type"]
        estimator_dict = est_entry["estimator"]

        if estimator_type not in ESTIMATOR_FILE_MAP:
            reason = _REMOVED_ESTIMATORS.get(
                estimator_type, "this estimator type is not supported in v2"
            )
            skipped[estimator_type] = reason
            log.warning(
                "Skipping estimator type %r for feature set %r in %s: %s",
                estimator_type,
                fs_id,
                json_path.name,
                reason,
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
            "training_source": provenance_source,
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
        #
        # Legacy feature/distance constraints use ±inf to mean "unbounded"; v2
        # records unbounded as null. Sanitize before validating so no infinities
        # survive into the serialized metadata — save_model_directory writes with
        # json.dumps, which would otherwise emit invalid `Infinity` tokens.
        # Constraint bounds are not part of the digest payload, so this does not
        # affect model_digest.
        config_obj = ModelConfig.model_validate(_sanitize_infinities(new_config))

        # Stamp the self-describing input/output contract onto the estimator so
        # the converted model is required-contract-complete: for ONNX this
        # re-embeds the ordered columns into the binary's metadata_props (and the
        # digest is minted over those embedded bytes). Legacy binaries carry no
        # contract, so we reconstruct the estimator and bind it from the config;
        # the re-serialization happens inside save_model_directory (below), which
        # calls estimator.to_bytes(). Only the estimator mechanics
        # (lookback/grouping/pad) go into metadata.json's input_spec; the ordered
        # columns live once in the ``contract`` section (and embedded in the
        # binary).
        estimator_cls = REGISTERED_ESTIMATORS[estimator_type]
        estimator = estimator_cls.from_bytes(model_bytes)
        estimator.bind_io_contract(config_obj)
        input_spec_dict = estimator.input_spec.model_dump(
            mode="json", include={"lookback", "grouping_column", "pad_strategy"}
        )

        metadata_obj = Metadata.from_config(
            config_obj,
            # Sanitize any infinite error metrics to null before they reach the
            # metadata; save_model_directory serializes with json.dumps, which
            # would otherwise emit invalid `Infinity` tokens. Errors are excluded
            # from the digest payload, so this does not affect model_digest.
            errors=ModelErrors.model_validate(_sanitize_infinities(new_errors)),
            estimator_type=estimator_type,
            model_file=model_filename,
            architecture_tag=arch_tag,
            input_spec=input_spec_dict,
            routee_version=old_routee_version,
            # Legacy v1 archives don't record when the model was trained, so we
            # leave trained_date null rather than guessing.
            trained_date=None,
        )
        model_id = ModelId.from_metadata(metadata_obj, version)
        model_dir = output_dir / f"v{schema_version}" / model_id.to_path()

        # Persist through the Model umbrella rather than writing the binary and
        # metadata by hand. save_model_directory is the single save choke point:
        # it enforces the required input/output contract and mints the instance
        # identity (estimator_sha256 + model_digest) via _ensure_digest, so the
        # converted library needs no separate digest backfill pass and this
        # module can't drift from the invariants the archive layer enforces.
        model = Model(estimator, metadata_obj)
        save_model_directory(model, model_dir)

        created.append(model_dir)
        log.info("Created model: %s", model_dir)

    if not created:
        detail = "; ".join(f"{name} ({why})" for name, why in sorted(skipped.items()))
        raise ValueError(
            f"Could not convert any estimator from {json_path}. "
            f"Unconvertible estimator types: {detail or 'none found'}. "
            f"See {MIGRATION_GUIDE_URL}"
        )

    return created


def convert_legacy_model(
    json_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    make: str,
    model: str,
    year: Union[int, str],
    variant: str = "default",
    fuel_type: Optional[str] = None,
    drivetrain: Optional[str] = None,
    engine: Optional[str] = None,
    trim: Optional[str] = None,
    version: int = 1,
) -> List[Path]:
    """
    Convert a legacy v1 ``.json`` model file into one or more v2 model directories.

    v1 packed every feature-set estimator into a single file, so one v1 model
    typically converts to several v2 models — one per feature set. Load each with
    ``pt.load_model(path)``.

    Args:
        json_path: path to the legacy ``.json`` model file
        output_dir: root directory to write converted models under
        make: vehicle make (e.g. ``"toyota"``)
        model: vehicle model designation (e.g. ``"camry"``, ``"golf_1.5tsi"``).
            The derived ``vehicle_slug`` is this plus the powertrain family.
        year: model year
        variant: short variant label folded into the derived ``config_slug``
            (e.g. ``"charge_depleting"``). ``"default"`` means no variant.
        fuel_type: e.g. ``"GASOLINE"``; inferred from powertrain type and model
            name when omitted
        drivetrain: e.g. ``"FWD"``, ``"AWD"``
        engine: e.g. ``"4cyl"``, ``"2.0tdi"``
        trim: e.g. ``"le"``, ``"sport"``
        version: registry version number to stamp (default 1)

    Returns:
        the list of created model directories

    Examples:
        >>> import routee.powertrain as pt
        >>> paths = pt.convert_legacy_model(
        ...     "MyModel.json", "out/", make="toyota", model="camry", year=2016
        ... )
        >>> models = [pt.load_model(p) for p in paths]
    """
    identity = VehicleIdentity(
        make=make,
        model=model,
        year=year,
        variant=variant,
        fuel_type=fuel_type,
        drivetrain=drivetrain,
        engine=engine,
        trim=trim,
    )
    return convert_legacy_json(
        json_path=json_path,
        output_dir=output_dir,
        identity=identity,
        version=version,
    )
