from __future__ import annotations

import re
from typing import Sequence

from routee.powertrain.core.year import parse_year
from routee.powertrain.io.archive import METADATA_FILENAME
from routee.powertrain.registry.model_id import ModelId, ModelInfo

# Pattern to extract the version from a path segment like "v1", "v2".
VERSION_RE = re.compile(r"^v(\d+)$")

#: The canonical registry entry layout, shared by every backend:
#: ``<make>/<vehicle_slug>/<year>/<config_slug>/v<N>/``
ENTRY_SEGMENTS = "<make>/<vehicle_slug>/<year>/<config_slug>/v<N>"


def parse_model_id_from_segments(segments: Sequence[str], source: str) -> ModelId:
    """
    Build a ModelId from the five path segments of a registry entry.

    This is the one place the registry layout is decoded, shared by every
    backend so a path means the same thing on disk, in S3, and on the Hub.

    Args:
        segments: the five segments
            ``<make>/<vehicle_slug>/<year>/<config_slug>/v<N>``
        source: the original path or key, used only in error messages

    Returns: a ModelId

    Raises:
        ValueError: if the segment count or the version segment is wrong
    """
    if len(segments) != 5:
        raise ValueError(
            f"Unexpected registry entry structure: {source}. "
            f"Expected {ENTRY_SEGMENTS}, got {len(segments)} segments."
        )

    make, vehicle_slug, year_str, config_slug, version_dir = segments

    match = VERSION_RE.match(version_dir)
    if not match:
        raise ValueError(
            f"Version segment '{version_dir}' does not match expected pattern v<N>"
        )

    return ModelId(
        make=make,
        vehicle_slug=vehicle_slug,
        year=parse_year(year_str),
        config_slug=config_slug,
        version=int(match.group(1)),
    )


def parse_model_id_from_metadata_key(key: str, prefix: str) -> ModelId:
    """
    Derive a ModelId from the key of an entry's ``metadata.json``.

    Used by the object-store backends (S3, HuggingFace Hub), whose listings
    return flat keys rather than directories.

    Expected key format::

        [<prefix>/]<make>/<vehicle_slug>/<year>/<config_slug>/v<N>/metadata.json

    Args:
        key: the full storage key
        prefix: the already-composed prefix the key is expected to start with
            (e.g. ``"routee-powertrain-model-library/v2"``); pass ``""`` when
            entries live at the storage root

    Returns: a ModelId

    Raises:
        ValueError: if the key does not start with ``prefix`` or does not
            match the expected structure
    """
    full_prefix = f"{prefix}/" if prefix else ""
    if not key.startswith(full_prefix):
        raise ValueError(f"Key {key} does not start with {full_prefix}")

    parts = key[len(full_prefix) :].split("/")
    if not parts or parts[-1] != METADATA_FILENAME:
        raise ValueError(
            f"Unexpected registry key: {key}. "
            f"Expected {ENTRY_SEGMENTS}/{METADATA_FILENAME}"
        )

    return parse_model_id_from_segments(parts[:-1], key)


def model_info_from_metadata(
    metadata_dict: dict, model_id: ModelId, path: str
) -> ModelInfo:
    """
    Convert an archive metadata dict + ModelId into a ModelInfo.

    Args:
        metadata_dict: the parsed contents of the entry's ``metadata.json``
        model_id: the identifier the entry was found under
        path: a backend-specific pointer to the entry, recorded verbatim on
            the returned ModelInfo (an S3 key prefix, a repo-relative path, ...)

    Returns: a lightweight ModelInfo summary carrying no binary data
    """
    vehicle = metadata_dict["vehicle"]
    contract = metadata_dict["contract"]
    estimator = metadata_dict["estimator"]

    feature_names = [f["name"] for f in contract["feature_set"]]
    target_names = [t["name"] for t in contract["target"]]

    return ModelInfo(
        model_id=model_id,
        vehicle_model=vehicle.get("model"),
        estimator_type=estimator["estimator_type"],
        architecture_tag=estimator.get("architecture_tag", "unknown"),
        input_spec=estimator.get("input_spec"),
        feature_names=feature_names,
        target_names=target_names,
        powertrain_type=vehicle["powertrain_type"],
        vehicle_description=vehicle["vehicle_description"],
        path=path,
        mass_lbs=vehicle.get("mass_lbs"),
        fuel_type=vehicle.get("fuel_type"),
        drivetrain=vehicle.get("drivetrain"),
        engine=vehicle.get("engine"),
        trim=vehicle.get("trim"),
        model_digest=metadata_dict.get("model_digest"),
    )
