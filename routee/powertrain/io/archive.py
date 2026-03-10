from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Union

from routee.powertrain.core.metadata import Metadata
from routee.powertrain.validation.errors import ModelErrors

if TYPE_CHECKING:
    from routee.powertrain.core.model import Model

SCHEMA_VERSION = 2
METADATA_FILENAME = "metadata.json"


def _get_estimator_registry():
    from routee.powertrain.core.model import REGISTERED_ESTIMATORS

    return REGISTERED_ESTIMATORS


def save_archive(model: Model, path: Union[str, Path]) -> None:
    """
    Save a model as a ZIP archive containing metadata.json and a binary model file.

    Args:
        model: the model to save
        path: the path to write the .zip file to
    """
    path = Path(path)
    if path.suffix != ".zip":
        raise ValueError("Model archive must be a .zip file")

    estimator = model.estimator
    estimator_type = estimator.__class__.__name__
    model_filename = "model" + estimator.file_extension

    metadata_dict = {
        "schema_version": SCHEMA_VERSION,
        "estimator_type": estimator_type,
        "model_file": model_filename,
        "metadata": model.metadata.to_dict(),
        "errors": model.errors.to_dict(),
    }

    model_bytes = estimator.to_bytes()

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(METADATA_FILENAME, json.dumps(metadata_dict))
        zf.writestr(model_filename, model_bytes)


def load_archive(path: Union[str, Path]) -> Model:
    """
    Load a model from a ZIP archive.

    Args:
        path: the path to the .zip file

    Returns: a Model instance
    """

    path = Path(path)
    if path.suffix != ".zip":
        raise ValueError("Model archive must be a .zip file")

    with zipfile.ZipFile(path, "r") as zf:
        metadata_dict = json.loads(zf.read(METADATA_FILENAME))
        return _model_from_archive_dict(metadata_dict, zf)


def load_archive_bytes(data: bytes) -> Model:
    """
    Load a model from in-memory ZIP archive bytes.

    Args:
        data: the raw bytes of the zip archive

    Returns: a Model instance
    """
    with zipfile.ZipFile(BytesIO(data), "r") as zf:
        metadata_dict = json.loads(zf.read(METADATA_FILENAME))
        return _model_from_archive_dict(metadata_dict, zf)


def read_archive_metadata(path: Union[str, Path]) -> dict:
    """
    Read only the metadata.json from a ZIP archive without loading the model binary.

    Args:
        path: the path to the .zip file

    Returns: the parsed metadata dictionary
    """
    path = Path(path)
    with zipfile.ZipFile(path, "r") as zf:
        return json.loads(zf.read(METADATA_FILENAME))


def _model_from_archive_dict(metadata_dict: dict, zf: zipfile.ZipFile) -> Model:
    from routee.powertrain.core.model import Model

    schema_version = metadata_dict.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported archive schema version: {schema_version}. "
            f"Expected: {SCHEMA_VERSION}"
        )

    estimator_type_str = metadata_dict.get("estimator_type")
    if estimator_type_str is None:
        raise ValueError("Archive metadata must contain 'estimator_type'")

    registry = _get_estimator_registry()
    estimator_cls = registry.get(estimator_type_str)
    if estimator_cls is None:
        raise ValueError(
            f"Estimator type '{estimator_type_str}' is not registered. "
            f"Available types: {list(registry.keys())}"
        )

    model_filename = metadata_dict.get("model_file")
    if model_filename is None:
        raise ValueError("Archive metadata must contain 'model_file'")

    model_bytes = zf.read(model_filename)
    estimator = estimator_cls.from_bytes(model_bytes)

    metadata = Metadata.from_dict(metadata_dict["metadata"])
    errors = ModelErrors.from_dict(metadata_dict["errors"])

    return Model(estimator=estimator, metadata=metadata, errors=errors)
