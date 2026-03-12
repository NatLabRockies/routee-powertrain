from __future__ import annotations

import json
import tarfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Union

from routee.powertrain.core.metadata import Metadata, SCHEMA_VERSION

if TYPE_CHECKING:
    from routee.powertrain.core.model import Model

METADATA_FILENAME = "metadata.json"


def _get_estimator_registry():
    from routee.powertrain.core.model import REGISTERED_ESTIMATORS

    return REGISTERED_ESTIMATORS


def _build_metadata_dict(model: Model) -> dict:
    """Build the metadata dictionary for serialization."""
    return model.metadata.to_dict()


def _model_from_metadata_and_bytes(metadata_dict: dict, model_bytes: bytes) -> Model:
    """Reconstruct a Model from a parsed metadata dict and raw model bytes."""
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

    estimator = estimator_cls.from_bytes(model_bytes)
    metadata = Metadata.from_dict(metadata_dict)

    return Model(estimator=estimator, metadata=metadata)


# ---------------------------------------------------------------------------
# Directory format (default)
# ---------------------------------------------------------------------------


def save_model_directory(model: Model, path: Union[str, Path]) -> None:
    """
    Save a model as a flat directory containing metadata.json and a binary model file.

    Args:
        model: the model to save
        path: the directory path to write to
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    metadata_dict = _build_metadata_dict(model)
    model_filename = metadata_dict["model_file"]

    (path / METADATA_FILENAME).write_text(json.dumps(metadata_dict, indent=2))
    (path / model_filename).write_bytes(model.estimator.to_bytes())


def load_model_directory(path: Union[str, Path]) -> Model:
    """
    Load a model from a directory containing metadata.json and a binary model file.

    Args:
        path: the directory path to read from

    Returns: a Model instance
    """
    path = Path(path)
    metadata_path = path / METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {path}")

    metadata_dict = json.loads(metadata_path.read_text())
    model_filename = metadata_dict.get("model_file")
    if model_filename is None:
        raise ValueError("metadata.json must contain 'model_file'")

    model_bytes = (path / model_filename).read_bytes()
    return _model_from_metadata_and_bytes(metadata_dict, model_bytes)


def read_directory_metadata(path: Union[str, Path]) -> dict:
    """
    Read only the metadata.json from a model directory.

    Args:
        path: the directory path

    Returns: the parsed metadata dictionary
    """
    path = Path(path)
    metadata_path = path / METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {path}")
    return json.loads(metadata_path.read_text())


# ---------------------------------------------------------------------------
# ZIP format
# ---------------------------------------------------------------------------


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

    metadata_dict = _build_metadata_dict(model)
    model_filename = metadata_dict["model_file"]
    model_bytes = model.estimator.to_bytes()

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
        model_filename = metadata_dict.get("model_file", "")
        model_bytes = zf.read(model_filename)
        return _model_from_metadata_and_bytes(metadata_dict, model_bytes)


def load_archive_bytes(data: bytes) -> Model:
    """
    Load a model from in-memory ZIP archive bytes.

    Args:
        data: the raw bytes of the zip archive

    Returns: a Model instance
    """
    with zipfile.ZipFile(BytesIO(data), "r") as zf:
        metadata_dict = json.loads(zf.read(METADATA_FILENAME))
        model_filename = metadata_dict.get("model_file")
        if model_filename is None:
            raise ValueError("Archive metadata must contain 'model_file'")
        model_bytes = zf.read(model_filename)
        return _model_from_metadata_and_bytes(metadata_dict, model_bytes)


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


# ---------------------------------------------------------------------------
# Tar format (.tar.gz)
# ---------------------------------------------------------------------------


def save_tar_archive(model: Model, path: Union[str, Path]) -> None:
    """
    Save a model as a .tar.gz archive containing metadata.json and a binary model file.

    Args:
        model: the model to save
        path: the path to write the .tar.gz file to
    """
    path = Path(path)

    metadata_dict = _build_metadata_dict(model)
    model_filename = metadata_dict["model_file"]
    model_bytes = model.estimator.to_bytes()
    metadata_bytes = json.dumps(metadata_dict).encode()

    with tarfile.open(path, "w:gz") as tf:
        meta_info = tarfile.TarInfo(name=METADATA_FILENAME)
        meta_info.size = len(metadata_bytes)
        tf.addfile(meta_info, BytesIO(metadata_bytes))

        model_info = tarfile.TarInfo(name=model_filename)
        model_info.size = len(model_bytes)
        tf.addfile(model_info, BytesIO(model_bytes))


def load_tar_archive(path: Union[str, Path]) -> Model:
    """
    Load a model from a .tar.gz archive.

    Args:
        path: the path to the .tar.gz file

    Returns: a Model instance
    """
    path = Path(path)
    with tarfile.open(path, "r:gz") as tf:
        meta_member = tf.getmember(METADATA_FILENAME)
        meta_file = tf.extractfile(meta_member)
        if meta_file is None:
            raise ValueError(f"Could not extract {METADATA_FILENAME} from {path}")
        metadata_dict = json.loads(meta_file.read())

        model_filename = metadata_dict.get("model_file", "")
        model_member = tf.getmember(model_filename)
        model_file = tf.extractfile(model_member)
        if model_file is None:
            raise ValueError(f"Could not extract {model_filename} from {path}")
        model_bytes = model_file.read()

        return _model_from_metadata_and_bytes(metadata_dict, model_bytes)


# ---------------------------------------------------------------------------
# Polymorphic helpers
# ---------------------------------------------------------------------------


def load_model_from_path(path: Union[str, Path]) -> Model:
    """
    Auto-detect format (directory, .zip, or .tar.gz) and load a model.

    Args:
        path: path to a model directory, .zip file, or .tar.gz file

    Returns: a Model instance
    """
    path = Path(path)
    if path.is_dir():
        return load_model_directory(path)
    elif path.suffix == ".zip":
        return load_archive(path)
    elif path.name.endswith(".tar.gz") or path.suffix == ".tar":
        return load_tar_archive(path)
    else:
        raise ValueError(
            f"Unsupported model format: {path}. "
            "Expected a directory, .zip file, or .tar.gz file."
        )


def read_metadata(path: Union[str, Path]) -> dict:
    """
    Read metadata.json from any supported format (directory, .zip, or .tar.gz).

    Args:
        path: path to a model directory, .zip file, or .tar.gz file

    Returns: the parsed metadata dictionary
    """
    path = Path(path)
    if path.is_dir():
        return read_directory_metadata(path)
    elif path.suffix == ".zip":
        return read_archive_metadata(path)
    elif path.name.endswith(".tar.gz") or path.suffix == ".tar":
        with tarfile.open(path, "r:gz") as tf:
            meta_member = tf.getmember(METADATA_FILENAME)
            meta_file = tf.extractfile(meta_member)
            if meta_file is None:
                raise ValueError(f"Could not extract {METADATA_FILENAME} from {path}")
            return json.loads(meta_file.read())
    else:
        raise ValueError(
            f"Unsupported model format: {path}. "
            "Expected a directory, .zip file, or .tar.gz file."
        )
