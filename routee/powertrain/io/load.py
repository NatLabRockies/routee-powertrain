import logging
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from routee.powertrain.core.model import Model
from routee.powertrain.registry.model_id import ModelId, ModelInfo
from routee.powertrain.registry.registry import ModelRegistry
from routee.powertrain.resources.sample_routes import sample_route_dir

log = logging.getLogger(__name__)


def list_available_models(
    make: Optional[str] = None,
    model_name: Optional[str] = None,
    year: Optional[int] = None,
    trim: Optional[str] = None,
    variant: Optional[str] = None,
    registry: Optional[ModelRegistry] = None,
) -> List[ModelInfo]:
    """
    Returns a list of available pretrained models from the registry.

    If no registry is provided, the default registry is used.

    Args:
        make: filter by vehicle make
        model_name: filter by model name
        year: filter by model year
        trim: filter by trim
        variant: filter by variant
        registry: a ModelRegistry instance; defaults to get_default_registry()

    Returns: list of ModelInfo with full metadata and error metrics
    """
    if registry is None:
        from routee.powertrain.registry.default import get_default_registry

        registry = get_default_registry()

    return registry.query(
        make=make,
        model_name=model_name,
        year=year,
        trim=trim,
        variant=variant,
    )


def load_model(
    name_or_path: Union[str, Path, ModelId],
    registry: Optional[ModelRegistry] = None,
) -> Model:
    """
    Load a pretrained model.

    Supports two loading modes:

    1. **File path** — pass a ``Path`` or string pointing to a model
       directory (containing ``metadata.json``), ``.zip`` archive,
       or ``.tar.gz`` archive on disk.
    2. **Registry ModelId** — pass a ``ModelId`` to fetch from a
       registry (uses the default registry if none is provided).

    Args:
        name_or_path: file/directory path or ModelId
        registry: optional ModelRegistry for remote loading

    Returns: a routee-powertrain Model

    Examples:

    >>> import routee.powertrain as pt
    >>>
    >>> # load from a directory
    >>> model = pt.load_model("path/to/my_model/")
    >>>
    >>> # load from a zip
    >>> model = pt.load_model("MyModel.zip")
    >>>
    >>> # load via registry
    >>> from routee.powertrain.registry import ModelId, get_default_registry
    >>> reg = get_default_registry()
    >>> mid = ModelId("toyota", "camry", 2016, "4cyl_fwd", "default", 1)
    >>> model = pt.load_model(mid, registry=reg)

    """
    # Mode 1: ModelId with registry
    if isinstance(name_or_path, ModelId):
        if registry is None:
            from routee.powertrain.registry.default import get_default_registry

            registry = get_default_registry()
        return registry.load(name_or_path)

    # Mode 2: file or directory path on disk
    path = Path(name_or_path)
    if path.exists():
        return Model.from_file(path)

    raise ValueError(
        f"Could not load model: {name_or_path}. "
        "Provide a valid file/directory path or a ModelId with a registry."
    )


def load_sample_route(name: Optional[str] = None) -> pd.DataFrame:
    """
    A helper function to load sample routes

    Args:
        name: The name of the route. Defaults to "sample_route".

    Returns: a pandas DataFrame representing the route

    """
    routes = {
        "sample_route": sample_route_dir() / "sample_route.csv",
    }

    if name is None:
        name = "sample_route"

    if name not in routes:
        raise KeyError(
            f"cannot find route with name: {name}; try one of {list(routes.keys())}"
        )

    df = pd.read_csv(routes[name])

    return df
