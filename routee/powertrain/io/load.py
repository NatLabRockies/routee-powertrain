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
    registry: Optional[ModelRegistry] = None,
) -> List[ModelId]:
    """
    Returns a list of model identifiers available in the registry.

    This is a lightweight operation that returns only the model paths
    without fetching any metadata or binaries.

    If no registry is provided, the default registry is used.

    Args:
        registry: a ModelRegistry instance; defaults to get_default_registry()

    Returns: list of ModelId for every model in the registry
    """
    if registry is None:
        from routee.powertrain.registry.default import get_default_registry

        registry = get_default_registry()

    return registry.list_models()


def query_available_models(
    make: Optional[str] = None,
    model_name: Optional[str] = None,
    year: Optional[int] = None,
    variant: Optional[str] = None,
    registry: Optional[ModelRegistry] = None,
    fuzzy: bool = True,
    fuzzy_threshold: int = 80,
) -> List[ModelInfo]:
    """
    Query available pretrained models from the registry with optional filters.

    Returns full metadata and error metrics for each matching model.
    If no registry is provided, the default registry is used.

    Args:
        make: filter by vehicle make
        model_name: filter by model name
        year: filter by model year
        variant: filter by variant
        registry: a ModelRegistry instance; defaults to get_default_registry()
        fuzzy: if True, use fuzzy string matching for string fields (default True)
        fuzzy_threshold: minimum score (0–100) for a fuzzy match (default 80)

    Returns: list of ModelInfo with full metadata and error metrics
    """
    if registry is None:
        from routee.powertrain.registry.default import get_default_registry

        registry = get_default_registry()

    return registry.query(
        make=make,
        model_name=model_name,
        year=year,
        variant=variant,
        fuzzy=fuzzy,
        fuzzy_threshold=fuzzy_threshold,
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
    2. **Registry ModelId** — pass a ``ModelId`` object or model id path to fetch from a
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
    >>> from routee.powertrain.registry import ModelId
    >>> mid = ModelId("toyota", "camry", 2016, "default", "grade_speed", 1)
    >>> model = pt.load_model(mid)
    >>>
    >>> mid_str = "toyota/camry/2016/default/grade_speed/v1"
    >>> model = pt.load_model(mid_str)

    """
    # First assume the model is local
    if isinstance(name_or_path, (str, Path)):
        path = Path(name_or_path)
        if path.exists():
            return Model.from_file(path)

    # If not found locally, try loading from registry if it's a ModelId or a valid ModelId string
    if registry is None:
        from routee.powertrain.registry.default import get_default_registry

        registry = get_default_registry()

    if isinstance(name_or_path, ModelId):
        return registry.load(name_or_path)

    if isinstance(name_or_path, str):
        try:
            mid = ModelId.from_path(name_or_path)
            return registry.load(mid)
        except ValueError:
            pass

    raise ValueError(
        f"Could not load model: {name_or_path}. "
        "Provide a valid local file/directory or a valid ModelId/string with a registry."
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
