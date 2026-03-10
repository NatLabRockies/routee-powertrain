import json
import logging
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from routee.powertrain.core.model import Model
from routee.powertrain.registry.model_id import ModelId, ModelInfo
from routee.powertrain.registry.registry import ModelRegistry
from routee.powertrain.resources.default_models import default_model_dir
from routee.powertrain.resources.sample_routes import sample_route_dir

log = logging.getLogger(__name__)

local_models = {
    "2016_TOYOTA_Camry_4cyl_2WD": default_model_dir()
    / "2016_TOYOTA_Camry_4cyl_2WD.json",
    "2017_CHEVROLET_Bolt": default_model_dir() / "2017_CHEVROLET_Bolt.json",
}


def list_available_models(
    make: Optional[str] = None,
    model_name: Optional[str] = None,
    year: Optional[int] = None,
    trim: Optional[str] = None,
    variant: Optional[str] = None,
    registry: Optional[ModelRegistry] = None,
    local: bool = True,
    external: bool = True,
) -> Union[List[ModelInfo], List[str]]:
    """
    Returns a list of available pretrained models.

    If a registry is provided, queries it and returns ``List[ModelInfo]``
    with full metadata and error metrics.  Otherwise falls back to the
    legacy catalog (local bundled models + external links) and returns
    ``List[str]`` of model name keys.

    Args:
        make: filter by vehicle make (registry mode only)
        model_name: filter by model name (registry mode only)
        year: filter by model year (registry mode only)
        trim: filter by trim (registry mode only)
        variant: filter by variant (registry mode only)
        registry: a ModelRegistry instance; when provided, uses registry mode
        local: include local bundled models (legacy mode only)
        external: include external models (legacy mode only)

    Returns: list of ModelInfo (registry mode) or list of name strings (legacy mode)
    """
    if registry is not None:
        return registry.query(
            make=make,
            model_name=model_name,
            year=year,
            trim=trim,
            variant=variant,
        )

    # Legacy catalog mode
    model_names: List[str] = []
    if local:
        model_names.extend(list(local_models.keys()))
    if external:
        external_path = default_model_dir() / "external_model_links.json"
        if external_path.exists():
            with open(external_path, "r") as jf:
                external_models = json.load(jf)
                model_names.extend(list(external_models.keys()))
    return model_names


def load_model(
    name_or_path: Union[str, Path, ModelId],
    registry: Optional[ModelRegistry] = None,
) -> Model:
    """
    Load a pretrained model.

    Supports three loading modes:

    1. **File path** — pass a ``Path`` or string pointing to a ``.zip``
       (new archive format) or ``.json`` (legacy) file on disk.
    2. **Registry ModelId** — pass a ``ModelId`` with an explicit
       ``registry`` to fetch from a remote or local registry.
    3. **Legacy name** — pass a model name string (e.g.
       ``"2016_TOYOTA_Camry_4cyl_2WD"``) to look up in the bundled
       catalog or external links (no registry needed).

    Args:
        name_or_path: file path, ModelId, or legacy model name
        registry: optional ModelRegistry for remote loading

    Returns: a routee-powertrain Model

    Examples:

    >>> import routee.powertrain as pt
    >>>
    >>> # load from a file
    >>> model = pt.load_model("MyModel.zip")
    >>>
    >>> # load via legacy name
    >>> model = pt.load_model("2016_TOYOTA_Camry_4cyl_2WD")
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

    # Mode 2: file path on disk
    path = Path(name_or_path)
    if path.exists():
        return Model.from_file(path)

    # Mode 3: legacy catalog lookup
    name = str(name_or_path)

    external_path = default_model_dir() / "external_model_links.json"
    external_models: dict = {}
    if external_path.exists():
        with open(external_path, "r") as jf:
            external_models = json.load(jf)

    if name in local_models:
        return Model.from_file(local_models[name])
    elif name in external_models:
        return Model.from_url(external_models[name])
    else:
        raise ValueError(
            f"Could not load model: {name}. "
            "Try listing available models with pt.list_available_models() "
            "or providing a path to a local model file."
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
