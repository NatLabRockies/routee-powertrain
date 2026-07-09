import logging
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import pandas as pd

from routee.powertrain.core.model import Model
from routee.powertrain.core.year import parse_year
from routee.powertrain.registry.filtering import VersionStrategy
from routee.powertrain.registry.model_id import ModelId, ModelInfo
from routee.powertrain.registry.registry import ModelRegistry
from routee.powertrain.resources.sample_routes import sample_route_dir

log = logging.getLogger(__name__)


def list_available_models(
    registry: Optional[ModelRegistry] = None,
    version_strategy: VersionStrategy = "latest",
) -> List[ModelId]:
    """
    Returns a list of model identifiers available in the registry.

    This is a lightweight operation that returns only the model paths
    without fetching any metadata or binaries.

    If no registry is provided, the default registry is used.

    Args:
        registry: a ModelRegistry instance; defaults to get_default_registry()
        version_strategy: ``"latest"`` (default) returns only the highest
            version per (make, model, year, config_slug) group; ``"all"``
            returns every versioned identifier.

    Returns: list of ModelId matching the strategy
    """
    if registry is None:
        from routee.powertrain.registry.default import get_default_registry

        registry = get_default_registry()

    return registry.list_models(version_strategy=version_strategy)


def query_available_models(
    make: Optional[str] = None,
    model: Optional[str] = None,
    year: Optional[int] = None,
    config_slug: Optional[str] = None,
    feature_names: Optional[Sequence[str]] = None,
    powertrain_type: Optional[str] = None,
    fuel_type: Optional[str] = None,
    drivetrain: Optional[str] = None,
    engine: Optional[str] = None,
    trim: Optional[str] = None,
    version: Optional[int] = None,
    model_digest: Optional[str] = None,
    version_strategy: VersionStrategy = "latest",
    custom_filters: Optional[Sequence[Callable[[ModelInfo], bool]]] = None,
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
        model: filter by model name
        year: filter by model year
        config_slug: filter by config slug (e.g. "rf_default", "cnn_5link")
        feature_names: filter to models whose feature set contains every listed
            feature column (subset match, exact names)
        powertrain_type: filter by powertrain type (e.g. "ICE", "BEV", "HEV")
        fuel_type: filter by fuel type (e.g. "GASOLINE", "DIESEL", "ELECTRICITY")
        drivetrain: filter by drivetrain (e.g. "FWD", "RWD", "AWD")
        engine: filter by engine specification (e.g. "4cyl", "2.0tdi")
        trim: filter by trim level (e.g. "sport", "active")
        version: pin results to an exact version (e.g. 2). When set,
            ``version_strategy`` is ignored.
        model_digest: pin results to an exact instance identity — the
            ``model_digest`` from a model's ``metadata.json`` (with or without
            the ``sha256:`` prefix; always matched exactly). Use this to
            resolve a model file in hand back to its registry entry. When set,
            ``version_strategy`` is ignored.
        version_strategy: how to collapse multiple versions of the same
            model. ``"latest"`` (default) keeps only the highest version
            per (make, model, year, config_slug) group; ``"all"`` returns
            every version. Ignored when ``version`` is specified.
        custom_filters: optional list of callables that accept a ModelInfo
            and return True to keep the model or False to exclude it.
            For example: ``[lambda m: m.mass_lbs is not None and m.mass_lbs > 10000]``
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
        model=model,
        year=year,
        config_slug=config_slug,
        feature_names=feature_names,
        powertrain_type=powertrain_type,
        fuel_type=fuel_type,
        drivetrain=drivetrain,
        engine=engine,
        trim=trim,
        version=version,
        model_digest=model_digest,
        version_strategy=version_strategy,
        custom_filters=custom_filters,
        fuzzy=fuzzy,
        fuzzy_threshold=fuzzy_threshold,
    )


def _resolve_load_target(name_or_path: str, registry: ModelRegistry) -> ModelId:
    """Resolve a string model identifier to a concrete ModelId.

    Accepts two forms:

    - ``<make>/<model>/<year>/<config_slug>/v<N>`` — explicit version.
    - ``<make>/<model>/<year>/<config_slug>`` — latest version is picked via
      a ``fuzzy=False`` registry query.

    Raises ``ValueError`` with an actionable message on unknown shapes,
    zero matches, or ambiguous matches.
    """
    parts = [p for p in name_or_path.strip("/").split("/") if p]
    if len(parts) == 5:
        return ModelId.from_path(name_or_path)
    if len(parts) == 4:
        make, model, year_str, config_slug = parts
        parsed_year = parse_year(year_str)
        year_filter = parsed_year if isinstance(parsed_year, int) else None
        hits = registry.query(
            make=make,
            model=model,
            year=year_filter,
            config_slug=config_slug,
            fuzzy=False,
            version_strategy="latest",
        )
        # If the path carries a year range, query() can't filter on it
        # via year_contains (which only accepts int), so drop any hits
        # whose stored year doesn't exactly match the requested shape.
        hits = [h for h in hits if h.model_id.year == parsed_year]
        if len(hits) == 1:
            return hits[0].model_id
        if not hits:
            raise ValueError(f"No model found for '{name_or_path}'")
        raise ValueError(
            f"Ambiguous partial id '{name_or_path}' — matched {len(hits)} models: "
            + ", ".join(str(h.model_id) for h in hits)
        )
    raise ValueError(
        f"Could not parse '{name_or_path}'. Expected either "
        "'<make>/<model>/<year>/<config_slug>/v<N>' (explicit version) or "
        "'<make>/<model>/<year>/<config_slug>' (latest)."
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
       registry (uses the default registry if none is provided). String
       paths may omit the ``v<N>`` segment to load the latest version.

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
    >>> mid = ModelId("toyota", "camry", 2016, "rf_default", 1)
    >>> model = pt.load_model(mid)
    >>>
    >>> # explicit version
    >>> model = pt.load_model("toyota/camry/2016/rf_default/v1")
    >>>
    >>> # latest version (no version segment)
    >>> model = pt.load_model("toyota/camry/2016/rf_default")

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

    if isinstance(name_or_path, (str, Path)):
        mid = _resolve_load_target(str(name_or_path), registry)
        return registry.load(mid)

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
