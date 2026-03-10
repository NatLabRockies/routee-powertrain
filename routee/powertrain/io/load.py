import json
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from routee.powertrain.core.model import Model
from routee.powertrain.resources.default_models import default_model_dir
from routee.powertrain.resources.sample_routes import sample_route_dir

local_models = {
    "2016_TOYOTA_Camry_4cyl_2WD": default_model_dir()
    / "2016_TOYOTA_Camry_4cyl_2WD.json",
    "2017_CHEVROLET_Bolt": default_model_dir() / "2017_CHEVROLET_Bolt.json",
}


def list_available_models(local: bool = True, external: bool = True) -> List[str]:
    """
    returns a list of all the available pretrained models

    Args:
        local: include local models?
        external: include external models?

    Returns: a list of model keys
    """
    model_names = []
    if local:
        model_names.extend(list(local_models.keys()))

    if external:
        with open(default_model_dir() / "external_model_links.json", "r") as jf:
            external_models = json.load(jf)

            model_names.extend(list(external_models.keys()))

    return model_names


def load_model(name: Union[str, Path]) -> Model:
    """
    A helper function to load a pretrained model.
    If the model is a file, it will be loaded from disk.
    If the model is a name, it will be loaded from the default model catalog
    (local or external).

    Args:
        name: the name of the file or default model to load

    Returns: a routee-powertrain model

    Examples:

    >>> import routee.powertrain as pt
    >>>
    >>> # load a default model
    >>> model = pt.load_model("2016_HYUNDAI_Elantra_4cyl_2WD")
    >>>
    >>> # load a model from file
    >>> model = pt.load_model("MyModel.json")

    """
    path = Path(name)
    if path.exists():
        return Model.from_file(path)

    # otherwise, assume the name is a model name to be loaded from the default catalog
    name = str(name)

    with open(default_model_dir() / "external_model_links.json", "r") as jf:
        external_models = json.load(jf)

    if name in local_models:
        fp = local_models[name]
        model = Model.from_file(fp)
        return model
    elif name in external_models:
        url = external_models[name]
        model = Model.from_url(url)
        return model
    else:
        raise ValueError(
            f"Could not load model: {name}."
            " try listing available models with pt.list_available_models()"
            " or providing a path to a local model file."
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
