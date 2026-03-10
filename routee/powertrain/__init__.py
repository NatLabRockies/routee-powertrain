import logging
from pathlib import Path

__all__ = [
    "DataColumn",
    "FeatureSet",
    "Constraints",
    "TargetSet",
    "Model",
    "ModelConfig",
    "PowertrainType",
    "list_available_models",
    "load_model",
    "load_sample_route",
    "visualize_features",
    "contour_plot",
]

from .core.features import DataColumn, FeatureSet, Constraints, TargetSet
from .core.model import Model
from .core.model_config import ModelConfig
from .core.powertrain_type import PowertrainType
from .io.load import list_available_models, load_model, load_sample_route
from .validation.feature_visualization import visualize_features, contour_plot

log = logging.getLogger()
log.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
log.addHandler(stream_handler)


def package_root() -> Path:
    return Path(__file__).parent
