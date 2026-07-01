import logging
from pathlib import Path

__all__ = [
    "DataColumn",
    "Drivetrain",
    "FeatureSet",
    "FuelType",
    "Constraints",
    "TargetSet",
    "Model",
    "ModelConfig",
    "Vehicle",
    "Contract",
    "TrainingConfig",
    "Metadata",
    "EstimatorInfo",
    "PowertrainType",
    "ModelId",
    "ModelKey",
    "ModelInfo",
    "ModelRegistry",
    "list_available_models",
    "query_available_models",
    "load_model",
    "load_sample_route",
    "save_to_registry",
    "visualize_features",
    "contour_plot",
]

from .core.drivetrain import Drivetrain
from .core.features import DataColumn, FeatureSet, Constraints, TargetSet
from .core.fuel_type import FuelType
from .core.model import Model
from .core.model_config import ModelConfig, Vehicle, Contract, TrainingConfig
from .core.metadata import Metadata, EstimatorInfo
from .core.powertrain_type import PowertrainType
from .io.archive import save_to_registry
from .io.load import (
    list_available_models,
    query_available_models,
    load_model,
    load_sample_route,
)
from .registry.model_id import ModelId, ModelInfo, ModelKey
from .registry.registry import ModelRegistry
from .validation.feature_visualization import visualize_features, contour_plot

log = logging.getLogger()
log.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
log.addHandler(stream_handler)


def package_root() -> Path:
    return Path(__file__).parent
