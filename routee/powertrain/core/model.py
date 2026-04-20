from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING, Union

import pandas as pd

from routee.powertrain.core.metadata import Metadata
from routee.powertrain.core.real_world_adjustments import ADJUSTMENT_FACTORS
from routee.powertrain.estimators.cnn import CNNEstimator
from routee.powertrain.estimators.estimator_interface import Estimator
from routee.powertrain.estimators.onnx import ONNXEstimator
from routee.powertrain.estimators.smart_core import SmartCoreEstimator
from routee.powertrain.estimators.ngboost_estimator import NGBoostEstimator

from routee.powertrain.io.archive import (
    load_model_from_path,
    save_model_directory,
    save_archive,
    save_tar_archive,
)
from routee.powertrain.io.to_lookup_table import to_lookup_table
from routee.powertrain.validation.feature_visualization import (
    contour_plot,
    visualize_features,
)

if TYPE_CHECKING:
    from pandas import Series

REGISTERED_ESTIMATORS = {
    "ONNXEstimator": ONNXEstimator,
    "SmartCoreEstimator": SmartCoreEstimator,
    "NGBoostEstimator": NGBoostEstimator,
    "CNNEstimator": CNNEstimator,
}


@dataclass
class Model:
    """
    A RouteE-Powertrain vehicle model represents a single vehicle
    (i.e. a 2016 Toyota Camry with a 1.5 L gasoline engine).
    """

    estimator: Estimator
    metadata: Metadata

    @property
    def feature_set(self):
        return self.metadata.config.feature_set

    @property
    def feature_names(self) -> List[str]:
        return self.metadata.config.feature_set.feature_name_list

    @classmethod
    def from_file(cls, file: Union[str, Path]):
        """
        Load a vehicle model from a file or directory.

        Supports directories (containing metadata.json + binary),
        .zip archives, and .tar.gz archives.

        Args:
            file: the path to the file or directory to load

        Returns: a powertrain vehicle
        """
        return load_model_from_path(file)

    def to_file(self, file: Union[str, Path]):
        """
        Save a vehicle model to a file or directory.

        If *file* has no suffix, saves as a flat directory.
        If it ends with ``.zip``, saves as a ZIP archive.
        If it ends with ``.tar.gz``, saves as a tar archive.

        Args:
            file: the path to save to
        """
        path = Path(file)
        if path.suffix == ".zip":
            path.parent.mkdir(parents=True, exist_ok=True)
            save_archive(self, path)
        elif path.name.endswith(".tar.gz") or path.suffix == ".tar":
            path.parent.mkdir(parents=True, exist_ok=True)
            save_tar_archive(self, path)
        else:
            # No extension → flat directory
            save_model_directory(self, path)

    def to_lookup_table(
        self,
        feature_parameters: list[dict],
        energy_target: str,
    ) -> pd.DataFrame:
        """
        Convert the the model to a lookup table for the given feature parameters.
        """
        return to_lookup_table(self, feature_parameters, energy_target)

    def visualize_features(
        self,
        n_samples: Optional[int] = 100,
        output_path: Optional[str] = None,
        return_predictions: Optional[bool] = False,
    ) -> Optional[Dict[str, "Series"]]:
        """
        generates test links to independently test the model's features
        and creates plots of those predictions

        Args:
            n_samples: the number of samples used to generate the plots
            output_path: an optional path to save the plots as png files.
            return_predictions: if true, returns the dictionary containing the prediction values

        Returns: optionally returns a dictionary containing the predictions where the key is the feature tested
        """
        feature_set = self.metadata.config.feature_set
        feature_ranges = {}
        for f in feature_set.features:
            if f.constraints.upper is None or f.constraints.lower is None:
                raise ValueError(
                    f"Feature: {f.name} has constraints without a lower/upper bound. "
                    f"You can add constraints when training a model or set custom constraints during visualization using "
                    f"routee.powertrain.validation.feature_visualization.visualize_features"
                )
            feature_ranges[f.name] = {
                "upper": f.constraints.upper,
                "lower": f.constraints.lower,
                "n_samples": n_samples,
            }

        return visualize_features(
            model=self,
            feature_ranges=feature_ranges,
            output_path=output_path,
            return_predictions=return_predictions,
        )

    def contour(
        self,
        x_feature: str,
        y_feature: str,
        n_samples: Optional[int] = 100,
        output_path: Optional[str] = None,
    ):
        """
        generates a contour plot of the two test features: x_feature and y_feature.

        Args:
            x_feature: one of the features used to generate the energy matrix
                and will be the x-axis feature
            y_feature: one of the features used to generate the energy matrix
                and will be the y-axis feature
            n_samples: the number of samples used to generate the plots
            output_path: an optional path to save the plots as png files.
        """
        feature_set = self.metadata.config.feature_set
        feature_ranges = {}
        for f in feature_set.features:
            if f.constraints.upper is None or f.constraints.lower is None:
                raise ValueError(
                    f"Feature: {f.name} has constraints without a lower/upper bound. "
                    f"You can add constraints when training a model or set custom constraints during visualization using "
                    f"routee.powertrain.validation.feature_visualization.contour_plot"
                )
            feature_ranges[f.name] = {
                "upper": f.constraints.upper,
                "lower": f.constraints.lower,
                "n_samples": n_samples,
            }

        return contour_plot(
            model=self,
            x_feature=x_feature,
            y_feature=y_feature,
            feature_ranges=feature_ranges,
            output_path=output_path,
        )

    def predict(
        self,
        links_df: pd.DataFrame,
        distance_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Predict absolute energy consumption for each link

        Args:
            links_df: a dataframe containing the links to predict on. Must contain
                every column in ``self.feature_set`` plus the distance column, and
                (if the estimator's ``input_spec`` declares a grouping column) that
                grouping column.
            distance_column: optional rename hook; if provided, this column is
                renamed to the model's configured distance column before prediction.

        Returns: a dataframe containing the predicted energy consumption for each link
        """
        config = self.metadata.config

        if distance_column is None:
            distance_column = config.distance.name
            if distance_column not in links_df.columns:
                raise ValueError(
                    f"links_df must contain a distance column named: '{distance_column}'"
                )
        else:
            links_df = links_df.rename(columns={distance_column: config.distance.name})

        feature_set = config.feature_set
        missing = [
            f.name for f in feature_set.features if f.name not in links_df.columns
        ]
        if missing:
            raise ValueError(
                f"links_df is missing the following required features: {missing}. "
                f"Expected features: {feature_set.feature_name_list}"
            )

        input_spec = self.estimator.input_spec
        if input_spec.grouping_column is not None:
            if input_spec.grouping_column not in links_df.columns:
                raise ValueError(
                    f"Estimator requires a grouping column '{input_spec.grouping_column}' "
                    f"(lookback={input_spec.lookback}) but it was not found in links_df."
                )

        pred_energy_df = self.estimator.predict(links_df, config)

        for energy in config.target.targets:
            if config.apply_real_world_adjustment:
                adjustment_factor = ADJUSTMENT_FACTORS.get(config.powertrain_type)
                if adjustment_factor is None:
                    raise ValueError(
                        f"Could not find an adjustment factor for powertrain type "
                        f"{config.powertrain_type}"
                    )
                pred_energy_df[energy.name] = (
                    pred_energy_df[energy.name] * adjustment_factor
                )

        return pred_energy_df

    def __repr__(self) -> str:
        """
        Shows a nice pretty printed summary of the model including:
         - Model average fuel consumption
         - Select set of errors
         - Expected features and their units
         - Powertrain specifications
        """
        config = self.metadata.config
        summary_lines = []
        summary_lines.append("=" * 40)
        summary_lines.append("Model Summary")
        summary_lines.append("-" * 20)
        summary_lines.append(f"Vehicle description: {config.vehicle_description}")
        summary_lines.append(f"Powertrain type: {config.powertrain_type.name}")
        summary_lines.append("=" * 40)

        estimator_errors = self.metadata.errors.estimator_errors
        summary_lines.append("Estimator Summary")
        summary_lines.append("-" * 20)
        feature_set = config.feature_set
        for feature in feature_set.features:
            summary_lines.append(f"Feature: {feature.name} ({feature.units})")
        summary_lines.append(
            f"Distance: {config.distance.name} ({config.distance.units})"
        )
        for target in config.target.targets:
            summary_lines.append(f"Target: {target.name} ({target.units})")
            target_errors = estimator_errors.error_by_target.get(target.name)
            if target_errors is None:
                raise ValueError(f"Could not find errors for target {target.name}")

            summary_lines.append(
                f"Raw Predicted Consumption: {target_errors.pred_dist_per_energy:.3f} "
                f"({config.distance.units}/{target.units})"
            )
            summary_lines.append(
                f"Real World Predicted Consumption: {target_errors.real_world_pred_dist_per_energy:.3f} "
                f"({config.distance.units}/{target.units})"
            )
        summary_lines.append(f"Predict Method: {config.predict_method.value.upper()}")
        summary_lines.append("=" * 40)
        return "\n".join(summary_lines)

    def _repr_html_(self) -> str:
        """
        Returns an html table of the model summary for display in a notebook
        """
        config = self.metadata.config

        # Start the HTML table
        html_lines = ['<table border="1" style="border-collapse: collapse;">']

        # Title: Model Summary
        html_lines.append(
            '<tr><th colspan="2" style="border-bottom: 2px solid black; text-align: center;">Model Summary</th></tr>'
        )
        html_lines.append(
            f"<tr><td>Vehicle description</td><td>{config.vehicle_description}</td></tr>"
        )
        html_lines.append(
            f"<tr><td>Powertrain type</td><td>{config.powertrain_type.name}</td></tr>"
        )

        estimator_errors = self.metadata.errors.estimator_errors

        # Title: Estimator Summary
        html_lines.append(
            '<tr><th colspan="2" style="border-bottom: 2px solid black; text-align: center;">Estimator Summary</th></tr>'
        )

        feature_set = config.feature_set
        for feature in feature_set.features:
            html_lines.append(
                f"<tr><td>Feature</td><td>{feature.name} ({feature.units})</td></tr>"
            )

        html_lines.append(
            "<tr><td>Distance</td>"
            f"<td>{config.distance.name} ({config.distance.units})</td></tr>"
        )

        for target in config.target.targets:
            html_lines.append(
                f"<tr><td>Target</td><td>{target.name} ({target.units})</td></tr>"
            )

            target_errors = estimator_errors.error_by_target.get(target.name)
            if target_errors is None:
                raise ValueError(f"Could not find errors for target {target.name}")

            html_lines.append(
                "<tr><td>Predicted Consumption</td>"
                f"<td>{target_errors.pred_dist_per_energy:.3f} "
                f"({config.distance.units}/{target.units})</td></tr>"
            )

            html_lines.append(
                "<tr><td>Real World Predicted Consumption</td>"
                f"<td>{target_errors.real_world_pred_dist_per_energy:.3f} "
                f"({config.distance.units}/{target.units})</td></tr>"
            )
        html_lines.append(
            f"<tr><td>Predict Method</td>"
            f"<td>{config.predict_method.value.upper()}</td></tr>"
        )

        # End the HTML table
        html_lines.append("</table>")

        return "".join(html_lines)
