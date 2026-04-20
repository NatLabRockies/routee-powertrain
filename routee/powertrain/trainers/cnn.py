from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import onnx
import pandas as pd

from routee.powertrain.core.model_config import ModelConfig
from routee.powertrain.estimators.cnn import (
    CNN_INPUT_NAME,
    DEFAULT_GROUPING_COLUMN,
    DEFAULT_LOOKBACK,
    CNNEstimator,
)
from routee.powertrain.estimators.estimator_interface import Estimator
from routee.powertrain.trainers.trainer import Trainer


class CNNTrainer(Trainer):
    """Train a 1D CNN over link sequences with a fixed lookback window.

    The training input dataframe must contain the grouping column
    (default ``route_id``) so training windows don't cross route boundaries.
    The trained model is exported to ONNX for runtime use, keeping
    ``torch`` out of the runtime dependency set.
    """

    architecture_tag: str = "cnn"

    @property
    def required_extra_columns(self):
        return [self.grouping_column]

    def __init__(
        self,
        lookback: int = DEFAULT_LOOKBACK,
        grouping_column: str = DEFAULT_GROUPING_COLUMN,
        pad_strategy: Literal["zero", "repeat_first"] = "zero",
        hidden_channels: int = 32,
        kernel_size: int = 3,
        epochs: int = 20,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        random_seed: int = 52,
    ):
        self.lookback = lookback
        self.grouping_column = grouping_column
        self.pad_strategy = pad_strategy
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_seed = random_seed

    def inner_train(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame,
        config: ModelConfig,
    ) -> Estimator:
        try:
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise ImportError(
                "CNNTrainer requires torch. Install via pip install routee.powertrain[pytorch]"
            ) from exc

        if self.grouping_column not in features.columns:
            raise ValueError(
                f"CNNTrainer requires a '{self.grouping_column}' column in the "
                f"training features so windows can be built per route."
            )

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        feature_cols = [c for c in features.columns if c != self.grouping_column]
        feature_matrix = features[feature_cols].to_numpy(dtype=np.float32)
        y_matrix = target.to_numpy(dtype=np.float32)
        if y_matrix.ndim == 1:
            y_matrix = y_matrix.reshape(-1, 1)
        n_features = len(feature_cols)
        n_targets = y_matrix.shape[1]

        windows = np.zeros((len(features), self.lookback, n_features), dtype=np.float32)
        groups = features.groupby(self.grouping_column, sort=False).indices
        for idx_array in groups.values():
            idx_array = np.asarray(idx_array)
            group_features = feature_matrix[idx_array]
            if self.pad_strategy == "repeat_first":
                pad_row = group_features[:1]
            else:
                pad_row = np.zeros((1, n_features), dtype=np.float32)
            padded = np.concatenate(
                [np.repeat(pad_row, self.lookback - 1, axis=0), group_features],
                axis=0,
            )
            for i, row_pos in enumerate(idx_array):
                windows[row_pos] = padded[i : i + self.lookback]

        X = torch.from_numpy(windows)
        y = torch.from_numpy(y_matrix)
        loader = DataLoader(
            TensorDataset(X, y), batch_size=self.batch_size, shuffle=True
        )

        model = _CNN1D(
            n_features=n_features,
            lookback=self.lookback,
            hidden_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            n_targets=n_targets,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

        model.eval()
        dummy = torch.zeros(1, self.lookback, n_features, dtype=torch.float32)
        with tempfile.TemporaryDirectory() as td:
            onnx_path = Path(td) / "cnn.onnx"
            torch.onnx.export(
                model,
                (dummy,),
                onnx_path,
                input_names=[CNN_INPUT_NAME],
                output_names=["output"],
                dynamic_axes={CNN_INPUT_NAME: {0: "batch"}, "output": {0: "batch"}},
                opset_version=17,
                dynamo=False,
            )
            onnx_proto = onnx.load_from_string(onnx_path.read_bytes())

        return CNNEstimator(
            onnx_proto,
            lookback=self.lookback,
            grouping_column=self.grouping_column,
            pad_strategy=self.pad_strategy,
        )


def _CNN1D(
    n_features: int,
    lookback: int,
    hidden_channels: int,
    kernel_size: int,
    n_targets: int,
):
    """A small 1D CNN that maps (N, lookback, F) → (N, n_targets)."""
    import torch
    from torch import nn

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            padding = kernel_size // 2
            self.conv1 = nn.Conv1d(
                n_features, hidden_channels, kernel_size, padding=padding
            )
            self.conv2 = nn.Conv1d(
                hidden_channels, hidden_channels, kernel_size, padding=padding
            )
            self.relu = nn.ReLU()
            self.head = nn.Linear(hidden_channels * lookback, n_targets)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (N, lookback, F) → (N, F, lookback) for Conv1d
            x = x.transpose(1, 2)
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = x.flatten(start_dim=1)
            return self.head(x)

    return CNN()
