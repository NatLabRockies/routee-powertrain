from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Literal

import numpy as np
import onnx
import pandas as pd

from routee.powertrain.core.model_config import ModelConfig
from routee.powertrain.estimators.estimator_interface import Estimator, InputSpec
from routee.powertrain.estimators.onnx import ONNX_INPUT_NAME, ONNXEstimator
from routee.powertrain.trainers.trainer import Trainer

log = logging.getLogger(__name__)

DEFAULT_LOOKBACK = 5
DEFAULT_GROUPING_COLUMN = "route_id"


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

    @property
    def split_grouping_column(self) -> str | None:
        return self.grouping_column

    def __init__(
        self,
        lookback: int = DEFAULT_LOOKBACK,
        grouping_column: str = DEFAULT_GROUPING_COLUMN,
        pad_strategy: Literal["zero", "repeat_first"] = "repeat_first",
        hidden_channels: int = 128,
        kernel_size: int = 3,
        head_hidden_1: int = 256,
        head_hidden_2: int = 128,
        dropout: float = 0.1,
        epochs: int = 15,
        batch_size: int = 2048,
        learning_rate: float = 3e-3,
        weight_decay: float = 1e-4,
        grad_clip_norm: float = 1.0,
        normalize_features: bool = True,
        random_seed: int = 52,
        device: str | None = None,
    ):
        self.lookback = lookback
        self.grouping_column = grouping_column
        self.pad_strategy = pad_strategy
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.head_hidden_1 = head_hidden_1
        self.head_hidden_2 = head_hidden_2
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.normalize_features = normalize_features
        self.random_seed = random_seed
        self.device = device

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

        def _pick_device(override: str | None) -> "torch.device":
            if override is not None:
                return torch.device(override)
            if torch.backends.mps.is_available():
                return torch.device("mps")
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")

        device = _pick_device(self.device)

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        feature_cols = [c for c in features.columns if c != self.grouping_column]
        feature_matrix = features[feature_cols].to_numpy(dtype=np.float32)
        y_matrix = target.to_numpy(dtype=np.float32)
        if y_matrix.ndim == 1:
            y_matrix = y_matrix.reshape(-1, 1)
        n_features = len(feature_cols)
        n_targets = y_matrix.shape[1]

        if self.normalize_features:
            feat_mean = feature_matrix.mean(axis=0).astype(np.float32)
            feat_std = feature_matrix.std(axis=0).astype(np.float32)
            feat_std[feat_std < 1e-8] = 1.0
        else:
            feat_mean = np.zeros(n_features, dtype=np.float32)
            feat_std = np.ones(n_features, dtype=np.float32)

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
        log.info("CNN training rows: %d", len(features))

        def _train_on(dev: "torch.device"):
            torch.manual_seed(self.random_seed)
            loader = DataLoader(
                TensorDataset(X, y),
                batch_size=self.batch_size,
                shuffle=True,
            )
            m = _CNN1D(
                n_features=n_features,
                lookback=self.lookback,
                hidden_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                head_hidden_1=self.head_hidden_1,
                head_hidden_2=self.head_hidden_2,
                dropout=self.dropout,
                n_targets=n_targets,
                feat_mean=feat_mean,
                feat_std=feat_std,
            )
            m = m.to(dev)
            optimizer = torch.optim.AdamW(
                m.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            steps_per_epoch = max(1, len(loader))
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                steps_per_epoch=steps_per_epoch,
                epochs=self.epochs,
            )
            loss_fn = nn.MSELoss()
            m.train()
            log.info(
                "CNN training: device=%s epochs=%d batches/epoch=%d",
                dev.type,
                self.epochs,
                steps_per_epoch,
            )
            for epoch_idx in range(self.epochs):
                epoch_start = time.time()
                loss_accum = torch.zeros((), device=dev)
                n_batches = 0
                for xb, yb in loader:
                    xb = xb.to(dev)
                    yb = yb.to(dev)
                    optimizer.zero_grad()
                    pred = m(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            m.parameters(), max_norm=self.grad_clip_norm
                        )
                    optimizer.step()
                    scheduler.step()
                    loss_accum += loss.detach()
                    n_batches += 1
                avg_loss = (loss_accum / max(1, n_batches)).item()
                log.info(
                    "  epoch %d/%d  loss=%.6f  elapsed=%.1fs",
                    epoch_idx + 1,
                    self.epochs,
                    avg_loss,
                    time.time() - epoch_start,
                )
            return m

        model = _train_on(device)

        def _has_nonfinite_params(m) -> bool:
            for p in m.parameters():
                if not torch.isfinite(p).all():
                    return True
            return False

        if device.type != "cpu" and _has_nonfinite_params(model):
            import warnings

            warnings.warn(
                f"CNN training on device={device.type} produced non-finite "
                "parameters; retraining on CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            device = torch.device("cpu")
            model = _train_on(device)

        model.eval()
        model = model.to("cpu")

        dummy = torch.zeros(1, self.lookback, n_features, dtype=torch.float32)
        with tempfile.TemporaryDirectory() as td:
            onnx_path = Path(td) / "cnn.onnx"
            torch.onnx.export(
                model,
                (dummy,),
                onnx_path,
                input_names=[ONNX_INPUT_NAME],
                output_names=["output"],
                dynamic_axes={ONNX_INPUT_NAME: {0: "batch"}, "output": {0: "batch"}},
                opset_version=17,
                dynamo=False,
            )
            onnx_proto = onnx.load_from_string(onnx_path.read_bytes())

        import onnxruntime as _rt

        rng = np.random.default_rng(self.random_seed)
        sanity_x = rng.standard_normal((4, self.lookback, n_features)).astype(
            np.float32
        )
        with torch.no_grad():
            torch_out = model(torch.from_numpy(sanity_x)).cpu().numpy()
        onnx_sess = _rt.InferenceSession(
            onnx_proto.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        onnx_out = onnx_sess.run(None, {ONNX_INPUT_NAME: sanity_x})[0]
        max_abs = float(np.max(np.abs(torch_out - onnx_out)))
        if max_abs > 1e-4:
            raise RuntimeError(
                f"ONNX export drift exceeded tolerance: max|Δ|={max_abs:.2e}"
            )

        return ONNXEstimator(
            onnx_proto,
            input_spec=InputSpec(
                lookback=self.lookback,
                grouping_column=self.grouping_column,
                pad_strategy=self.pad_strategy,
            ),
        )


def _CNN1D(
    n_features: int,
    lookback: int,
    hidden_channels: int,
    kernel_size: int,
    head_hidden_1: int,
    head_hidden_2: int,
    dropout: float,
    n_targets: int,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
):
    """A 1D CNN that maps (N, lookback, F) → (N, n_targets).

    Architecture mirrors the Conv1DModel used in the autoresearch training
    sweep: three Conv1d → ReLU blocks with ``hidden_channels`` channels each,
    followed by a fully-connected head ``conv_out → head_hidden_1 →
    head_hidden_2 → n_targets`` with ReLU activations and a single dropout
    between the first two linear layers. Feature mean/std are baked into the
    graph as non-trainable buffers so the exported ONNX normalizes inputs at
    inference time.
    """
    import torch
    from torch import nn

    padding = kernel_size // 2
    mean_tensor = torch.from_numpy(feat_mean.astype(np.float32)).view(1, n_features, 1)
    std_tensor = torch.from_numpy(feat_std.astype(np.float32)).view(1, n_features, 1)

    class CNN(nn.Module):
        feat_mean: torch.Tensor
        feat_std: torch.Tensor

        def __init__(self):
            super().__init__()
            self.register_buffer("feat_mean", mean_tensor)
            self.register_buffer("feat_std", std_tensor)
            self.conv = nn.Sequential(
                nn.Conv1d(n_features, hidden_channels, kernel_size, padding=padding),
                nn.ReLU(),
                nn.Conv1d(
                    hidden_channels, hidden_channels, kernel_size, padding=padding
                ),
                nn.ReLU(),
                nn.Conv1d(
                    hidden_channels, hidden_channels, kernel_size, padding=padding
                ),
                nn.ReLU(),
            )
            conv_out_dim = hidden_channels * lookback
            self.head = nn.Sequential(
                nn.Linear(conv_out_dim, head_hidden_1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_1, head_hidden_2),
                nn.ReLU(),
                nn.Linear(head_hidden_2, n_targets),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (N, lookback, F) → (N, F, lookback) for Conv1d
            x = x.transpose(1, 2)
            x = (x - self.feat_mean) / self.feat_std
            x = self.conv(x)
            x = x.flatten(start_dim=1)
            return self.head(x)

    return CNN()
