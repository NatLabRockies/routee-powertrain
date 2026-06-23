from __future__ import annotations

import logging
import time
from typing import Literal

import numpy as np
import onnxruntime as rt
import pandas as pd
import torch

from routee.powertrain.core.model_config import ModelConfig
from routee.powertrain.estimators.estimator_interface import Estimator, InputSpec
from routee.powertrain.estimators.onnx import ONNX_INPUT_NAME, ONNXEstimator
from routee.powertrain.trainers.trainer import Trainer
from routee.powertrain.utils.threading import get_restricted_threads

log = logging.getLogger(__name__)

DEFAULT_LOOKBACK = 5
DEFAULT_GROUPING_COLUMN = "route_id"


def _auto_n_conv_layers(lookback: int, kernel_size: int) -> int:
    """Pick the most VALID-padded conv layers that fit a given lookback.

    Each kernel-K VALID conv shrinks the sequence by K-1. We stack until the
    next layer would not have enough input. This makes the receptive field at
    the head match (or as close as possible to) the lookback window.
    """
    n = lookback
    layers = 0
    while n - kernel_size + 1 >= 1:
        n = n - kernel_size + 1
        layers += 1
    return max(1, layers)


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
        pad_strategy: Literal["zero", "repeat_first"] = "zero",
        hidden_channels: int = 64,
        kernel_size: int = 3,
        n_conv_layers: int | None = None,
        epochs: int = 50,
        batch_size: int = 4096,
        learning_rate: float = 1e-3,
        min_learning_rate: float = 1e-5,
        weight_decay: float = 1e-4,
        grad_clip_norm: float = 1.0,
        normalize_features: bool = True,
        random_seed: int = 52,
        device: str | None = None,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 1e-6,
        warmup_epochs: int = 0,
    ):
        self.lookback = lookback
        self.grouping_column = grouping_column
        self.pad_strategy = pad_strategy
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_conv_layers = (
            n_conv_layers
            if n_conv_layers is not None
            else _auto_n_conv_layers(lookback, kernel_size)
        )
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.normalize_features = normalize_features
        self.random_seed = random_seed
        self.device = device
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.warmup_epochs = warmup_epochs

    def inner_train(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame,
        config: ModelConfig,
        test_features: pd.DataFrame | None = None,
        test_target: pd.DataFrame | None = None,
        **kwargs: object,
    ) -> Estimator:
        try:
            import torch
            from torch import nn
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
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
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

        def _build_windows(df: pd.DataFrame, matrix: np.ndarray) -> np.ndarray:
            windows = np.zeros((len(df), self.lookback, n_features), dtype=np.float32)
            groups = df.groupby(self.grouping_column, sort=False).indices
            for idx_array in groups.values():
                idx_array = np.asarray(idx_array)
                group_features = matrix[idx_array]
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
            return windows

        windows = _build_windows(features, feature_matrix)

        X_cpu = torch.from_numpy(windows)
        y_cpu = torch.from_numpy(y_matrix)
        X_val_cpu: torch.Tensor | None = None
        y_val_cpu: torch.Tensor | None = None
        if test_features is not None and test_target is not None:
            if self.grouping_column not in test_features.columns:
                raise ValueError(
                    f"CNNTrainer requires a '{self.grouping_column}' column in the "
                    f"validation features so windows can be built per route."
                )
            test_matrix = test_features[feature_cols].to_numpy(dtype=np.float32)
            y_val_matrix = test_target.to_numpy(dtype=np.float32)
            if y_val_matrix.ndim == 1:
                y_val_matrix = y_val_matrix.reshape(-1, 1)
            X_val_cpu = torch.from_numpy(_build_windows(test_features, test_matrix))
            y_val_cpu = torch.from_numpy(y_val_matrix)

        n_train = X_cpu.shape[0]
        log.info("CNN training rows: %d", n_train)

        def _train_on(dev: "torch.device"):
            torch.manual_seed(self.random_seed)
            # Move the full training tensors to the target device once.
            X_dev = X_cpu.to(dev)
            y_dev = y_cpu.to(dev)
            X_val_dev = X_val_cpu.to(dev) if X_val_cpu is not None else None
            y_val_dev = y_val_cpu.to(dev) if y_val_cpu is not None else None
            m = _CNN1D(
                n_features=n_features,
                hidden_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                n_conv_layers=self.n_conv_layers,
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
            steps_per_epoch = max(1, (n_train + self.batch_size - 1) // self.batch_size)
            warmup_steps = self.warmup_epochs * steps_per_epoch
            cosine_epochs = max(1, self.epochs - self.warmup_epochs)
            cosine_steps = max(1, cosine_epochs * steps_per_epoch)
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cosine_steps,
                eta_min=self.min_learning_rate,
            )
            if warmup_steps > 0:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1e-8,
                    total_iters=warmup_steps,
                )
                scheduler = torch.optim.lr_scheduler.ChainedScheduler(
                    [warmup_scheduler, cosine_scheduler]
                )
            else:
                scheduler = cosine_scheduler
            loss_fn = nn.MSELoss()
            m.train()
            log.info(
                "CNN training: device=%s epochs=%d batches/epoch=%d hidden=%d conv_layers=%d",
                dev.type,
                self.epochs,
                steps_per_epoch,
                self.hidden_channels,
                self.n_conv_layers,
            )
            rng = np.random.default_rng(self.random_seed)
            best_val_loss = float("inf")
            best_state: dict | None = None
            patience_counter = 0
            stopped_early = False
            for epoch_idx in range(self.epochs):
                epoch_start = time.time()
                perm = torch.from_numpy(rng.permutation(n_train)).to(dev)
                loss_accum = torch.zeros((), device=dev)
                n_batches = 0
                for s in range(0, n_train, self.batch_size):
                    b = perm[s : s + self.batch_size]
                    xb = X_dev[b]
                    yb = y_dev[b]
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
                val_loss = float("nan")
                if X_val_dev is not None and y_val_dev is not None:
                    m.eval()
                    with torch.no_grad():
                        val_loss = loss_fn(m(X_val_dev), y_val_dev).item()
                    m.train()
                    if self.early_stopping_patience is not None:
                        if val_loss < best_val_loss - self.early_stopping_min_delta:
                            best_val_loss = val_loss
                            best_state = {
                                k: v.clone() for k, v in m.state_dict().items()
                            }
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        if patience_counter >= self.early_stopping_patience:
                            log.info(
                                "  Early stopping at epoch %d/%d (no improvement for %d epochs)",
                                epoch_idx + 1,
                                self.epochs,
                                self.early_stopping_patience,
                            )
                            stopped_early = True
                log.info(
                    "  epoch %d/%d  loss=%.6f  val_loss=%.6f  lr=%.2e  elapsed=%.1fs",
                    epoch_idx + 1,
                    self.epochs,
                    avg_loss,
                    val_loss,
                    optimizer.param_groups[0]["lr"],
                    time.time() - epoch_start,
                )
                if stopped_early:
                    break
            if best_state is not None:
                m.load_state_dict(best_state)
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
        batch_dim = torch.export.Dim("batch")
        onnx_program = torch.onnx.export(
            model,
            (dummy,),
            input_names=[ONNX_INPUT_NAME],
            output_names=["output"],
            dynamic_shapes={"x": {0: batch_dim}},
            opset_version=18,
            dynamo=True,
        )
        assert onnx_program is not None  # dynamo=True always returns an ONNXProgram
        onnx_proto = onnx_program.model_proto

        rng = np.random.default_rng(self.random_seed)
        sanity_x = rng.standard_normal((4, self.lookback, n_features)).astype(
            np.float32
        )
        with torch.no_grad():
            torch_out = model(torch.from_numpy(sanity_x)).cpu().numpy()

        sess_options = rt.SessionOptions()
        restricted_threads = get_restricted_threads()
        if restricted_threads is not None:
            sess_options.intra_op_num_threads = restricted_threads
        onnx_sess = rt.InferenceSession(
            onnx_proto.SerializeToString(),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
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
    hidden_channels: int,
    kernel_size: int,
    n_conv_layers: int,
    n_targets: int,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
):
    """A 1D CNN that maps (N, T, F) → (N, n_targets) for sequence length T.

    Stack of ``n_conv_layers`` VALID-padded kernel-K Conv1D → ReLU layers, an
    adaptive average pool over any remaining sequence positions, and a single
    linear head ``hidden → n_targets``.
    """
    import torch
    from torch import nn

    mean_tensor = torch.from_numpy(feat_mean.astype(np.float32)).view(1, n_features, 1)
    std_tensor = torch.from_numpy(feat_std.astype(np.float32)).view(1, n_features, 1)

    class CNN(nn.Module):
        feat_mean: torch.Tensor
        feat_std: torch.Tensor

        def __init__(self):
            super().__init__()
            self.register_buffer("feat_mean", mean_tensor)
            self.register_buffer("feat_std", std_tensor)
            layers: list[nn.Module] = []
            in_ch = n_features
            for _ in range(n_conv_layers):
                layers.append(nn.Conv1d(in_ch, hidden_channels, kernel_size, padding=0))
                layers.append(nn.ReLU())
                in_ch = hidden_channels
            self.conv = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(hidden_channels, n_targets)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (N, T, F) → (N, F, T) for Conv1d
            x = x.transpose(1, 2)
            x = (x - self.feat_mean) / self.feat_std
            x = self.conv(x)
            x = self.pool(x).squeeze(-1)
            return self.head(x)

    return CNN()
