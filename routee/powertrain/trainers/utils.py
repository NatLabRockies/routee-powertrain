from __future__ import annotations

import numpy as np
import pandas as pd


def test_train_validation_split(
    df: pd.DataFrame,
    test_size: float = 0.1,
    validation_size: float = 0.1,
    seed: int = 123,
    grouping_column: str | None = None,
):
    """
    Split a dataframe into testing, training, and validation sets.

    When ``grouping_column`` is provided, the split operates on unique values
    of that column so that every row belonging to a given group lands
    entirely in test, training, or validation. This is required for sequence
    models whose inputs are built from consecutive rows of the same group
    (e.g. the 1D CNN's per-route lookback windows) — a plain row-level split
    would leave each group with gapped rows and destroy the sequence.

    Returns a tuple of three dataframes: (train, validation, test).
    """
    if not 0 <= test_size < 1:
        raise ValueError("test_size must be in the range [0, 1)")
    if not 0 <= validation_size < 1:
        raise ValueError("validation_size must be in the range [0, 1)")
    if test_size + validation_size >= 1:
        raise ValueError(
            "test_size + validation_size must be less than 1 "
            "to leave room for training data"
        )

    np.random.seed(seed)
    train_size = 1 - test_size - validation_size

    if grouping_column is None:
        draw = np.random.rand(len(df))
        train_mask = draw < train_size
        val_mask = (draw >= train_size) & (draw < train_size + validation_size)
    else:
        unique_groups = df[grouping_column].unique()
        shuffled = np.random.permutation(unique_groups)
        n_train = int(round(len(shuffled) * train_size))
        n_val = int(round(len(shuffled) * validation_size))
        train_groups = set(shuffled[:n_train].tolist())
        val_groups = set(shuffled[n_train : n_train + n_val].tolist())
        train_mask = df[grouping_column].isin(train_groups).to_numpy()
        val_mask = df[grouping_column].isin(val_groups).to_numpy()

    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[~(train_mask | val_mask)]
    return train_df, val_df, test_df
