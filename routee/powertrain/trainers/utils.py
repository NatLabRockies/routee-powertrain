from __future__ import annotations

import numpy as np
import pandas as pd


def test_train_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 123,
    grouping_column: str | None = None,
):
    """
    Split a dataframe into training and testing sets.

    When ``grouping_column`` is provided, the split operates on unique values
    of that column so that every row belonging to a given group lands
    entirely in train or entirely in test. This is required for sequence
    models whose inputs are built from consecutive rows of the same group
    (e.g. the 1D CNN's per-route lookback windows) — a plain row-level split
    would leave each group with gapped rows and destroy the sequence.
    """
    np.random.seed(seed)
    if grouping_column is None:
        mask = np.random.rand(len(df)) < (1 - test_size)
    else:
        unique_groups = df[grouping_column].unique()
        shuffled = np.random.permutation(unique_groups)
        n_train = int(round(len(shuffled) * (1 - test_size)))
        train_groups = set(shuffled[:n_train].tolist())
        mask = df[grouping_column].isin(train_groups).to_numpy()
    train_df = df[mask]
    test_df = df[~mask]
    return train_df, test_df
