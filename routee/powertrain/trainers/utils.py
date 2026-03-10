import numpy as np
import pandas as pd


def test_train_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 123):
    """
    Split a dataframe into training and testing sets
    """
    np.random.seed(seed)
    mask = np.random.rand(len(df)) < (1 - test_size)
    train_df = df[mask]
    test_df = df[~mask]
    return train_df, test_df
