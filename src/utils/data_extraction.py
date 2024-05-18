import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def create_train_test(df: pd.DataFrame):
    """
    Args:
      df: Pandas DataFrame obtained from GTZAN dataset.

    Returns:
      Training and testing datasets.
    """
    # df[0] is filename, df[1] is length, df[-1] is genre
    X = df.iloc[:, 2:-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test


def create_train_test_concat(df: pd.DataFrame):
    """
    Args:
      df: Pandas DataFrame obtained from GTZAN dataset.

    Returns:
      Training and testing datasets, with features for every 10 3-second
      song excerpts merged together into one row.
    """
    # df[0] is filename, df[1] is length, df[-1] is genre
    X = df.iloc[:, 2:-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    X = X.reshape(999, -1)
    y_reshaped = y.reshape(999, 10)
    y = y_reshaped[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test
