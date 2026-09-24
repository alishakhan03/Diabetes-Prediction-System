"""Data loading, cleaning and splitting helpers."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_data(path=config.DATA_PATH) -> pd.DataFrame:
    """Load the Pima Indians Diabetes dataset and validate its schema."""
    df = pd.read_csv(path)
    expected = set(config.FEATURES + [config.TARGET])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return df


def mark_invalid_as_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Replace physiologically impossible zeros (e.g. Glucose = 0) with NaN.

    The values are imputed later, inside the model Pipeline, so that the
    medians are learned from the training data only (no data leakage).
    """
    df = df.copy()
    df[config.ZERO_INVALID_COLS] = df[config.ZERO_INVALID_COLS].replace(0, np.nan)
    return df


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise missing / invalid values per column."""
    report = pd.DataFrame({
        "missing_values": df.isna().sum(),
        "zero_values": (df == 0).sum(),
    })
    report["zero_is_invalid"] = report.index.isin(config.ZERO_INVALID_COLS)
    return report


def get_train_test(df: pd.DataFrame):
    """Clean the data and return a stratified train/test split."""
    df = mark_invalid_as_nan(df).drop_duplicates()
    X, y = df[config.FEATURES], df[config.TARGET]
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
