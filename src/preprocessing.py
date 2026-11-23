"""Data Preprocessing and Pipeline Module for FRAUDGUARD.

Author: Isbandu Rahuman R.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.data_loader import EXPECTED_FEATURES, TARGET_COLUMN


class FraudFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Custom preprocessor for credit card fraud data.

    Scales 'Time' and 'Amount' using RobustScaler (resistant to financial outliers).
    Leaves pre-computed PCA components (V1 - V28) unscaled to prevent distortion.
    """

    def __init__(self):
        self.amount_scaler = RobustScaler()
        self.time_scaler = RobustScaler()
        self.feature_names = list(EXPECTED_FEATURES)
        self.is_fitted = False

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[np.ndarray] = None):
        """Fit scalers strictly on training data."""
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = X.copy()

        self.amount_scaler.fit(X_df[["Amount"]])
        self.time_scaler.fit(X_df[["Time"]])
        self.is_fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform feature matrix using fitted scalers."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")

        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = X.copy()

        X_df["scaled_amount"] = self.amount_scaler.transform(X_df[["Amount"]])
        X_df["scaled_time"] = self.time_scaler.transform(X_df[["Time"]])

        # Assemble feature array in order: scaled_time, V1..V28, scaled_amount
        v_cols = [f"V{i}" for i in range(1, 29)]
        ordered_cols = ["scaled_time"] + v_cols + ["scaled_amount"]
        return X_df[ordered_cols].to_numpy(dtype=np.float64)

    def get_feature_names_out(self) -> list:
        v_cols = [f"V{i}" for i in range(1, 29)]
        return ["scaled_time"] + v_cols + ["scaled_amount"]


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a stratified train/test split to maintain class ratio.

    Args:
        df: Full creditcard DataFrame.
        test_size: Proportion of test sample (default 0.20).
        random_state: Reproducibility seed (default 42).

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df[EXPECTED_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def fit_and_transform_pipeline(
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None
) -> Tuple[FraudFeaturePreprocessor, np.ndarray, Optional[np.ndarray]]:
    """Fit preprocessor only on X_train and transform both splits.

    Guarantees no data leakage from test split.
    """
    preprocessor = FraudFeaturePreprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)

    X_test_transformed = None
    if X_test is not None:
        X_test_transformed = preprocessor.transform(X_test)

    return preprocessor, X_train_transformed, X_test_transformed


def save_preprocessor(preprocessor: FraudFeaturePreprocessor, path: Union[str, Path]) -> None:
    """Save fitted preprocessor pipeline to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)


def load_preprocessor(path: Union[str, Path]) -> FraudFeaturePreprocessor:
    """Load fitted preprocessor pipeline from disk."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Preprocessor file not found at '{path}'.")
    return joblib.load(path)
