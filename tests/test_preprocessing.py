"""Tests for Preprocessing and Data Pipeline.

Author: Isbandu Rahuman R.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import EXPECTED_FEATURES, TARGET_COLUMN
from src.preprocessing import (
    FraudFeaturePreprocessor,
    fit_and_transform_pipeline,
    load_preprocessor,
    save_preprocessor,
    split_data,
)


@pytest.fixture
def mock_dataset():
    """Create a minimal synthetic DataFrame mimicking the credit card schema."""
    np.random.seed(42)
    n_samples = 200
    data = {"Time": np.linspace(0, 1000, n_samples)}
    for i in range(1, 29):
        data[f"V{i}"] = np.random.randn(n_samples)
    data["Amount"] = np.random.exponential(scale=50, size=n_samples)
    # 5% fraud rate for test fixture
    classes = np.zeros(n_samples, dtype=int)
    classes[:10] = 1
    np.random.shuffle(classes)
    data[TARGET_COLUMN] = classes
    return pd.DataFrame(data)


def test_preprocessor_fit_transform(mock_dataset):
    """Test fitting and transforming features without errors."""
    X = mock_dataset[EXPECTED_FEATURES]
    preprocessor = FraudFeaturePreprocessor()
    transformed = preprocessor.fit_transform(X)

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == (200, 30)
    assert preprocessor.is_fitted is True


def test_preprocessor_feature_ordering(mock_dataset):
    """Verify preprocessor produces expected column ordering."""
    preprocessor = FraudFeaturePreprocessor()
    preprocessor.fit(mock_dataset[EXPECTED_FEATURES])
    feature_names = preprocessor.get_feature_names_out()

    assert feature_names[0] == "scaled_time"
    assert feature_names[-1] == "scaled_amount"
    assert len(feature_names) == 30


def test_no_data_leakage(mock_dataset):
    """Verify that transformation on test data uses strictly training scaler parameters."""
    X_train, X_test, y_train, y_test = split_data(mock_dataset, test_size=0.25, random_state=42)

    preprocessor, X_train_proc, X_test_proc = fit_and_transform_pipeline(X_train, X_test)

    # Train shape: 150x30, Test shape: 50x30
    assert X_train_proc.shape == (150, 30)
    assert X_test_proc.shape == (50, 30)

    # Scaler center and scale must match X_train
    assert preprocessor.amount_scaler.center_[0] == pytest.approx(X_train["Amount"].median(), rel=1e-2)


def test_save_and_load_preprocessor(tmp_path, mock_dataset):
    """Test persisting and reloading fitted preprocessor."""
    save_path = tmp_path / "test_preprocessor.joblib"
    X = mock_dataset[EXPECTED_FEATURES]

    prep = FraudFeaturePreprocessor().fit(X)
    save_preprocessor(prep, save_path)

    loaded = load_preprocessor(save_path)
    assert loaded.is_fitted is True
    assert np.allclose(prep.transform(X), loaded.transform(X))


def test_stratified_split_preserves_class_ratio(mock_dataset):
    """Ensure stratified split maintains the minority class proportion."""
    X_train, X_test, y_train, y_test = split_data(mock_dataset, test_size=0.20, random_state=42)

    train_fraud_ratio = y_train.mean()
    test_fraud_ratio = y_test.mean()

    assert abs(train_fraud_ratio - test_fraud_ratio) < 0.02
