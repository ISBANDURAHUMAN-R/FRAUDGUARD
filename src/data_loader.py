"""Data Loading and Validation Module for FRAUDGUARD.

Author: Isbandu Rahuman R.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

EXPECTED_FEATURES: List[str] = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount",
]
TARGET_COLUMN: str = "Class"
DEFAULT_DATA_PATH = Path("data") / "creditcard.csv"
DEFAULT_PARQUET_PATH = Path("data") / "creditcard.parquet"

# In-memory DataFrame singleton cache to ensure instant UI rendering
_CACHED_DATASET: Optional[pd.DataFrame] = None


class DatasetValidationError(ValueError):
    """Raised when uploaded or loaded dataset fails validation checks."""
    pass


def is_dataset_available(path: Optional[Union[str, Path]] = None) -> bool:
    """Check whether creditcard dataset exists on disk (CSV or fast Parquet)."""
    if path is None:
        if DEFAULT_PARQUET_PATH.exists() and DEFAULT_PARQUET_PATH.stat().st_size > 0:
            return True
        return DEFAULT_DATA_PATH.exists() and DEFAULT_DATA_PATH.is_file() and DEFAULT_DATA_PATH.stat().st_size > 0

    target_path = Path(path)
    return target_path.exists() and target_path.is_file() and target_path.stat().st_size > 0


def validate_dataset(
    df: pd.DataFrame,
    require_target: bool = True,
    allow_extra_columns: bool = True
) -> Tuple[bool, List[str]]:
    """Validate DataFrame against European Credit Card schema.

    Args:
        df: Input DataFrame to validate.
        require_target: Whether the 'Class' column must be present.
        allow_extra_columns: Whether non-standard extra columns are permitted.

    Returns:
        Tuple of (is_valid, list_of_error_messages).
    """
    errors: List[str] = []

    if df is None or not isinstance(df, pd.DataFrame):
        return False, ["Input is not a valid pandas DataFrame."]

    if df.empty:
        return False, ["Dataset is empty (0 rows)."]

    # Check required feature columns
    missing_features = [col for col in EXPECTED_FEATURES if col not in df.columns]
    if missing_features:
        errors.append(f"Missing required feature columns: {', '.join(missing_features)}")

    # Check target column if required
    if require_target and TARGET_COLUMN not in df.columns:
        errors.append(f"Missing required target column '{TARGET_COLUMN}'.")

    # Validate data types for available expected columns
    for col in EXPECTED_FEATURES:
        if col in df.columns:
            if not np.issubdtype(df[col].dtype, np.number):
                errors.append(f"Column '{col}' must contain numerical data (found {df[col].dtype}).")

    if require_target and TARGET_COLUMN in df.columns:
        unique_classes = set(df[TARGET_COLUMN].dropna().unique())
        if not unique_classes.issubset({0, 1}):
            errors.append(
                f"Target column '{TARGET_COLUMN}' must contain binary values (0 or 1). Found: {unique_classes}"
            )

    return len(errors) == 0, errors


def load_dataset(
    path: Optional[Union[str, Path]] = None,
    validate: bool = True,
    use_cache: bool = True
) -> pd.DataFrame:
    """Load the European Credit Card fraud dataset from RAM cache, Parquet, or CSV.

    Args:
        path: Optional path to creditcard.csv. Defaults to data/creditcard.csv.
        validate: Whether to run schema validation.
        use_cache: Whether to use and update the in-memory RAM singleton.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the dataset file is missing.
        DatasetValidationError: If validation fails.
    """
    global _CACHED_DATASET
    if use_cache and path is None and _CACHED_DATASET is not None:
        return _CACHED_DATASET

    target_path = Path(path) if path else DEFAULT_DATA_PATH
    parquet_path = DEFAULT_PARQUET_PATH if path is None else None

    # Load from high-speed binary Parquet if available
    if parquet_path and parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif target_path.exists():
        df = pd.read_csv(target_path)
        # Automatically cache to Parquet format for fast subsequent cold starts
        if path is None:
            try:
                DEFAULT_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(DEFAULT_PARQUET_PATH, index=False)
            except Exception:
                pass
    else:
        raise FileNotFoundError(
            f"Dataset not found at '{target_path}'. Please download 'creditcard.csv' "
            "from Kaggle (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) "
            "and place it in the 'data/' directory, or upload it via the dashboard."
        )

    if validate:
        is_valid, errors = validate_dataset(df, require_target=True)
        if not is_valid:
            raise DatasetValidationError("; ".join(errors))

    if path is None and use_cache:
        _CACHED_DATASET = df

    return df


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Union[int, float, Dict]]:
    """Compute summary statistics for exploratory analysis.

    Args:
        df: Validated credit card DataFrame.

    Returns:
        Dictionary with dataset metrics.
    """
    total_records = int(len(df))
    missing_values = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    summary: Dict[str, Union[int, float, Dict]] = {
        "total_records": total_records,
        "total_features": int(len(df.columns)),
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
    }

    if TARGET_COLUMN in df.columns:
        class_counts = df[TARGET_COLUMN].value_counts().to_dict()
        legit_count = int(class_counts.get(0, 0))
        fraud_count = int(class_counts.get(1, 0))
        fraud_rate = float(fraud_count / total_records * 100) if total_records > 0 else 0.0

        summary["legitimate_count"] = legit_count
        summary["fraudulent_count"] = fraud_count
        summary["fraud_rate_pct"] = round(fraud_rate, 4)

    if "Amount" in df.columns:
        summary["amount_stats"] = {
            "min": float(df["Amount"].min()),
            "max": float(df["Amount"].max()),
            "mean": float(df["Amount"].mean()),
            "median": float(df["Amount"].median()),
            "std": float(df["Amount"].std()),
        }

    return summary
