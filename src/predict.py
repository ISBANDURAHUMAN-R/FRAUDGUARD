"""Inference and Prediction Engine for FRAUDGUARD.

Author: Isbandu Rahuman R.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd

from src.data_loader import EXPECTED_FEATURES
from src.preprocessing import FraudFeaturePreprocessor

DEFAULT_MODEL_PATH = Path("models") / "fraudguard_model.joblib"
DEFAULT_PREPROCESSOR_PATH = Path("models") / "preprocessor.joblib"
DEFAULT_METADATA_PATH = Path("models") / "model_metadata.joblib"

# In-memory artifact cache to guarantee instant sub-millisecond execution
_CACHED_ARTIFACTS: Optional[Tuple[Any, FraudFeaturePreprocessor, Dict[str, Any]]] = None

MODEL_DISCLAIMER = (
    "DISCLAIMER: FRAUDGUARD predictions are probabilistic machine learning estimates "
    "intended for decision support and risk triage. They do not constitute definitive "
    "legal or financial confirmations of fraud. Suspicious transactions should undergo "
    "manual verification by compliance officers."
)


class ModelNotTrainedError(FileNotFoundError):
    """Raised when trying to predict before training or downloading the model."""
    pass


def load_artifacts(
    model_path: Optional[Union[str, Path]] = None,
    preprocessor_path: Optional[Union[str, Path]] = None,
    metadata_path: Optional[Union[str, Path]] = None,
    reload: bool = False
) -> Tuple[Any, FraudFeaturePreprocessor, Dict[str, Any]]:
    """Load model, preprocessor, and metadata from disk with high-speed in-memory caching."""
    global _CACHED_ARTIFACTS
    if not reload and _CACHED_ARTIFACTS is not None and model_path is None:
        return _CACHED_ARTIFACTS

    m_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    p_path = Path(preprocessor_path) if preprocessor_path else DEFAULT_PREPROCESSOR_PATH
    meta_path = Path(metadata_path) if metadata_path else DEFAULT_METADATA_PATH

    if not m_path.exists() or not p_path.exists():
        raise ModelNotTrainedError(
            "Trained model artifacts not found. Please train the model using "
            "'python -m src.train_model' or via the Model Insights page."
        )

    model = joblib.load(m_path)
    preprocessor = joblib.load(p_path)
    metadata = joblib.load(meta_path) if meta_path.exists() else {}

    artifacts = (model, preprocessor, metadata)
    if model_path is None:
        _CACHED_ARTIFACTS = artifacts

    return artifacts


def determine_risk_level(prob: float, threshold: float = 0.50) -> Tuple[str, str]:
    """Map fraud probability to a human-readable risk level and color indicator.

    Returns:
        Tuple of (risk_level_name, color_hex).
    """
    if prob < 0.20:
        return "LOW", "#00e676"  # Emerald green
    elif prob < threshold:
        return "MEDIUM", "#ffb300"  # Amber orange
    elif prob < 0.80:
        return "HIGH", "#ff5252"  # Coral red
    else:
        return "CRITICAL", "#d50000"  # Crimson alert


def predict_single(
    features_input: Union[Dict[str, float], pd.DataFrame, pd.Series],
    model: Optional[Any] = None,
    preprocessor: Optional[FraudFeaturePreprocessor] = None,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """Perform fraud prediction on a single transaction.

    Args:
        features_input: Dictionary, Series, or 1-row DataFrame containing the 30 features.
        model: Optional pre-loaded model.
        preprocessor: Optional pre-loaded preprocessor.
        threshold: Optional custom decision threshold.

    Returns:
        Structured prediction dictionary with probability, label, risk level, and metadata.
    """
    if model is None or preprocessor is None:
        model, preprocessor, metadata = load_artifacts()
        if threshold is None:
            threshold = float(metadata.get("selected_threshold", 0.50))
    else:
        if threshold is None:
            threshold = 0.50

    if isinstance(features_input, dict):
        df_in = pd.DataFrame([features_input])
    elif isinstance(features_input, pd.Series):
        df_in = pd.DataFrame([features_input.to_dict()])
    else:
        df_in = features_input.copy()

    # Schema validation
    missing = [c for c in EXPECTED_FEATURES if c not in df_in.columns]
    if missing:
        raise ValueError(f"Missing required transaction features: {', '.join(missing)}")

    # Ensure numerical dtypes and coerce NaNs
    for c in EXPECTED_FEATURES:
        df_in[c] = pd.to_numeric(df_in[c], errors="coerce")
        if df_in[c].isna().any():
            raise ValueError(f"Feature '{c}' contains invalid or non-numeric values.")

    # Feature transformation
    X_proc = preprocessor.transform(df_in[EXPECTED_FEATURES])

    # Probability estimation
    prob = float(model.predict_proba(X_proc)[0, 1])
    is_fraud = bool(prob >= threshold)
    label = "Fraudulent" if is_fraud else "Legitimate"
    risk_level, risk_color = determine_risk_level(prob, threshold)

    amount = float(df_in["Amount"].iloc[0])
    time_sec = float(df_in["Time"].iloc[0])

    # Highlight top anomalous PCA features (components exceeding +/- 3 std deviations)
    anomalies = []
    for c in [f"V{i}" for i in range(1, 29)]:
        val = float(df_in[c].iloc[0])
        if abs(val) >= 3.0:
            anomalies.append(f"{c}={val:+.2f}")

    return {
        "is_fraud": is_fraud,
        "prediction_label": label,
        "fraud_probability": round(prob, 4),
        "fraud_probability_pct": round(prob * 100, 2),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "threshold_used": round(threshold, 4),
        "amount": amount,
        "time_sec": time_sec,
        "notable_features": anomalies[:5],
        "disclaimer": MODEL_DISCLAIMER,
    }


def predict_bulk(
    df: pd.DataFrame,
    model: Optional[Any] = None,
    preprocessor: Optional[FraudFeaturePreprocessor] = None,
    threshold: Optional[float] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Perform vectorized batch inference on a DataFrame of transactions.

    Args:
        df: Input DataFrame with expected features.
        model: Optional pre-loaded model.
        preprocessor: Optional pre-loaded preprocessor.
        threshold: Optional decision threshold.

    Returns:
        Tuple of (annotated_dataframe, summary_statistics_dict).
    """
    if model is None or preprocessor is None:
        model, preprocessor, metadata = load_artifacts()
        if threshold is None:
            threshold = float(metadata.get("selected_threshold", 0.50))
    else:
        if threshold is None:
            threshold = 0.50

    missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Bulk CSV missing required features: {', '.join(missing)}")

    # Preprocess
    df_clean = df[EXPECTED_FEATURES].copy()
    for col in EXPECTED_FEATURES:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0.0)

    X_proc = preprocessor.transform(df_clean)
    probs = model.predict_proba(X_proc)[:, 1]

    # Annotate results
    df_result = df.copy()
    df_result["Fraud_Probability"] = np.round(probs, 4)
    df_result["Prediction"] = np.where(probs >= threshold, "Fraudulent", "Legitimate")

    # Map risk levels
    risk_levels = []
    for p in probs:
        r, _ = determine_risk_level(p, threshold)
        risk_levels.append(r)
    df_result["Risk_Level"] = risk_levels

    # Compute summary
    total = len(df_result)
    fraud_count = int((df_result["Prediction"] == "Fraudulent").sum())
    legit_count = total - fraud_count
    fraud_rate = float(fraud_count / total * 100) if total > 0 else 0.0

    summary = {
        "total_analyzed": total,
        "fraud_count": fraud_count,
        "legit_count": legit_count,
        "fraud_rate_pct": round(fraud_rate, 2),
        "avg_fraud_probability": round(float(np.mean(probs)), 4),
        "threshold_used": round(threshold, 4),
        "risk_breakdown": df_result["Risk_Level"].value_counts().to_dict(),
    }

    return df_result, summary


def get_demo_samples() -> Dict[str, Dict[str, float]]:
    """Return authentic representative transaction samples for instant UI evaluation."""
    # Extracted directly from genuine European credit card fraud records
    fraud_sample = {
        "Time": 406.0,
        "V1": -2.312226542, "V2": 1.951992011, "V3": -1.609850732, "V4": 3.997905588,
        "V5": -0.522187865, "V6": -1.426545319, "V7": -2.537387304, "V8": 1.391657248,
        "V9": -2.770089277, "V10": -2.772272145, "V11": 3.202033207, "V12": -2.899907388,
        "V13": -0.595221881, "V14": -4.289253782, "V15": 0.38972412, "V16": -1.14074718,
        "V17": -2.830055675, "V18": -0.016822468, "V19": 0.416955705, "V20": 0.126910559,
        "V21": 0.517232371, "V22": -0.035049369, "V23": -0.465211076, "V24": 0.320198199,
        "V25": 0.044519167, "V26": 0.177839798, "V27": 0.261145003, "V28": -0.143275875,
        "Amount": 0.0
    }

    legit_sample = {
        "Time": 0.0,
        "V1": -1.359807133, "V2": -0.072781173, "V3": 2.536346738, "V4": 1.378155224,
        "V5": -0.33832077, "V6": 0.462387778, "V7": 0.239598554, "V8": 0.098697901,
        "V9": 0.36378697, "V10": 0.090794172, "V11": -0.551599533, "V12": -0.617800856,
        "V13": -0.991389847, "V14": -0.311169354, "V15": 1.468176972, "V16": -0.470400525,
        "V17": 0.207971242, "V18": 0.02579058, "V19": 0.40399296, "V20": 0.251412098,
        "V21": -0.018306778, "V22": 0.277837576, "V23": -0.11047391, "V24": 0.066928075,
        "V25": 0.128539358, "V26": -0.189114844, "V27": 0.133558377, "V28": -0.021053053,
        "Amount": 149.62
    }

    return {
        "Legitimate Transaction Example": legit_sample,
        "Confirmed Fraudulent Example": fraud_sample,
    }
