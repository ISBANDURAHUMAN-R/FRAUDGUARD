"""Tests for Inference Engine, Single Prediction, and Bulk CSV Validation.

Author: Isbandu Rahuman R.
"""

import pandas as pd
import pytest

from src.data_loader import validate_dataset
from src.predict import (
    determine_risk_level,
    get_demo_samples,
    load_artifacts,
    predict_bulk,
    predict_single,
)


@pytest.fixture(scope="module")
def artifacts():
    """Load production model artifacts once for all inference tests."""
    return load_artifacts()


def test_single_prediction_output_structure(artifacts):
    """Verify predict_single returns complete schema and valid types."""
    model, preprocessor, metadata = artifacts
    demos = get_demo_samples()
    legit_sample = demos["Legitimate Transaction Example"]

    result = predict_single(
        features_input=legit_sample,
        model=model,
        preprocessor=preprocessor,
        threshold=metadata.get("selected_threshold", 0.62)
    )

    assert isinstance(result["is_fraud"], bool)
    assert result["prediction_label"] in ["Legitimate", "Fraudulent"]
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert 0.0 <= result["fraud_probability_pct"] <= 100.0
    assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert isinstance(result["notable_features"], list)
    assert "disclaimer" in result


def test_demo_sample_discrimination(artifacts):
    """Verify model distinguishes confirmed fraud demo from verified legitimate demo."""
    model, preprocessor, metadata = artifacts
    demos = get_demo_samples()

    legit_res = predict_single(demos["Legitimate Transaction Example"], model, preprocessor)
    fraud_res = predict_single(demos["Confirmed Fraudulent Example"], model, preprocessor)

    # Fraud probability of the fraud vector should be significantly higher
    assert fraud_res["fraud_probability"] > legit_res["fraud_probability"]
    assert fraud_res["risk_level"] in ["HIGH", "CRITICAL"]
    assert legit_res["risk_level"] == "LOW"


def test_invalid_input_handling(artifacts):
    """Verify graceful rejection of missing columns or non-numeric values."""
    model, preprocessor, _ = artifacts

    # Missing Amount
    bad_features = {"Time": 100.0, "V1": 0.5}
    with pytest.raises(ValueError, match="Missing required transaction features"):
        predict_single(bad_features, model, preprocessor)

    # Non-numeric value in input
    demos = get_demo_samples()
    corrupt_features = demos["Legitimate Transaction Example"].copy()
    corrupt_features["V5"] = "INVALID_STRING"
    with pytest.raises(ValueError, match="contains invalid or non-numeric values"):
        predict_single(corrupt_features, model, preprocessor)


def test_bulk_prediction_pipeline(artifacts):
    """Verify batch inference produces correctly annotated DataFrame and summary."""
    model, preprocessor, metadata = artifacts
    demos = get_demo_samples()

    df_test = pd.DataFrame([
        demos["Legitimate Transaction Example"],
        demos["Confirmed Fraudulent Example"]
    ])

    df_scored, summary = predict_bulk(df_test, model, preprocessor)

    assert len(df_scored) == 2
    assert "Fraud_Probability" in df_scored.columns
    assert "Prediction" in df_scored.columns
    assert "Risk_Level" in df_scored.columns

    assert summary["total_analyzed"] == 2
    assert summary["fraud_count"] == 1
    assert summary["legit_count"] == 1
    assert summary["fraud_rate_pct"] == 50.0


def test_csv_validation():
    """Verify CSV validation rules and error reporting."""
    # Valid minimal DataFrame
    demos = get_demo_samples()
    valid_df = pd.DataFrame([demos["Legitimate Transaction Example"]])
    is_valid, errors = validate_dataset(valid_df, require_target=False)
    assert is_valid is True
    assert len(errors) == 0

    # Missing column
    invalid_df = valid_df.drop(columns=["Amount"])
    is_valid, errors = validate_dataset(invalid_df, require_target=False)
    assert is_valid is False
    assert any("Amount" in e for e in errors)

    # Empty DataFrame
    is_valid, errors = validate_dataset(pd.DataFrame(), require_target=False)
    assert is_valid is False
