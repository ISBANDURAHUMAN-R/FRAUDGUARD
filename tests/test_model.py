"""Tests for Model Loading, Training Utilities, and Metric Evaluation.

Author: Isbandu Rahuman R.
"""

import numpy as np
import pytest

from src.evaluation import calculate_metrics, find_optimal_threshold
from src.predict import load_artifacts


def test_load_production_model_artifacts():
    """Verify production model, preprocessor, and metadata load cleanly from models/."""
    model, preprocessor, metadata = load_artifacts()

    assert model is not None
    assert preprocessor is not None
    assert isinstance(metadata, dict)
    assert "best_model_name" in metadata
    assert "selected_threshold" in metadata
    assert "model_evaluations" in metadata
    assert metadata["project"] == "FRAUDGUARD"
    assert metadata["developer"] == "Isbandu Rahuman R."


def test_calculate_metrics():
    """Verify calculation of binary classification metrics."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9])

    metrics = calculate_metrics(y_true, y_pred, y_prob)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0

    cm = metrics["confusion_matrix"]
    assert cm["tp"] == 3
    assert cm["fn"] == 1
    assert cm["fp"] == 1
    assert cm["tn"] == 3


def test_find_optimal_threshold():
    """Verify threshold optimization returns valid probability threshold."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.05, 0.1, 0.15, 0.2, 0.35, 0.45, 0.6, 0.75, 0.85, 0.95])

    thresh, score = find_optimal_threshold(y_true, y_prob, metric="f2", min_precision=0.40)

    assert 0.0 <= thresh <= 1.0
    assert score > 0.0
