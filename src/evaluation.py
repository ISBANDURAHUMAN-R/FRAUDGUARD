"""Model Evaluation and Metric Calculation Module for FRAUDGUARD.

Author: Isbandu Rahuman R.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics for binary classification.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Binary predictions at chosen threshold.
        y_prob: Predicted probability for class 1 (Fraud).

    Returns:
        Dictionary containing all evaluation metrics.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # ROC and PR curves
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_prob)

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))

    # Subsample curve points for efficient visualization and JSON/joblib serialization
    max_pts = 100
    step_roc = max(1, len(fpr) // max_pts)
    step_pr = max(1, len(precisions) // max_pts)

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2_score": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "roc_curve": {
            "fpr": fpr[::step_roc].tolist(),
            "tpr": tpr[::step_roc].tolist(),
        },
        "pr_curve": {
            "precision": precisions[::step_pr].tolist(),
            "recall": recalls[::step_pr].tolist(),
        },
        "classification_report": classification_report(
            y_true, y_pred, target_names=["Legitimate", "Fraudulent"], output_dict=True
        ),
    }

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f2",
    min_precision: float = 0.50
) -> Tuple[float, float]:
    """Find the optimal probability threshold balancing high recall and reasonable precision.

    Args:
        y_true: True binary labels.
        y_prob: Predicted fraud probabilities.
        metric: 'f1', 'f2' (prioritizes recall), or 'balanced'.
        min_precision: Guardrail minimum precision requirement.

    Returns:
        Tuple of (optimal_threshold, best_score).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    best_threshold = 0.50
    best_score = -1.0

    for p, r, th in zip(precisions[:-1], recalls[:-1], thresholds):
        if p + r == 0:
            continue

        if metric == "f2":
            # Weight recall twice as high as precision
            score = (5 * p * r) / (4 * p + r) if (4 * p + r) > 0 else 0
        elif metric == "f1":
            score = (2 * p * r) / (p + r)
        else:
            # Balanced product
            score = np.sqrt(p * r)

        # Apply constraint if possible
        if p >= min_precision and score > best_score:
            best_score = float(score)
            best_threshold = float(th)

    # Fallback if no threshold met the min_precision constraint
    if best_score < 0:
        for p, r, th in zip(precisions[:-1], recalls[:-1], thresholds):
            score = (5 * p * r) / (4 * p + r) if (4 * p + r) > 0 else 0
            if score > best_score:
                best_score = float(score)
                best_threshold = float(th)

    return round(best_threshold, 4), round(best_score, 4)
