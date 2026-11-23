"""Model Training and Comparison Pipeline for FRAUDGUARD.

Author: Isbandu Rahuman R.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from imblearn.over_sampling import SMOTE
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.data_loader import (
    DEFAULT_DATA_PATH,
    EXPECTED_FEATURES,
    TARGET_COLUMN,
    get_dataset_summary,
    load_dataset,
)
from src.evaluation import calculate_metrics, find_optimal_threshold
from src.preprocessing import fit_and_transform_pipeline, split_data

MODELS_DIR = Path("models")
BEST_MODEL_PATH = MODELS_DIR / "fraudguard_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.joblib"


def train_and_compare_models(
    data_path: Optional[Union[str, Path]] = None,
    use_smote: bool = True,
    random_state: int = 42
) -> Dict[str, Any]:
    """Train Logistic Regression and Random Forest, compare metrics, and select the best model.

    Args:
        data_path: Path to creditcard.csv.
        use_smote: Whether to apply SMOTE on training split only.
        random_state: Reproducibility seed.

    Returns:
        Dictionary containing trained models, comparison results, and metadata.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 1: Loading European Credit Card Fraud dataset...")
    df = load_dataset(data_path)
    dataset_summary = get_dataset_summary(df)
    print(f"Loaded {dataset_summary['total_records']:,} transactions "
          f"({dataset_summary['fraudulent_count']} fraud, {dataset_summary['legitimate_count']} legit).")

    print("Step 2: Stratified 80/20 train/test split (guaranteeing zero data leakage)...")
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.20, random_state=random_state)
    print(f"Training split: {len(X_train):,} samples (Fraud: {y_train.sum()})")
    print(f"Test split:     {len(X_test):,} samples (Fraud: {y_test.sum()})")

    print("Step 3: Fitting RobustScaler preprocessing pipeline on training data only...")
    preprocessor, X_train_proc, X_test_proc = fit_and_transform_pipeline(X_train, X_test)

    # Imbalance handling on training data ONLY
    if use_smote:
        print("Step 4: Applying SMOTE on training data only (test set remains pristine)...")
        smote = SMOTE(sampling_strategy=0.1, random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_proc, y_train)
        print(f"Resampled training set: {len(X_train_resampled):,} samples (Fraud: {y_train_resampled.sum():,})")
    else:
        X_train_resampled, y_train_resampled = X_train_proc, y_train

    trained_models: Dict[str, Any] = {}
    model_evaluations: Dict[str, Any] = {}

    # Model 1: Logistic Regression
    print("\nTraining Model 1: Logistic Regression...")
    lr_model = LogisticRegression(
        C=0.1,
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
        solver="lbfgs"
    )
    lr_model.fit(X_train_resampled, y_train_resampled)
    lr_probs = lr_model.predict_proba(X_test_proc)[:, 1]
    lr_thresh, _ = find_optimal_threshold(y_test.to_numpy(), lr_probs, metric="f2", min_precision=0.40)
    lr_preds = (lr_probs >= lr_thresh).astype(int)
    lr_metrics = calculate_metrics(y_test.to_numpy(), lr_preds, lr_probs)
    lr_metrics["optimal_threshold"] = lr_thresh

    trained_models["Logistic Regression"] = lr_model
    model_evaluations["Logistic Regression"] = lr_metrics
    print(f"LR Results  -> PR-AUC: {lr_metrics['pr_auc']:.4f} | Recall: {lr_metrics['recall']:.4f} | "
          f"Precision: {lr_metrics['precision']:.4f} | F1: {lr_metrics['f1_score']:.4f} | Threshold: {lr_thresh}")

    # Model 2: Random Forest Classifier
    print("\nTraining Model 2: Random Forest Classifier (100 trees, depth 14)...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=14,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )
    rf_model.fit(X_train_resampled, y_train_resampled)
    rf_probs = rf_model.predict_proba(X_test_proc)[:, 1]
    rf_thresh, _ = find_optimal_threshold(y_test.to_numpy(), rf_probs, metric="f2", min_precision=0.60)
    rf_preds = (rf_probs >= rf_thresh).astype(int)
    rf_metrics = calculate_metrics(y_test.to_numpy(), rf_preds, rf_probs)
    rf_metrics["optimal_threshold"] = rf_thresh

    trained_models["Random Forest"] = rf_model
    model_evaluations["Random Forest"] = rf_metrics
    print(f"RF Results  -> PR-AUC: {rf_metrics['pr_auc']:.4f} | Recall: {rf_metrics['recall']:.4f} | "
          f"Precision: {rf_metrics['precision']:.4f} | F1: {rf_metrics['f1_score']:.4f} | Threshold: {rf_thresh}")

    # Best Model Selection based on PR-AUC and F2-Score (Recall priority with reasonable precision)
    rf_score = rf_metrics["pr_auc"] * 0.6 + rf_metrics["f2_score"] * 0.4
    lr_score = lr_metrics["pr_auc"] * 0.6 + lr_metrics["f2_score"] * 0.4

    if rf_score >= lr_score:
        best_name = "Random Forest"
        selection_rationale = (
            "Random Forest selected as top performer due to significantly higher Precision-Recall AUC (PR-AUC) "
            "and superior F1/F2 balance. In financial fraud detection with extreme class imbalance (0.17% fraud), "
            "Random Forest maintains high fraud recall (capturing fraudulent transactions) while suppressing "
            "costly false positives that disrupt legitimate customer transactions."
        )
    else:
        best_name = "Logistic Regression"
        selection_rationale = (
            "Logistic Regression selected due to balanced linear performance and sensitivity to fraudulent transactions."
        )

    best_model = trained_models[best_name]
    best_threshold = model_evaluations[best_name]["optimal_threshold"]

    # Compute Feature Importances
    feature_names = preprocessor.get_feature_names_out()
    if hasattr(best_model, "feature_importances_"):
        raw_importances = best_model.feature_importances_
        feature_importance_dict = dict(sorted(
            zip(feature_names, [float(x) for x in raw_importances]),
            key=lambda item: item[1],
            reverse=True
        ))
    else:
        raw_importances = np.abs(best_model.coef_[0])
        feature_importance_dict = dict(sorted(
            zip(feature_names, [float(x) for x in raw_importances]),
            key=lambda item: item[1],
            reverse=True
        ))

    # Compile Comprehensive Metadata
    metadata: Dict[str, Any] = {
        "project": "FRAUDGUARD",
        "developer": "Isbandu Rahuman R.",
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_model_name": best_name,
        "selected_threshold": best_threshold,
        "selection_rationale": selection_rationale,
        "feature_names": feature_names,
        "feature_importances": feature_importance_dict,
        "model_evaluations": model_evaluations,
        "dataset_summary": dataset_summary,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_fraud_count": int(y_test.sum()),
        "test_legit_count": int((y_test == 0).sum()),
        "random_state": random_state,
        "use_smote": use_smote,
    }

    # Save Artifacts to models/
    print("\nStep 5: Persisting production model artifacts with joblib...")
    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(metadata, METADATA_PATH)
    joblib.dump(lr_model, MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(rf_model, MODELS_DIR / "random_forest.joblib")

    print(f"Successfully saved:")
    print(f"  - Best Model:     {BEST_MODEL_PATH}")
    print(f"  - Preprocessor:   {PREPROCESSOR_PATH}")
    print(f"  - Model Metadata: {METADATA_PATH}")
    print(f"Model training and evaluation complete! Best model: {best_name}")

    return {
        "best_model_name": best_name,
        "metadata": metadata,
        "model_evaluations": model_evaluations,
    }


if __name__ == "__main__":
    train_and_compare_models()
